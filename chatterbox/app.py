from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import torch
import torchaudio as ta
import torchaudio.transforms as transforms
from chatterbox import ChatterboxMultilingualTTS
import tempfile
import os
import uvicorn
import logging
from starlette.background import BackgroundTask
import uuid
from typing import Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Patch torch.load to force CPU mapping
torch_load_orig = torch.load
def torch_load_cpu(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return torch_load_orig(*args, **kwargs)
torch.load = torch_load_cpu

# Load model
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
logging.info("✅ Models loaded successfully!")

# Init FastAPI
app = FastAPI()

# In-memory job storage (use Redis in production)
jobs: Dict[str, dict] = {}

def process_audio_task(job_id: str, temp_audio_path: str, segments: list, silence_duration: float):
    """Background task to process audio"""
    try:
        jobs[job_id]["status"] = "processing"
        
        all_wavs = []
        silence_samples = int(silence_duration * model.sr)
        silence = torch.zeros(1, silence_samples)
        
        total_segments = len(segments)
        
        for counter, segment in enumerate(segments):
            logging.info(f"Job {job_id}: Synthesizing line {counter+1}/{total_segments}: {segment['translated_text']}")
            
            # Update progress
            jobs[job_id]["progress"] = f"{counter + 1}/{total_segments}"
            
            original_duration = segment['end'] - segment['start']
            
            wav = model.generate(
                segment['translated_text'], 
                "hi", 
                audio_prompt_path=temp_audio_path, 
                exaggeration=0.2,
                cfg_weight=0.8,
                temperature=0.4,
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=0.9
            )
            
            generated_duration = wav.shape[-1] / model.sr
            speed_factor = generated_duration / original_duration
            
            if speed_factor > 1.1 or speed_factor < 0.9:
                logging.info(f"  Applying speed factor: {speed_factor:.2f}x")
                speed_transform = transforms.Speed(model.sr, speed_factor)
                wav_adjusted, _ = speed_transform(wav)
                all_wavs.append(wav_adjusted)
            else:
                all_wavs.append(wav)
            
            if counter < len(segments) - 1:
                all_wavs.append(silence)
        
        # Save output
        output_path = f"/tmp/audio_{job_id}.wav"
        combined_wav = torch.cat(all_wavs, dim=-1)
        ta.save(output_path, combined_wav, model.sr)
        
        total_duration = combined_wav.shape[-1] / model.sr
        logging.info(f"Job {job_id}: Completed! Total duration: {total_duration:.2f}s")
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = output_path
        jobs[job_id]["duration"] = total_duration
        jobs[job_id]["progress"] = f"{total_segments}/{total_segments}"
        
        # Clean up temp audio
        try:
            os.unlink(temp_audio_path)
        except:
            pass
            
    except Exception as e:
        logging.exception(f"Job {job_id}: Error processing audio")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/generate-audio-async")
async def generate_audio_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    segments: str = Form(...),
    silence_duration: float = Form(0.5),
):
    """
    Submit a job to generate audio asynchronously.
    Returns a job_id to check status and download result.
    """
    try:
        segments_list = json.loads(segments)
        
        if not isinstance(segments_list, list) or len(segments_list) == 0:
            return JSONResponse(content={"error": "Segments must be a non-empty list."}, status_code=400)
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "queued",
            "progress": "0/0",
            "created_at": os.path.getmtime(temp_audio_path)
        }
        
        # Start background task
        background_tasks.add_task(
            process_audio_task,
            job_id,
            temp_audio_path,
            segments_list,
            silence_duration
        )
        
        logging.info(f"Created job {job_id} with {len(segments_list)} segments")
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "queued",
            "message": f"Processing {len(segments_list)} segments. Use /job/{job_id} to check status."
        })
        
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON in segments"}, status_code=400)
    except Exception as e:
        logging.exception("Error creating job")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a job"""
    if job_id not in jobs:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", "0/0")
    }
    
    if job["status"] == "completed":
        response["download_url"] = f"/download/{job_id}"
        response["duration"] = job.get("duration")
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    
    return JSONResponse(content=response)


@app.get("/download/{job_id}")
async def download_audio(job_id: str):
    """Download the generated audio file"""
    if job_id not in jobs:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        return JSONResponse(
            content={"error": f"Job status is '{job['status']}', not completed"},
            status_code=400
        )
    
    output_path = job["output_path"]
    
    if not os.path.exists(output_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="generated_audio.wav",
        background=BackgroundTask(lambda: cleanup_job(job_id))
    )


def cleanup_job(job_id: str):
    """Clean up job files and data"""
    if job_id in jobs:
        try:
            output_path = jobs[job_id].get("output_path")
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass
        
        # Remove job from memory after 1 hour (in production, use TTL in Redis)
        # For now, just keep it
        jobs[job_id]["status"] = "downloaded"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_jobs": len([j for j in jobs.values() if j["status"] in ["queued", "processing"]])}


# Keep the synchronous endpoint for backward compatibility (but not recommended for long jobs)
@app.post("/generate-audio")
async def generate_audio(
    file: UploadFile = File(...),
    segments: str = Form(...),
    silence_duration: float = Form(0.5),
):
    """
    Synchronous audio generation (not recommended for multiple segments).
    Use /generate-audio-async instead for better reliability.
    """
    try:
        segments_list = json.loads(segments)
        
        if len(segments_list) > 2:
            return JSONResponse(
                content={
                    "error": "Too many segments for synchronous processing. Use /generate-audio-async instead.",
                    "suggestion": "POST to /generate-audio-async, then poll /job/{job_id} and download from /download/{job_id}"
                },
                status_code=400
            )
        
        # ... rest of your original code for short jobs ...
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        all_wavs = []
        silence_samples = int(silence_duration * model.sr)
        silence = torch.zeros(1, silence_samples)
        
        for counter, segment in enumerate(segments_list):
            original_duration = segment['end'] - segment['start']
            
            wav = model.generate(
                segment['translated_text'], 
                "hi", 
                audio_prompt_path=temp_audio_path, 
                exaggeration=0.2,
                cfg_weight=0.8,
                temperature=0.4,
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=0.9
            )
            
            generated_duration = wav.shape[-1] / model.sr
            speed_factor = generated_duration / original_duration
            
            if speed_factor > 1.1 or speed_factor < 0.9:
                speed_transform = transforms.Speed(model.sr, speed_factor)
                wav_adjusted, _ = speed_transform(wav)
                all_wavs.append(wav_adjusted)
            else:
                all_wavs.append(wav)
            
            if counter < len(segments_list) - 1:
                all_wavs.append(silence)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            output_path = temp_output.name

        combined_wav = torch.cat(all_wavs, dim=-1)
        ta.save(output_path, combined_wav, model.sr)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_audio.wav",
            background=BackgroundTask(lambda: os.unlink(output_path))
        )

    except Exception as e:
        logging.exception("Error in generate_audio")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# if __name__ == "__main__":
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=4000,
#         timeout_keep_alive=300
#     )