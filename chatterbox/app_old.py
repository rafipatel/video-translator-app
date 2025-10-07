from fastapi import FastAPI, UploadFile, File, Form
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Patch torch.load to force CPU mapping
torch_load_orig = torch.load
def torch_load_cpu(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return torch_load_orig(*args, **kwargs)
torch.load = torch_load_cpu

# Lazy load model variable
model = None

# Init FastAPI
app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    if model is None:
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        logging.info("✅ Models loaded successfully!")

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy"}

@app.post("/generate-audio")
async def generate_audio(
    file: UploadFile = File(...),  # Audio file upload instead of path
    segments: str = Form(...),  # JSON string of segments (list of dict)
    silence_duration: float = Form(0.5),
):
    """
    Generate and concatenate TTS audio.
    `segments` must be a JSON string like:
    [
        {"translated_text": "नमस्ते", "start": 0.0, "end": 2.0},
        {"translated_text": "आप कैसे हैं?", "start": 2.5, "end": 5.0}
    ]
    """
    global model
    if model is None:
        return JSONResponse(content={"error": "Model not loaded yet."}, status_code=503)
    try:
        import json
        segments = json.loads(segments)

        if not isinstance(segments, list) or len(segments) == 0:
            return JSONResponse(content={"error": "Segments must be a non-empty list."}, status_code=400)
        
        all_wavs = []
    
        # Create silence tensor
        silence_samples = int(silence_duration * model.sr)
        silence = torch.zeros(1, silence_samples)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        try:
            for counter, segment in enumerate(segments):
                logging.info(f"Synthesizing line {counter+1}: {segment['translated_text']}")
                
                # Calculate original segment duration
                original_duration = segment['end'] - segment['start']
                
                # Generate audio
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
                
                # Calculate generated audio duration
                generated_duration = wav.shape[-1] / model.sr
                
                # Calculate speed factor to match original duration
                speed_factor = generated_duration / original_duration
                
                # Apply speed adjustment if needed
                if speed_factor > 1.1 or speed_factor < 0.9:  # Only adjust if difference is significant
                    logging.info(f"  Original duration: {original_duration:.2f}s, Generated: {generated_duration:.2f}s")
                    logging.info(f"  Applying speed factor: {speed_factor:.2f}x")
                    
                    # Use torchaudio's speed transformation
                    speed_transform = transforms.Speed(model.sr, speed_factor)
                    wav_adjusted, _ = speed_transform(wav)  # Speed returns (waveform, sample_rate)
                    
                    # Verify new duration
                    new_duration = wav_adjusted.shape[-1] / model.sr
                    logging.info(f"  Adjusted duration: {new_duration:.2f}s")
                    
                    all_wavs.append(wav_adjusted)
                else:
                    logging.info(f"  Duration close enough ({generated_duration:.2f}s vs {original_duration:.2f}s), no speed adjustment needed")
                    all_wavs.append(wav)
                
                # Add silence between segments (except after the last one)
                if counter < len(segments) - 1:
                    all_wavs.append(silence)
            
            # Concatenate all audio
            combined_wav = torch.cat(all_wavs, dim=-1)

            # Save to temporary output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
                output_path = temp_output.name

            ta.save(output_path, combined_wav, model.sr)

        finally:
            # Cleanup uploaded temp audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        total_duration = combined_wav.shape[-1] / model.sr
        logging.info(f"Combined audio with silence saved as: {output_path}")
        logging.info(f"Total duration: {total_duration:.2f}s")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_audio.wav",
            background=BackgroundTask(lambda: os.unlink(output_path))
        )

    except Exception as e:
        logging.exception("Error in generate_audio")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)