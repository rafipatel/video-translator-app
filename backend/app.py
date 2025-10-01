from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip
import whisperx
import torch
import torchaudio as ta
import torchaudio.transforms as transforms
from googletrans import Translator
from chatterbox import ChatterboxMultilingualTTS
import asyncio
import nest_asyncio

nest_asyncio.apply()

app = FastAPI(title="Video Voice Cloner & Translator")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
device = "cpu"
compute_type = "int8"
whisper_model = None
tts_model = None

@app.on_event("startup")
async def load_models():
    """Load models on startup to avoid reloading"""
    global whisper_model, tts_model
    print("Loading WhisperX model...")
    whisper_model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    print("Loading TTS model...")
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    print("Models loaded successfully!")

# ==================== Helper Functions ====================

def audio_extractor(video_path):
    """Extract audio from video"""
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    full_audio_path = temp_file.name
    temp_file.close()
    
    audio_clip.write_audiofile(full_audio_path, codec='pcm_s16le', logger=None)
    audio_clip.close()
    video_clip.close()
    return full_audio_path

def transcribe(full_audio_path):
    """Transcribe audio with speaker diarization"""
    audio = whisperx.load_audio(full_audio_path)
    
    # Transcribe
    result = whisper_model.transcribe(audio, batch_size=16)
    detected_language = result.get("language", "en")
    
    # Align
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Diarization
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(full_audio_path)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    result['language_code'] = detected_language
    return result

async def translate_segments(segments, target_lang="hi"):
    """Translate segments to target language"""
    results = []
    translator = Translator()
    
    for seg in segments:
        clean_seg = {k: v for k, v in seg.items() if k != "words"}
        result = translator.translate(clean_seg["text"], dest=target_lang)
        clean_seg["translated_text"] = result.text
        results.append(clean_seg)
    
    return results

def concatenate_audio_with_silence(segments, audio_prompt_path, silence_duration=0.5, output_path="output.wav"):
    """Generate and concatenate TTS audio"""
    all_wavs = []
    silence_samples = int(silence_duration * tts_model.sr)
    silence = torch.zeros(1, silence_samples)
    
    for i, segment in enumerate(segments):
        print(f"Synthesizing {i+1}/{len(segments)}: {segment['translated_text']}")
        
        original_duration = segment['end'] - segment['start']
        
        wav = tts_model.generate(
            segment['translated_text'],
            "hi",
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.2,
            cfg_weight=0.8,
            temperature=0.4,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=0.9
        )
        
        generated_duration = wav.shape[-1] / tts_model.sr
        speed_factor = generated_duration / original_duration
        
        if speed_factor > 1.1 or speed_factor < 0.9:
            speed_transform = transforms.Speed(tts_model.sr, speed_factor)
            wav_adjusted, _ = speed_transform(wav)
            all_wavs.append(wav_adjusted)
        else:
            all_wavs.append(wav)
        
        if i < len(segments) - 1:
            all_wavs.append(silence)
    
    combined_wav = torch.cat(all_wavs, dim=-1)
    ta.save(output_path, combined_wav, tts_model.sr)
    return combined_wav

def replace_video_audio(video_path, new_audio_path, output_video_path):
    """Replace video audio"""
    video_clip = VideoFileClip(video_path)
    new_audio_clip = AudioFileClip(new_audio_path)
    
    video_duration = video_clip.duration
    audio_duration = new_audio_clip.duration
    
    if audio_duration < video_duration:
        final_video = video_clip.subclip(0, audio_duration)
        final_audio = new_audio_clip
    elif audio_duration > video_duration:
        final_video = video_clip
        final_audio = new_audio_clip.subclip(0, video_duration)
    else:
        final_video = video_clip
        final_audio = new_audio_clip
    
    final_clip = final_video.set_audio(final_audio)
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', logger=None)
    
    video_clip.close()
    new_audio_clip.close()
    final_audio.close()
    final_video.close()
    final_clip.close()

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {"message": "Video Voice Cloner & Translator API", "status": "running"}

@app.post("/translate-video")
async def translate_video(
    video: UploadFile = File(...),
    target_language: str = Form("hi")
):
    """
    Complete pipeline: Upload video → Translate → Return translated video
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded video
        input_video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # Step 1: Extract audio
        print("Extracting audio...")
        audio_path = audio_extractor(input_video_path)
        
        # Step 2: Transcribe
        print("Transcribing...")
        transcription = transcribe(audio_path)
        
        # Step 3: Translate
        print("Translating...")
        translated_segments = await translate_segments(transcription['segments'], target_language)
        
        # Step 4: Generate TTS
        print("Generating TTS...")
        output_audio_path = os.path.join(temp_dir, "translated_audio.wav")
        concatenate_audio_with_silence(translated_segments, audio_path, output_path=output_audio_path)
        
        # Step 5: Replace audio in video
        print("Replacing audio in video...")
        output_video_path = os.path.join(temp_dir, "translated_video.mp4")
        replace_video_audio(input_video_path, output_audio_path, output_video_path)
        
        # Return the translated video
        return FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename="translated_video.mp4"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup happens after response is sent
        pass

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "tts_loaded": tts_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)