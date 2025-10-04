import streamlit as st
import tempfile
import os
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip
import whisperx
import torch
import requests
import json
import time
import logging
from typing import List, Dict
from pathlib import Path
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Video Voice Translator",
    page_icon="🎬",
    layout="wide"
)

# Configuration
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:4000")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "600"))
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# ==================== Model Loading ====================

@st.cache_resource
def load_whisper_model():
    """Load WhisperX model (cached)"""
    with st.spinner("Loading WhisperX model..."):
        model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
    return model

def check_tts_service():
    """Check if TTS service is available"""
    try:
        response = requests.get(f"{TTS_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# ==================== TTS Service Client ====================

class TTSServiceError(Exception):
    pass

def generate_translated_audio_via_service(
    reference_audio_path: str,
    segments: List[Dict],
    output_path: str,
    progress_bar,
    status_text,
    silence_duration: float = 0.5,
    poll_interval: int = 2
) -> str:
    """Generate translated audio using remote TTS service with progress updates"""
    
    status_text.text(f"Submitting TTS job for {len(segments)} segments...")
    
    # Submit job
    try:
        with open(reference_audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {
                "segments": json.dumps(segments, ensure_ascii=False),
                "silence_duration": str(silence_duration)
            }
            
            response = requests.post(
                f"{TTS_SERVICE_URL}/generate-audio-async",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise TTSServiceError(f"Failed to submit: {response.text}")
            
            job_data = response.json()
            job_id = job_data["job_id"]
            status_text.text(f"Job submitted: {job_id}")
    
    except requests.RequestException as e:
        raise TTSServiceError(f"Network error: {e}")
    
    # Poll for completion
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TTS_TIMEOUT:
            raise TimeoutError(f"Job exceeded {TTS_TIMEOUT}s timeout")
        
        try:
            status_response = requests.get(f"{TTS_SERVICE_URL}/job/{job_id}", timeout=10)
            
            if status_response.status_code == 200:
                status = status_response.json()
                current_progress = status.get('progress', 'N/A')
                
                # Update progress bar
                if '/' in str(current_progress):
                    current, total = map(int, current_progress.split('/'))
                    progress_bar.progress(current / total if total > 0 else 0)
                
                status_text.text(f"Status: {status['status']} | Progress: {current_progress}")
                
                if status["status"] == "completed":
                    status_text.text("TTS generation completed!")
                    break
                
                elif status["status"] == "failed":
                    raise TTSServiceError(f"Job failed: {status.get('error')}")
            
            time.sleep(poll_interval)
        
        except requests.RequestException:
            time.sleep(poll_interval)
    
    # Download result
    status_text.text("Downloading generated audio...")
    
    try:
        download_response = requests.get(
            f"{TTS_SERVICE_URL}/download/{job_id}",
            stream=True,
            timeout=300
        )
        
        if download_response.status_code != 200:
            raise TTSServiceError("Failed to download audio")
        
        with open(output_path, "wb") as f:
            f.write(download_response.content)
        
        status_text.text("Audio downloaded successfully!")
        return output_path
    
    except requests.RequestException as e:
        raise TTSServiceError(f"Download error: {e}")

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

def transcribe(full_audio_path, whisper_model, progress_callback=None):
    """Transcribe audio with speaker diarization"""
    if progress_callback:
        progress_callback("Loading audio...")
    
    audio = whisperx.load_audio(full_audio_path)
    
    if progress_callback:
        progress_callback("Transcribing...")
    
    result = whisper_model.transcribe(audio, batch_size=16)
    detected_language = result.get("language", "en")
    
    if progress_callback:
        progress_callback("Aligning transcription...")
    
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    
    if progress_callback:
        progress_callback("Performing speaker diarization...")
    
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=None, device=DEVICE)
    diarize_segments = diarize_model(full_audio_path)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    result['language_code'] = detected_language
    return result

def translate_segments(segments: List[Dict], target_lang: str) -> List[Dict]:
    """Translate segments to target language using deep-translator"""
    results = []
    translator = GoogleTranslator(source='auto', target=target_lang)
    for seg in segments:
        clean_seg = {k: v for k, v in seg.items() if k != "words"}
        
        if not clean_seg["text"] or clean_seg["text"].isspace():
            translated_text = ""
        else:
            translated_text = translator.translate(clean_seg["text"])
        
        clean_seg["translated_text"] = translated_text
        results.append(clean_seg)
    return results

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

# ==================== Streamlit UI ====================

def main():
    st.title("🎬 Video Voice Translator")
    st.markdown("Upload a video, and we'll translate it to your target language while preserving the voice!")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Language selection
        target_language = st.selectbox(
            "Target Language",
            options=[
                ("English", "en"),
                ("Hindi", "hi"),
                ("Spanish", "es"),
                ("French", "fr"),
                ("German", "de"),
                ("Italian", "it"),
                ("Portuguese", "pt"),
                ("Russian", "ru"),
                ("Japanese", "ja"),
                ("Korean", "ko"),
                ("Chinese (Simplified)", "zh-cn"),
            ],
            format_func=lambda x: x[0]
        )[1]
        
        st.markdown("---")
        
        # Service status
        st.subheader("Service Status")
        tts_status = check_tts_service()
        st.write(f"TTS Service: {'🟢 Online' if tts_status else '🔴 Offline'}")
        st.write(f"Service URL: `{TTS_SERVICE_URL}`")
        
        if not tts_status:
            st.error("TTS service is not available. Please check the service URL.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - **WhisperX** for transcription
        - **Google Translate** for translation
        - **Chatterbox** for voice cloning Text-to-Speech (TTS)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a video file to translate"
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            st.info(f"File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
    
    with col2:
        st.subheader("⚙️ Processing")
        
        if uploaded_file and st.button("🚀 Start Translation", type="primary", disabled=not tts_status):
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Save uploaded video
                input_video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load model
                status_text.text("Loading WhisperX model...")
                whisper_model = load_whisper_model()
                progress_bar.progress(0.1)
                
                # Step 2: Extract audio
                status_text.text("Extracting audio from video...")
                audio_path = audio_extractor(input_video_path)
                progress_bar.progress(0.2)
                
                # Step 3: Transcribe
                status_text.text("Transcribing audio...")
                transcription = transcribe(
                    audio_path,
                    whisper_model,
                    progress_callback=status_text.text
                )
                progress_bar.progress(0.4)
                st.success(f"Transcribed {len(transcription['segments'])} segments")
                
                # Step 4: Translate
                status_text.text("Translating segments...")
                translated_segments = translate_segments(transcription['segments'], target_language)
                progress_bar.progress(0.6)
                st.success(f"Translated {len(translated_segments)} segments")
                
                # Step 5: Generate TTS
                status_text.text("Generating voice-cloned audio...")
                output_audio_path = os.path.join(temp_dir, "translated_audio.wav")
                
                tts_progress = st.progress(0)
                tts_status = st.empty()
                
                try:    
                    generate_translated_audio_via_service(
                        reference_audio_path=audio_path,
                        segments=translated_segments,
                        output_path=output_audio_path,
                        progress_bar=tts_progress,
                        status_text=tts_status,
                        silence_duration=0.5
                    )
                    progress_bar.progress(0.8)
                    st.success("TTS audio generated successfully!")
                
                except (TTSServiceError, TimeoutError) as e:
                    st.error(f"TTS generation failed: {e}")
                    return
                
                # Step 6: Merge audio with video
                status_text.text("Merging audio with video...")
                output_video_path = os.path.join(temp_dir, "translated_video.mp4")
                replace_video_audio(input_video_path, output_audio_path, output_video_path)
                progress_bar.progress(1.0)
                status_text.text("Translation complete!")
                
                st.success("Video translation completed successfully!")
                
                # Display result
                st.subheader("📥 Download Translated Video")
                
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Translated Video",
                        data=video_bytes,
                        file_name="translated_video.mp4",
                        mime="video/mp4"
                    )
                
                # Preview
                st.video(output_video_path)
                
                # Display transcription
                with st.expander("📝 View Transcription & Translation"):
                    for i, seg in enumerate(translated_segments):
                        st.markdown(f"**Segment {i+1}** ({seg['start']:.2f}s - {seg['end']:.2f}s)")
                        st.markdown(f"*Original:* {transcription['segments'][i]['text']}")
                        st.markdown(f"*Translated:* {seg['translated_text']}")
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception("Error in translation pipeline")
            
            finally:
                # Cleanup
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

if __name__ == "__main__":
    main()