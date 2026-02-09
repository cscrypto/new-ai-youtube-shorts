import streamlit as st
import urllib.request
from pytube import YouTube
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
import numpy as np
import librosa
from transformers import pipeline
import whisper
import time

st.set_page_config(page_title="AI YouTube Shorts Generator", layout="wide")
st.title("AI YouTube Shorts Generator 🤖🎬")

# User settings
num_shorts = st.slider("Number of Shorts to generate:", 1, 5, 3)
short_length = st.slider("Length of each Short (seconds):", 5, 30, 15)
caption_fontsize = st.slider("Caption Font Size:", 20, 80, 40)

youtube_url = st.text_input("Paste YouTube video URL:")

if youtube_url:
    shorts_dir = "shorts_ai"
    os.makedirs(shorts_dir, exist_ok=True)

    st.info("Downloading video...")
    try:
        req = urllib.request.Request(youtube_url, headers={'User-Agent': 'Mozilla/5.0'})
        yt = YouTube(req)
        stream = yt.streams.filter(progressive=True, file_extension='mp4')\
                           .order_by('resolution').desc().first()
        video_path = stream.download(filename="video.mp4")
        st.success("Video downloaded!")
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        st.stop()

    video_clip = VideoFileClip(video_path)
    audio_path = "audio.wav"
    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

    st.info("Loading emotion recognition model...")
    emotion_recognizer = pipeline("audio-classification", model="superb/wav2vec2-large-superb-er")

    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    st.info("Analyzing audio for highlights...")
    chunk_size = 3
    scores = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_chunks = int(np.ceil(duration / chunk_size))

    for i, start in enumerate(np.arange(0, duration, chunk_size)):
        chunk, _ = librosa.load(audio_path, sr=sr, offset=start, duration=chunk_size)
        chunk = chunk.astype(np.float32)
        try:
            result = emotion_recognizer(chunk, chunk_length=chunk_size)
            score = max([r['score'] for r in result])
        except:
            score = 0
        scores.append((start, score))
        progress_text.text(f"Analyzing chunk {i+1}/{total_chunks}...")
        progress_bar.progress((i+1)/total_chunks)
        time.sleep(0.01)

    top_segments = sorted(scores, key=lambda x: x[1], reverse=True)[:num_shorts]
    st.success(f"Top {num_shorts} highlights detected!")

    st.info("Loading Whisper model...")
    whisper_model = whisper.load_model("tiny")

    st.info("Creating Shorts...")
    progress_bar_shorts = st.progress(0)
    progress_text_shorts = st.empty()

    for idx, (start, _) in enumerate(top_segments):
        progress_text_shorts.text(f"Creating Short {idx+1}/{num_shorts}...")
        end = min(start + short_length, video_clip.duration)
        clip = video_clip.subclip(start, end)

        clip = clip.resize(height=1080)

        segment_audio = f"segment_{idx+1}.wav"
        clip.audio.write_audiofile(segment_audio, codec='pcm_s16le')
        result = whisper_model.transcribe(segment_audio)
        captions = result["text"]

        txt_clip = TextClip(
            captions,
            fontsize=caption_fontsize,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(clip.w * 0.9, None)
        ).set_position(('center', 'bottom')).set_duration(clip.duration)

        final_clip = CompositeVideoClip([clip, txt_clip])

        output_path = os.path.join(shorts_dir, f"short_ai_{idx+1}.mp4")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        clip.close()
        final_clip.close()

        st.subheader(f"Short {idx+1}")
        st.video(output_path)
        st.caption(f"Captions preview: {captions[:200]}...")
        st.download_button(
            label=f"Download Short {idx+1}",
            data=open(output_path, "rb"),
            file_name=f"short_ai_{idx+1}.mp4"
        )

        progress_bar_shorts.progress((idx+1)/num_shorts)
        time.sleep(0.01)

    st.success("All Shorts ready! 🎉")
