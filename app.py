import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os
import wave
import time  # <--- Added for the trick

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="AI Voice Studio", page_icon="🎙️", layout="centered")

# Custom CSS for Minimal UI
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 25px;
        height: 50px;
        font-weight: bold;
        background-color: #000000;
        color: white;
    }
    .stButton button:hover {
        background-color: #333333;
        color: white;
    }
    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---

def split_text_into_chunks(text, max_chars=250):
    """Splits text carefully to avoid cutting words"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def merge_wav_files(file_list, output_filename):
    """Merges the audio parts"""
    if not file_list: return
    with wave.open(output_filename, 'wb') as wav_out:
        for i, wav_path in enumerate(file_list):
            try:
                with wave.open(wav_path, 'rb') as wav_in:
                    if i == 0:
                        wav_out.setparams(wav_in.getparams())
                    wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            except Exception as e:
                print(f"Error merging file {wav_path}: {e}")

# --- UI LAYOUT ---

st.title("🎙️ AI Voice Studio (Unlimited Mode)")
st.info("💡 Tip: For 20k characters, this will take time. We added a 'Sleep Timer' to prevent server bans.")

# 1. Input
text_input = st.text_area("Script", height=300, placeholder="Paste your long text here...")
st.caption(f"{len(text_input)} characters")

# 2. Settings
with st.expander("⚙️ Settings"):
    uploaded_file = st.file_uploader("Clone Voice (WAV/MP3)", type=['wav', 'mp3'])
    exaggeration = st.slider("Stability", 0.0, 1.0, 0.5)
    temperature = st.slider("Clarity", 0.0, 1.0, 0.8)

# 3. Logic
if st.button("Generate Long Speech"):
    if not text_input:
        st.warning("No text found!")
    else:
        progress_bar = st.progress(0, text="Starting...")
        status_area = st.empty()
        
        try:
            # Initialize Client
            client = Client("ResembleAI/Chatterbox")
            
            # Setup Audio Prompt
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                    tmp_ref.write(uploaded_file.getvalue())
                    audio_input = handle_file(tmp_ref.name)
            else:
                audio_input = handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')

            chunks = split_text_into_chunks(text_input)
            total = len(chunks)
            temp_files = []

            # --- THE LOOP WITH THE TRICK ---
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                
                # Retry logic: Try 3 times before giving up
                success = False
                attempts = 0
                
                while not success and attempts < 3:
                    try:
                        status_area.text(f"Processing part {i+1}/{total} (Attempt {attempts+1})...")
                        
                        result_path = client.predict(
                            text_input=chunk,
                            audio_prompt_path_input=audio_input,
                            exaggeration_input=exaggeration,
                            temperature_input=temperature,
                            seed_num_input=0,
                            cfgw_input=0.5,
                            vad_trim_input=True,
                            api_name="/generate_tts_audio"
                        )
                        temp_files.append(result_path)
                        success = True
                        
                        # TRICK 1: Wait 15 seconds between successful requests to cool down GPU
                        time.sleep(15) 

                    except Exception as e:
                        error_msg = str(e).lower()
                        if "quota" in error_msg or "exceeded" in error_msg:
                            # TRICK 2: If we hit the limit, wait 60 seconds then try again
                            status_area.warning(f"Quota hit! Pausing for 60 seconds to cooldown... (Part {i+1})")
                            time.sleep(60)
                            attempts += 1
                        else:
                            status_area.error(f"Error: {e}")
                            break # Unknown error, stop trying

                progress_bar.progress((i + 1) / total)

            # --- FINISH ---
            if temp_files:
                status_area.text("Stitching audio files...")
                final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                merge_wav_files(temp_files, final_output)
                
                st.success("Generation Complete!")
                st.audio(final_output)
                
                with open(final_output, "rb") as f:
                    st.download_button("⬇️ Download Final Audio", f, "speech.wav", "audio/wav")
            else:
                st.error("Failed to generate audio.")

        except Exception as e:
            st.error(f"Critical Error: {e}")
