import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os
import wave

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="AI Voice Studio", page_icon="🎙️", layout="centered")

# Custom CSS to make it look clean (ElevenLabs style)
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 18px !important;
        line-height: 1.5;
        border-radius: 10px;
        border: 1px solid #ddd;
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
    /* Hide the top colored bar */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---

def split_text_into_chunks(text, max_chars=250):
    """Splits long text into safe chunks for the API"""
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
    """Merges multiple WAV files into one"""
    with wave.open(output_filename, 'wb') as wav_out:
        for i, wav_path in enumerate(file_list):
            with wave.open(wav_path, 'rb') as wav_in:
                if i == 0:
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

# --- UI LAYOUT ---

st.title("🎙️ AI Voice Studio")

# 1. Main Input Area
text_input = st.text_area(
    "Script", 
    height=300, 
    placeholder="Type or paste your text here (up to 20,000 characters)...",
    label_visibility="collapsed"
)

# Character counter
char_count = len(text_input)
st.caption(f"{char_count} / 20,000 characters")

# 2. Hidden Settings (Accordion style to keep it clean)
with st.expander("⚙️ Voice Settings & Reference Audio"):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Clone Voice (Optional)", type=['wav', 'mp3'])
    with col2:
        exaggeration = st.slider("Stability/Exaggeration", 0.0, 1.0, 0.5)
        temperature = st.slider("Clarity/Temperature", 0.0, 1.0, 0.8)

# 3. Generation Logic
if st.button("Generate Speech"):
    if not text_input:
        st.warning("Please enter some text first.")
    else:
        # Create a progress bar
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            client = Client("ResembleAI/Chatterbox")
            
            # Prepare chunks
            chunks = split_text_into_chunks(text_input)
            total_chunks = len(chunks)
            temp_files = []
            
            # Prepare Voice Reference
            audio_input = None
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                    tmp_ref.write(uploaded_file.getvalue())
                    audio_input = handle_file(tmp_ref.name)
            else:
                audio_input = handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')

            # Loop through chunks
            for index, chunk in enumerate(chunks):
                if not chunk.strip(): 
                    continue
                    
                progress_bar.progress((index / total_chunks), text=f"Converting part {index+1} of {total_chunks}...")
                
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

            # Merge files
            progress_bar.progress(0.9, text="Stitching audio together...")
            
            final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            merge_wav_files(temp_files, final_output)
            
            progress_bar.empty() # Remove progress bar
            
            # --- SUCCESS UI ---
            st.success("Generation Complete!")
            
            # Audio Player
            st.audio(final_output)
            
            # Download Button (Standard Streamlit button, styled by CSS above)
            with open(final_output, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Download Audio (WAV)",
                    data=file,
                    file_name="generated_speech.wav",
                    mime="audio/wav"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Note: If the text is extremely long, the free API might time out.")
