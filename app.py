import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os

# 1. App Title and Description
st.set_page_config(page_title="Chatterbox TTS", page_icon="🗣️")
st.title("🗣️ Free Chatterbox Text-to-Speech")
st.write("This app uses the ResembleAI/Chatterbox model to generate audio.")

# 2. Input Widgets (The User Interface)
st.subheader("1. Enter Text")
text_input = st.text_area(
    "Text to synthesize", 
    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit."
)

st.subheader("2. Reference Audio (Voice Clone)")
st.info("Upload a short audio clip (WAV/MP3) to clone the voice. If you don't upload one, a default voice is used.")
uploaded_file = st.file_uploader("Upload Audio Reference", type=['wav', 'mp3'])

st.subheader("3. Settings")
col1, col2 = st.columns(2)

with col1:
    exaggeration = st.slider("Exaggeration", min_value=0.0, max_value=1.0, value=0.5, help="Neutral = 0.5")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.8)
    
with col2:
    cfgw = st.slider("CFG/Pace", min_value=0.0, max_value=1.0, value=0.5)
    seed = st.number_input("Random Seed (0 = Random)", value=0, step=1)

vad_trim = st.checkbox("Trim Silence (VAD)", value=False)

# 3. The Logic (Sending data to the API)
if st.button("Generate Audio", type="primary"):
    if not text_input:
        st.error("Please enter some text.")
    else:
        status_text = st.empty()
        status_text.text("Connecting to AI model... please wait...")
        
        try:
            # Initialize the client
            client = Client("ResembleAI/Chatterbox")
            
            # Handle the audio file
            # If user uploaded a file, we must save it temporarily so Gradio can read it
            if uploaded_file is not None:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                # prepare the file for the API
                audio_input = handle_file(audio_path)
            else:
                # Use the default sample if no file provided
                audio_input = handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')

            status_text.text("Generating audio... (This might take 10-20 seconds)")

            # Call the API exactly as documented
            result_path = client.predict(
                text_input=text_input,
                audio_prompt_path_input=audio_input,
                exaggeration_input=exaggeration,
                temperature_input=temperature,
                seed_num_input=seed,
                cfgw_input=cfgw,
                vad_trim_input=vad_trim,
                api_name="/generate_tts_audio"
            )
            
            # Display the result
            status_text.text("Done!")
            st.success("Audio Generated successfully!")
            st.audio(result_path)
            
            # Cleanup temp file if it exists
            if uploaded_file is not None and os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            status_text.empty()
