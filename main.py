import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch
from dotenv import load_dotenv
import os
import openai
import streamlit as st
import nltk
from pytorch_pretrained_biggan import (
    BigGAN,
    one_hot_from_names,
    truncated_noise_sample,
)
import numpy as np
from PIL import Image

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


VISIONCHAT_API_KEY = os.getenv("visionchat_api")
nltk.download('wordnet')

@st.cache_resource
def load_model_and_tokenizer(target_language):
    model_name = f'Helsinki-NLP/opus-mt-en-{target_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(input_text, tokenizer, model):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tokenized_text = tokenizer.tokenize(input_text)
    return translated_text, tokenized_text

def Language_Translation_using_Transformers():
    st.title("Language Translation Using Transformer")

    st.sidebar.header("Configuration")
    
    language_codes = {
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it"
    }
    target_language = st.sidebar.selectbox(
        "Choose Target Language", 
        list(language_codes.keys())
    )
    target_language_code = language_codes[target_language]

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    input_text = st.text_area(
        "Enter text to translate:", 
        value=st.session_state.input_text, 
        placeholder="Type a sentence in English..."
    )

    st.session_state.input_text = input_text

    if st.button("Translate"):
        if input_text.strip():
            tokenizer, model = load_model_and_tokenizer(target_language_code)
            with st.spinner("Translating..."):
                translated_text, tokenized_text = translate_text(input_text, tokenizer, model)
            st.success("Translation Complete!")
            st.write("### Translated Text:")
            st.write(translated_text)
            # st.write("### Tokenized Text:")
            # st.write(tokenized_text)
        else:
            st.error("Please enter some text to translate.")

#GANS
def Image_Generation_using_GAN():
    # Load pre-trained BigGAN model
    @st.cache_resource
    def load_biggan_model():
        return BigGAN.from_pretrained("biggan-deep-512")

    model = load_biggan_model()

    valid_class_names = [
        "goldfish, Carassius auratus",
        "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
        "cock",
        "hen",
        "ostrich, Struthio camelus",
        "peacock",
        "quail",
        "African grey, African gray, Psittacus erithacus",
        "German shepherd, German shepherd dog, German police dog, alsatian",
        "pug, pug-dog",
        "dalmatian, coach dog, carriage dog",
    ]

    st.title("BigGAN Image Generator")
    st.write("Enter a class name to generate an image using the BigGAN model.")

    st.subheader("Valid Class Names")
    st.write(", ".join(valid_class_names))

    class_name = st.text_input("Class Name", value="speedboat", help="Enter a valid class name for the BigGAN model.")
    truncation = 0.4

    if st.button("Generate Image"):
        try:
            class_vector = one_hot_from_names([class_name], batch_size=1)
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noise_vector = noise_vector.to(device)
            class_vector = class_vector.to(device)
            model.to(device)

            with torch.no_grad():
                output = model(noise_vector, class_vector, truncation)

            output = output.cpu().numpy()
            img = ((output[0] + 1) / 2 * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  
            img = Image.fromarray(img)

            st.image(img, caption=f"Generated Image for '{class_name}'", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Ensure the class name is valid for BigGAN.")
    st.markdown(
        """
        ---
        Built with [Streamlit](https://streamlit.io) and [PyTorch](https://pytorch.org).
        """
    )
    
def main():
    st.sidebar.title("Choose a Functionality")
    st.sidebar.write("Select an option below to explore different features of the app:")
    
    options = [
        "Language Translation using Transformers",
        "Image Generation using GAN",
    ]
    choice = st.sidebar.radio("Select an option:", options)

    if choice == "Language Translation using Transformers":
        Language_Translation_using_Transformers()
    elif choice == "Image Generation using GAN":
        Image_Generation_using_GAN()
    
if __name__ == "__main__":
    main()
