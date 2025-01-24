import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
#import tempfile
# from langchain_community.document_loaders.pdf import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
import torch
# from langchain.embeddings.openai import OpenAIEmbeddings  # Corrected import
# from langchain.document_loaders import PyPDFLoader
# from transformers import T5Tokenizer, T5ForConditionalGeneration 
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI instead of ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import openai
import streamlit as st
# import requests
# import base64

# # import google.generativeai as genai 
# from diffusers import StableDiffusionPipeline
from pytorch_pretrained_biggan import (
    BigGAN,
    one_hot_from_names,
    truncated_noise_sample,
)
import numpy as np
from PIL import Image

load_dotenv()
# GOOGLE API configuration
# os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# NVIDIA API configuration
# invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"
# api_key = os.getenv("NVIDIA_API_KEY")

# OPENAI API configuration
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


VISIONCHAT_API_KEY = os.getenv("visionchat_api")

# Initialize OpenAI client
# client = openai.Completion.create(
#     base_url=API_BASE_URL,
#     api_key=API_KEY
# )




### Transformers
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
            st.write("### Tokenized Text:")
            st.write(tokenized_text)
        else:
            st.error("Please enter some text to translate.")





# ###RAG
# # Fix for OpenMP runtime conflict
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def Chat_with_document_using_RAG():
#     st.title("Chat with Document using RAG")
#     st.write("Chat with your uploaded documents!")

#     uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
#     if uploaded_file:
#         st.success("Document uploaded successfully!")
#         with st.spinner("Processing the document..."):
#             with open("uploaded_document.pdf", "wb") as f:
#                 f.write(uploaded_file.read())

#             # Load the document using PyPDFLoader
#             loader = PyPDFLoader("uploaded_document.pdf")
#             documents = loader.load()

#             # Split the document into chunks
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#             text_chunks = text_splitter.split_documents(documents)

#             # Create embeddings and vector store
#             embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use OpenAI embeddings
#             vector_store = FAISS.from_documents(text_chunks, embeddings)
#             retriever = vector_store.as_retriever(search_kwargs={"k": 2})

#             # Set up memory
#             memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#             # Load the conversational model (using ChatOpenAI)
#             model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

#             # Create the retrieval chain
#             chain = ConversationalRetrievalChain.from_llm(
#                 llm=model,
#                 retriever=retriever,
#                 memory=memory
#             )

#         st.write("You can now ask questions about your document!")
#         user_input = st.text_input("Ask a question about the document:")
#         if st.button("Get Answer"):
#             if user_input.strip():
#                 with st.spinner("Generating answer..."):
#                     # Generate the response using the chain
#                     result = chain.invoke({"question": user_input, "chat_history": []})
#                     st.write("**Answer:**", result["answer"])
#             else:
#                 st.error("Please enter a question.")




# ###NVLM
# def Chat_with_Image_using_NVLM():
#     # Load environment variables
#     load_dotenv()

#     # NVIDIA API configuration
#     invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
#     api_key = os.getenv("NVIDIA_API_KEY")

#     # Streamlit App UI
#     st.title("Image Question-Answering Chat")
#     st.write("Upload an image and ask a question about it!")

#     # Initialize session states
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []  # List of {image_b64, messages} dictionaries
#     if "current_image_b64" not in st.session_state:
#         st.session_state.current_image_b64 = None

#     # Sidebar for image upload
#     with st.sidebar:
#         st.header("Upload Image")
#         uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
#         if uploaded_file:
#             # Convert image to Base64
#             image_b64 = base64.b64encode(uploaded_file.read()).decode()
#             # Check size
#             if len(image_b64) >= 180_000:
#                 st.error("The image is too large! Please upload an image smaller than ~135 KB.")
#             else:
#                 st.session_state.current_image_b64 = image_b64
#                 st.image(uploaded_file, caption="Uploaded Image", width=250)

#     # Main chatbox section
#     st.subheader("Ask Your Question")
#     if st.session_state.current_image_b64:
#         # Option to continue with the previous image or upload a new one
#         if st.session_state.chat_history:
#             use_previous_image = st.checkbox("Continue with the previous image?", value=True)
#             if not use_previous_image:
#                 st.session_state.current_image_b64 = None

#         # Display chat history for the selected image
#         for entry in st.session_state.chat_history:
#             st.image(
#                 base64.b64decode(entry["image_b64"]),
#                 caption="Image for the conversation",
#                 width=150,
#             )
#             for message in entry["messages"]:
#                 if message["role"] == "user":
#                     st.markdown(f"**You:** {message['content']}")
#                 elif message["role"] == "assistant":
#                     st.markdown(f"**Assistant:** {message['content']}")

#         if st.session_state.current_image_b64:
#             # Input field for user question
#             user_input = st.text_input("Enter your question here:", key="chat_input")

#             if st.button("Send"):
#                 if user_input.strip():
#                     # Find or create chat history for the current image
#                     image_chat = next(
#                         (
#                             chat
#                             for chat in st.session_state.chat_history
#                             if chat["image_b64"] == st.session_state.current_image_b64
#                         ),
#                         None,
#                     )
#                     if not image_chat:
#                         image_chat = {
#                             "image_b64": st.session_state.current_image_b64,
#                             "messages": [],
#                         }
#                         st.session_state.chat_history.append(image_chat)

#                     # Add user question to chat history
#                     image_chat["messages"].append({"role": "user", "content": user_input})

#                     # Prepare API request
#                     headers = {
#                         "Authorization": f"Bearer {api_key}",
#                         "Accept": "application/json",
#                     }
#                     payload = {
#                         "messages": [
#                             {
#                                 "role": "user",
#                                 "content": f'{user_input} <img src="data:image/png;base64,{st.session_state.current_image_b64}" />',
#                             }
#                         ],
#                         "max_tokens": 512,
#                         "temperature": 1.00,
#                         "top_p": 0.70,
#                         "stream": False,
#                     }

#                     # Make API call
#                     with st.spinner("Processing your question..."):
#                         try:
#                             response = requests.post(invoke_url, headers=headers, json=payload)
#                             if response.status_code == 200:
#                                 # Extract content from the JSON response
#                                 response_json = response.json()
#                                 assistant_content = response_json.get(
#                                     "choices", [{}]
#                                 )[0].get("message", {}).get("content", "No response received.")

#                                 # Add assistant's response to chat history
#                                 image_chat["messages"].append(
#                                     {"role": "assistant", "content": assistant_content}
#                                 )
#                             else:
#                                 st.error(f"Error {response.status_code}: Unable to process your request.")
#                                 image_chat["messages"].append(
#                                     {
#                                         "role": "assistant",
#                                         "content": "I'm sorry, I couldn't process your request. Please try again!",
#                                     }
#                                 )
#                         except Exception as e:
#                             st.error("An error occurred while processing your request.")
#                             image_chat["messages"].append(
#                                 {
#                                     "role": "assistant",
#                                     "content": "An unexpected error occurred. Please try again later!",
#                                 }
#                             )
#     else:
#         st.info("Please upload an image to start asking questions.")




# ###LLAMA
# def Essay_Generation_using_LLAMA():
#     """
#     Function to generate essays or responses using NVIDIA's LLaMA model via NVIDIA's API.
#     This function implements a Streamlit interface for user interaction.
#     """
#     # Load environment variables
#     load_dotenv()

#     # Get API configuration from .env
#     API_BASE_URL = os.getenv("API_BASE_URL")  # e.g., "https://api.nvidia.com" or another URL
#     API_KEY = os.getenv("NVIDIA_API_KEY")

#     # Check if API credentials are available
#     if not API_BASE_URL or not API_KEY:
#         st.error("API_BASE_URL or NVIDIA_API_KEY is missing. Please check your .env file.")
#         return

#     # Streamlit UI
#     st.title("Chat with NVIDIA's LLaMA Model")
#     st.write("Generate responses using the NVIDIA LLaMA 3.1 Nemotron model.")

#     # User input
#     user_prompt = st.text_area("Enter your prompt:", placeholder="Write a short essay on llamas...", height=150)

#     if st.button("Generate Response"):
#         if not user_prompt.strip():
#             st.error("Please enter a valid prompt.")
#         else:
#             # Placeholder for streaming response
#             response_placeholder = st.empty()

#             # Prepare the API request
#             headers = {
#                 "Authorization": f"Bearer {API_KEY}",
#                 "Content-Type": "application/json"
#             }

#             # Prepare the payload for the request
#             payload = {
#                 "model": "meta/llama-3.2-90b-vision-instruct",
#                 "messages": [{"role": "user", "content": user_prompt}],
#                 "temperature": 0.5,
#                 "top_p": 1,
#                 "max_tokens": 1024,
#                 "stream": True
#             }

#             try:
#                 # Make the API request to NVIDIA's model
#                 with st.spinner("Generating response..."):
#                     response = requests.post(f"{API_BASE_URL}/v1/completions", headers=headers, json=payload, stream=True)

#                     if response.status_code == 200:
#                         # Initialize an empty string to store the response
#                         response_text = ""

#                         # Streaming the response
#                         for chunk in response.iter_lines():
#                             if chunk:
#                                 chunk_data = chunk.decode('utf-8')
#                                 # Assuming the chunk contains the response content
#                                 response_text += chunk_data
#                                 # Update the Streamlit placeholder
#                                 response_placeholder.markdown(f"**Generated Response:**\n\n{response_text}")
#                     else:
#                         st.error(f"Error {response.status_code}: {response.text}")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")


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

    # Streamlit app setup
    st.title("BigGAN Image Generator")
    st.write("Enter a class name to generate an image using the BigGAN model.")

    st.subheader("Valid Class Names")
    st.write(", ".join(valid_class_names))

    # User input for class name
    class_name = st.text_input("Class Name", value="speedboat", help="Enter a valid class name for the BigGAN model.")

    # Slider for truncation value
    truncation = 0.4

    # Generate button
    if st.button("Generate Image"):
        try:
            # Prepare input
            class_vector = one_hot_from_names([class_name], batch_size=1)
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

            # Convert inputs to tensors
            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector)

            # If you have a GPU, use it
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noise_vector = noise_vector.to(device)
            class_vector = class_vector.to(device)
            model.to(device)

            # Generate image
            with torch.no_grad():
                output = model(noise_vector, class_vector, truncation)

            # Move output to CPU and normalize
            output = output.cpu().numpy()
            img = ((output[0] + 1) / 2 * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC format for PIL
            img = Image.fromarray(img)

            # Display the image
            st.image(img, caption=f"Generated Image for '{class_name}'", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Ensure the class name is valid for BigGAN.")

    # Footer
    st.markdown(
        """
        ---
        Built with [Streamlit](https://streamlit.io) and [PyTorch](https://pytorch.org).
        """
    )




# #LLMS
# def Meeting_Notes_Summarizer_using_LLMS():
#     """
#     This function combines the entire process of summarizing meeting notes into a single function.
#     It integrates loading the model, preprocessing input, generating a summary, and displaying the result via Streamlit.
#     """

#     # Step 2: Load Pre-trained Model and Tokenizer
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")

#     # Step 3: Preprocess Input Text
#     def preprocess_input(text):
#         """
#         Prepares the input text for the model by encoding it.
#         The input text is prepended with "summarize: " to specify the task.
#         """
#         inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
#         return inputs

#     # Step 4: Generate Summary using T5
#     def generate_summary(text):
#         """
#         Generates a summary for the input text using the T5 model.
#         The summary is decoded and split into bullet points.
#         """
#         # Preprocess input text
#         inputs = preprocess_input(text)

#         # Generate summary using the model
#         summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

#         # Decode the generated summary
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#         # Split the summary into bullet points
#         summary_points = summary.split(". ")
#         return summary_points

#     # Step 5: Initialize Streamlit UI for Summarizing Meeting Notes
#     st.title("Meeting Notes Summarizer")
#     st.write("This web app summarizes your meeting notes into concise bullet points.")

#     # Step 6: Input Area for Meeting Notes
#     user_input = st.text_area("Enter Meeting Notes:")

#     # Button to trigger summarization
#     if st.button("Summarize Meeting Notes"):
#         if user_input:
#             # Display the input text first
#             st.write("### Original Meeting Notes:")
#             st.write(user_input)

#             # Generate the summary and display it
#             summary_points = generate_summary(user_input)
#             st.write("### Summarized Meeting Notes:")

#             # Display the bullet points
#             for i, point in enumerate(summary_points, 1):
#                 st.write(f"• {point}")
#         else:
#             st.write("Please enter meeting notes to summarize.")

# Step 7: Run the Streamlit Application
# After saving this file, run it with the following command:
# streamlit run meeting_notes_summarizer.py




# ###Diffusion Models
# def Image_Generation_using_Diffusion_Models():
#     """
#     Streamlit app to generate images using Stable Diffusion and Hugging Face's Diffusers library.
#     """
#     # Load the Stable Diffusion pipeline with caching to improve load times
#     @st.cache_resource
#     def load_pipeline():
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#         pipe = pipe.to(device)
#         return pipe

#     # Initialize the pipeline
#     pipe = load_pipeline()

#     # Streamlit UI
#     st.title("Stable Diffusion Image Generator")
#     st.write("Enter a prompt to generate images using Stable Diffusion.")

#     # User input for prompt
#     prompt = st.text_input(
#         "Prompt",
#         value="a flying dog",
#         help="Enter a text prompt to generate images."
#     )

#     # Settings for generation
#     num_images = st.slider(
#         "Number of Images",
#         min_value=1,
#         max_value=4,
#         value=2
#     )
#     guidance_scale = st.slider(
#         "Guidance Scale",
#         min_value=1.0,
#         max_value=20.0,
#         value=7.5,
#         step=0.5,
#         help="Controls how closely the image matches the prompt."
#     )
#     steps = st.slider(
#         "Inference Steps",
#         min_value=10,
#         max_value=100,
#         value=50,
#         step=1,
#         help="Controls the quality of the generated images."
#     )

#     # Generate button
#     if st.button("Generate Images"):
#         with st.spinner("Generating images..."):
#             try:
#                 # Generate images using the pipeline
#                 results = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images)
#                 images = results.images

#                 # Display the images
#                 st.write(f"Generated Images for Prompt: **{prompt}**")
#                 for idx, img in enumerate(images):
#                     st.image(img, caption=f"Image {idx+1}", use_column_width=True)

#             except Exception as e:
#                 st.error(f"An error occurred: {e}")

#     # Footer
#     st.markdown(
#         """
#         ---
#         Built with [Streamlit](https://streamlit.io) and [Hugging Face Diffusers](https://huggingface.co/docs/diffusers).
#         """
#     )



# Main Streamlit app
def main():
    # Sidebar configuration
    st.sidebar.title("Choose a Functionality")
    st.sidebar.write("Select an option below to explore different features of the app:")
    
    # Sidebar options
    options = [
        "Language Translation using Transformers",
        # "Chat with Document using RAG",
        # "Chat with Image using NVLM",
        # "Essay Generation using LLAMA",
        "Image Generation using GAN",
        # "Meeting Notes Summarizer",
        # "Image Generation using Diffusion Models"
    ]
    choice = st.sidebar.radio("Select an option:", options)

    # Call the relevant function based on user choice
    if choice == "Language Translation using Transformers":
        Language_Translation_using_Transformers()
    # elif choice == "Chat with Document using RAG":
    #     Chat_with_document_using_RAG()
    # elif choice == "Chat with Image using NVLM":
    #     Chat_with_Image_using_NVLM()
    # elif choice == "Essay Generation using LLAMA":
    #     Essay_Generation_using_LLAMA()
    elif choice == "Image Generation using GAN":
        Image_Generation_using_GAN()
    # elif choice == "Meeting Notes Summarizer":
        # Meeting_Notes_Summarizer_using_LLMS()
    # elif choice == "Image Generation using Diffusion Models":
    #     Image_Generation_using_Diffusion_Models()
    

# Run the app
if __name__ == "__main__":
    main()
