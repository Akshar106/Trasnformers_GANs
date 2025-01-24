# **AI Streamlit Application**

This application provides various AI-powered functionalities using pre-trained models from Hugging Face and other libraries like PyTorch and Streamlit. The following features are included in the app:

1. **Language Translation using Transformers**
2. **Chat with Document using RAG**
3. **Chat with Image using NVLM**
4. **Essay Generation using LLAMA**
5. **Image Generation using GAN**
6. **Meeting Notes Summarizer using LLMS**
7. **Image Generation using Diffusion Models**

## **Technologies Used**

- **Streamlit**: Framework to build interactive web apps.
- **PyTorch**: For deep learning models.
- **Hugging Face Transformers**: Pre-trained NLP models.
- **BigGAN**: For image generation based on class names.
- **Stable Diffusion**: For text-to-image generation.
- **T5**: For text summarization.

## **Features**

### 1. **Language Translation using Transformers**
   - Translate text from one language to another using Hugging Face's pre-trained models.
   - Supports translation between various languages.

### 2. **Chat with Document using RAG**
   - Allows users to upload a document and ask questions about it using Retrieval-Augmented Generation (RAG).
   - Answers are generated based on the context of the document.

### 3. **Chat with Image using NVLM**
   - Generate responses related to images using the NVLM (Neural Visual Language Model) model.
   - Users can upload an image and ask questions about it.

### 4. **Essay Generation using LLAMA**
   - Generate an essay on a given topic using LLAMA model.
   - Users input a topic, and the model generates an essay with structured content.

### 5. **Image Generation using GAN**
   - Generate images based on user-input class names using the BigGAN model.
   - Class names should correspond to valid categories in the model (e.g., "goldfish", "shark", etc.).

### 6. **Meeting Notes Summarizer using LLMS**
   - Summarize meeting notes into bullet points using the T5 model.
   - Provides a concise summary of meeting discussions in a readable format.

### 7. **Image Generation using Diffusion Models**
   - Generate images based on textual descriptions using Stable Diffusion.
   - Users can adjust parameters such as the number of images, guidance scale, and inference steps to fine-tune the output.

## **Installation**

To run this application locally, follow these steps:

### Step 1: Install Dependencies

1. Clone the repository:
   git clone https://github.com/BusinessOptimaCloud/Gen-AI-demos.git
   cd GenAI App

2. Install required libraries:
   pip install -r requirements.txt


### Step 2: Run the Application

To start the app, run the following command:

   streamlit run main.py

