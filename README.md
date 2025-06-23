# AI Multi-Tool: Translation & Image Generation

A comprehensive Streamlit web application that combines language translation and AI-powered image generation in one intuitive interface.

## 🌟 Features

### 🔤 Language Translation
- **Multi-language Support**: Translate English text to French, German, Spanish, and Italian
- **Transformer-based**: Uses state-of-the-art MarianMT models from Hugging Face
- **Real-time Translation**: Fast and accurate neural machine translation
- **User-friendly Interface**: Clean text input with instant results

### 🎨 Image Generation
- **BigGAN Integration**: Generate high-quality 512x512 images using BigGAN-deep model
- **Class-based Generation**: Create images from predefined class names
- **GPU Acceleration**: Automatic GPU detection for faster generation
- **Visual Results**: Display generated images directly in the app

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for faster image generation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-multi-tool.git
   cd ai-multi-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   API_BASE_URL=your_api_base_url_here
   visionchat_api=your_visionchat_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit
transformers
torch
torchvision
python-dotenv
openai
nltk
pytorch-pretrained-biggan
numpy
Pillow
```

## 🎯 Usage

### Language Translation
1. Select "Language Translation using Transformers" from the sidebar
2. Choose your target language (French, German, Spanish, Italian)
3. Enter English text in the text area
4. Click "Translate" to get instant results

### Image Generation
1. Select "Image Generation using GAN" from the sidebar
2. Choose from available class names or enter a custom one
3. Click "Generate Image" to create a new image
4. View the generated 512x512 image

### Available Image Classes
The BigGAN model supports various classes including:
- Animals: goldfish, great white shark, German shepherd, pug, dalmatian
- Birds: cock, hen, ostrich, peacock, quail, African grey
- And many more...

## 🏗️ Architecture

```
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Key Components

- **Model Caching**: Uses `@st.cache_resource` for efficient model loading
- **Session State**: Maintains user input across interactions
- **Error Handling**: Comprehensive error handling for model operations
- **Responsive Design**: Mobile-friendly interface with Streamlit

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI features)
- `API_BASE_URL`: Base URL for API calls
- `visionchat_api`: VisionChat API key (if applicable)

### Model Settings
- **Translation Models**: Helsinki-NLP MarianMT models
- **Image Generation**: BigGAN-deep-512 pretrained model
- **Truncation**: Set to 0.4 for balanced quality/diversity

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU mode
   - Close other GPU-intensive applications

2. **Model Download Errors**
   - Ensure stable internet connection
   - Check Hugging Face model availability

3. **Translation Errors**
   - Verify input text is in English
   - Check target language selection

4. **Image Generation Fails**
   - Ensure class name is valid for BigGAN
   - Check available GPU memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) for image generation
- [Streamlit](https://streamlit.io/) for the web framework
- [PyTorch](https://pytorch.org/) for deep learning backend

## 📊 Performance

- **Translation Speed**: ~1-2 seconds per sentence
- **Image Generation**: ~5-10 seconds (GPU) / ~30-60 seconds (CPU)
- **Memory Usage**: ~2-4GB RAM, ~4-6GB VRAM (for image generation)

## 🔮 Future Enhancements

- [ ] Additional language pairs
- [ ] Custom image prompts
- [ ] Batch processing
- [ ] API endpoint creation
- [ ] Docker containerization
- [ ] Cloud deployment guides
