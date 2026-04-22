This project is a deepfake detection system using:

- CNN (EfficientNet)
- FFT-based frequency analysis
- Hybrid model (CNN + FFT fusion)
- Grad-CAM for explainability
- Streamlit UI for user interaction

The dataset used includes Celeb-DF and additional GAN/diffusion images.

Goal:
- Detect deepfake images
- Provide explainability
- Generate human-readable explanations using LLM

Tech stack:
- Python
- PyTorch
- OpenCV
- Streamlit
- Ollama (local LLM)

Folder structure:
- src/
- data/
- models/

Focus:
- Improving generalization
- Enhancing explanations