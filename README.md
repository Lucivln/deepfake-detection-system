🧠 Deepfake Detection System



A Hybrid deepfake detection system that combines Convolutional Neural Networks (CNN), Frequency Domain Analysis (FFT), and Explainable AI (Grad-CAM + LLM) to accurately identify manipulated facial images.



\---



🚀 Overview



This project focuses on detecting deepfake images by leveraging both spatial features (CNN) and frequency-based artifacts (FFT).

It further enhances interpretability using Grad-CAM visualizations and AI-generated explanations via Ollama (LLM).



\---



✨ Features



\- 🧠 CNN-Based Detection (EfficientNet-B0)

\- 📊 FFT Analysis for frequency-domain inconsistencies

\- 🔗 Hybrid Model (CNN + FFT Fusion)

\- 🔍 Grad-CAM Visualization for model explainability

\- 🤖 AI Explanation System (Ollama - qwen2.5-coder)

\- 🌐 Streamlit Web App UI

\- ⚡ GPU acceleration support (CUDA)



\---



🛠️ Tech Stack



\- Python

\- PyTorch

\- OpenCV

\- NumPy

\- Streamlit

\- Ollama (Local LLM)

\- Torchvision



\---



🏗️ Project Architecture



Input Image

&#x20;    │

&#x20;    ▼

&#x20;CNN Model (EfficientNet)

&#x20;    │

&#x20;    ├──► Prediction Score

&#x20;    │

&#x20;    ▼

&#x20;FFT Analysis

&#x20;    │

&#x20;    ├──► Frequency Score

&#x20;    │

&#x20;    ▼

&#x20;Hybrid Fusion (CNN + FFT)

&#x20;    │

&#x20;    ▼

&#x20;Final Prediction (Real / Fake)

&#x20;    │

&#x20;    ├──► Grad-CAM (visual explanation)

&#x20;    └──► Ollama LLM (text explanation)



\---



📂 Project Structure



deepfake-detection-system/

│

├── src/

│   ├── app.py

│   ├── train\_model.py

│   ├── hybrid\_model.py

│   ├── fft\_model.py

│   ├── gradcam.py

│   └── ...

│

├── README.md

├── .gitignore



\---



⚙️ Setup \& Installation



1️⃣ Clone the repository



git clone https://github.com/Lucivln/deepfake-detection-system.git

cd deepfake-detection-system



2️⃣ Create environment



conda create -n deepfake python=3.11

conda activate deepfake



3️⃣ Install dependencies



pip install -r requirements.txt



4️⃣ Install \& run Ollama



Download: https://ollama.com/download



Run model:



ollama run qwen2.5-coder:7b



\---



▶️ Run the Application



streamlit run src/app.py



\---



🧪 Workflow



1\. Upload an image

2\. Model predicts Real / Fake

3\. Grad-CAM highlights important regions

4\. AI generates explanation using LLM



\---



📊 Dataset



\- Celeb-DF Dataset

\- Processed into:

&#x20; - Real face images

&#x20; - Fake face images

\- Balanced dataset (\~70k samples total)



\---



⚠️ Limitations



\- Trained primarily on face-swap deepfakes

\- May not generalize well to:

&#x20; - GAN-generated images

&#x20; - Diffusion-based images (e.g., Midjourney, DALL·E)



\---



🔮 Future Improvements



\- Add GAN/diffusion datasets

\- Improve generalization across domains

\- Deploy as web service

\- Add real-time video detection

\- Enhance explanation quality



\---

