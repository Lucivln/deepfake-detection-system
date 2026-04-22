import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
import ollama

# import FFT
from fft_model import fft_score

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "deepfake_model_augmented.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD MODEL (once)
# ----------------------------
@st.cache_resource
def load_model():
    model = efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# CNN SCORE
# ----------------------------
def cnn_score(image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)

    return probs[0][0].item()  # fake prob


# ----------------------------
# GRAD-CAM
# ----------------------------
def generate_gradcam(image):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    features.clear()
    gradients.clear()

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    fmap = features[0]

    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    cam = cv2.resize(cam, (224, 224))

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + img_np
    return overlay.astype(np.uint8)


# ----------------------------
# OLLAMA EXPLANATION
# ----------------------------
def generate_explanation(label, cnn_score, fft_score):
    prompt = f"Explain why this image is predicted as {label}. CNN score: {cnn_score:.4f}, FFT score: {fft_score:.4f}. Focus on the {label.lower()} regions such as face, eyes, mouth, and blending areas. Keep it short, 3-4 lines."
    try:
        response = ollama.chat(model='qwen2.5-coder:7b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


# ----------------------------
# UI
# ----------------------------
st.title("🔍 Deepfake Detection System")
st.write("CNN + FFT + Explainability (Grad-CAM)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # CNN + FFT
    cnn_s = cnn_score(image)
    fft_raw = fft_score(image)

    fft_s = min(max((fft_raw - 0.25) * 3, 0), 1)
    final_score = 0.8 * cnn_s + 0.2 * fft_s

    label = "FAKE" if final_score > 0.5 else "REAL"

    st.subheader("Prediction")
    st.write(f"**Label:** {label}")
    st.write(f"Confidence: {final_score:.4f}")

    # Grad-CAM
    st.subheader("Grad-CAM Explanation")
    cam_image = generate_gradcam(image)

    st.image(cam_image, caption="Model Attention", use_column_width=True)

    # Ollama Explanation
    if st.button("Explain Prediction"):
        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(label, cnn_s, fft_s)
        st.subheader("AI Explanation")
        st.write(explanation)