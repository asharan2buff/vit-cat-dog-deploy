import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
import os
import requests
import gdown

import gdown

@st.cache_resource
def load_model():
    model_path = "vit_cats_dogs_model.pth"

    # Google Drive File ID
    file_id = "1K6QfhlEijduFu2EvlSZPcD7ma7eNZTX_"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded!")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model



# Page setup
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 Cat vs 🐶 Dog Classifier (ViT)")


@st.cache_resource
def load_model():
    model_path = "a2_bonus_vit_apurvara_asharan2.pth"

    # ✅ Replace this with your actual direct download link:
    url = "https://drive.google.com/uc?export=download&id=1K6QfhlEijduFu2EvlSZPcD7ma7eNZTX_"

    # Download model if not already present
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(url)
            with open(model_path, "wb") as f:
                f.write(r.content)
            st.success("✅ Model downloaded")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()
class_names = ['cat', 'dog']  # Ensure this matches your dataset order

# File uploader
uploaded_file = st.file_uploader("Upload a cat or dog image 🐾", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted = torch.argmax(outputs.logits, dim=1).item()
            label = class_names[predicted]

        st.markdown(f"### 🧠 Prediction: **{label.upper()}**")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
