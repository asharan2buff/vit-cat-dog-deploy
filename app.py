import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
import os
import gdown

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 Cat vs 🐶 Dog Classifier (ViT)")

# -------------------------------
# Load model with caching
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "a2_bonus_vit_apurvara_asharan2.pth"

    # ✅ Your new Google Drive file link
    file_id = "1sGwomfIM9PgTYLeXINf5Yh9fr81rVBss"
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

model = load_model()
class_names = ['cat', 'dog']  # Adjust if needed based on your training

# -------------------------------
# File uploader
# -------------------------------
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
