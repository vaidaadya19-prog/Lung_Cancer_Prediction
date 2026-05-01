import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Malignancy Predictor",
    page_icon="🫁",
    layout="wide"
)

# ── Download ResNet18 model from Google Drive if not present ──────────────────
@st.cache_resource
def load_models():
    # Download resnet18 from Google Drive
    model_path = 'resnet18_best.pth'
    if not os.path.exists(model_path):
        with st.spinner('Downloading CNN model... (first time only, ~43MB)'):
            gdown.download(
                'https://drive.google.com/uc?id=1vhwb31Bz5QGNHR-X2kDBtLlgetOb0WAb',
                model_path, quiet=False)

    # Load Random Forest
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    # Load Scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load ResNet18
    device    = torch.device('cpu')
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(
        torch.load(model_path, map_location=device))
    cnn_model.eval()

    return rf_model, scaler, cnn_model

rf_model, scaler, cnn_model = load_models()

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, inp, class_idx):
        self.model.eval()
        out = self.model(inp)
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).detach().numpy()
        cam     = cv2.resize(cam, (224, 224))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Lung Cancer Malignancy Predictor")
st.markdown("**AI-assisted tool for lung nodule malignancy classification "
            "using the LIDC-IDRI dataset**")
st.markdown("---")

tab1, tab2 = st.tabs([
    "📊 Tabular Input — Feature Scores",
    "🖼️  Image Input — Nodule Image"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — TABULAR
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Radiologist Feature Scores")
    st.info(
        "Rate each feature on a scale based on CT scan assessment. "
        "These are the same 8 features rated by radiologists in the "
        "LIDC-IDRI dataset.")

    col1, col2 = st.columns(2)

    with col1:
        subtlety = st.slider(
            "Subtlety",
            1, 5, 3,
            help="1 = Extremely subtle, 5 = Obvious")
        internalStructure = st.slider(
            "Internal Structure",
            1, 4, 1,
            help="1 = Soft tissue, 2 = Fluid, 3 = Fat, 4 = Air")
        calcification = st.slider(
            "Calcification",
            1, 6, 1,
            help="1 = Absent, 2 = Laminated, 3 = Popcorn, "
                 "4 = Dystrophic, 5 = Central, 6 = Eccentric")
        sphericity = st.slider(
            "Sphericity",
            1, 5, 3,
            help="1 = Linear, 3 = Ovoid, 5 = Round")

    with col2:
        margin = st.slider(
            "Margin",
            1, 5, 3,
            help="1 = Poorly defined, 5 = Sharp")
        lobulation = st.slider(
            "Lobulation",
            1, 5, 1,
            help="1 = No lobulation, 5 = Marked lobulation")
        spiculation = st.slider(
            "Spiculation",
            1, 5, 1,
            help="1 = No spiculation, 5 = Marked spiculation")
        texture = st.slider(
            "Texture",
            1, 5, 4,
            help="1 = Non-solid/GGO, 3 = Part solid, 5 = Solid")

    st.markdown("---")
    predict_tab1 = st.button(
        "🔍 Predict Malignancy", key="tab1_btn",
        use_container_width=True, type="primary")

    if predict_tab1:
        features = np.array([[subtlety, internalStructure, calcification,
                               sphericity, margin, lobulation,
                               spiculation, texture]])

        features_scaled = scaler.transform(features)
        prediction      = rf_model.predict(features_scaled)[0]
        probability     = rf_model.predict_proba(features_scaled)[0]
        confidence      = probability[prediction] * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            if prediction == 1:
                st.error("### 🔴 MALIGNANT")
                st.markdown("The nodule shows characteristics "
                            "**suspicious for malignancy**.")
            else:
                st.success("### 🟢 BENIGN")
                st.markdown("The nodule shows characteristics "
                            "**consistent with a benign lesion**.")

        with res_col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        with res_col3:
            st.metric("Malignancy Probability",
                      f"{probability[1]*100:.1f}%")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2))
        bars = ax.barh(
            ['Benign', 'Malignant'],
            [probability[0]*100, probability[1]*100],
            color=['#2ecc71', '#e74c3c'], height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Class Probabilities')
        for bar, val in zip(bars,
                            [probability[0]*100, probability[1]*100]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature importance
        st.markdown("---")
        st.subheader("Feature Importance")
        feature_names = ['subtlety','internalStructure','calcification',
                         'sphericity','margin','lobulation',
                         'spiculation','texture']
        importances = rf_model.feature_importances_
        sorted_idx  = np.argsort(importances)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        colors = ['#e74c3c' if importances[i] == max(importances)
                  else 'steelblue' for i in sorted_idx]
        ax2.barh([feature_names[i] for i in sorted_idx],
                 importances[sorted_idx], color=colors)
        ax2.set_title('Feature Importance (Random Forest)')
        ax2.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.caption("⚠️ This prediction is for research purposes only "
                   "and should not replace professional medical diagnosis.")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload Nodule Image")
    st.info(
        "Upload a cropped nodule image (JPG or PNG). "
        "For best results, use a grayscale CT nodule patch "
        "with the nodule centered in the image.")

    uploaded_file = st.file_uploader(
        "Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image     = Image.open(uploaded_file).convert('L')
        img_array = np.array(image, dtype=np.float32)

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(image, caption='Uploaded Nodule',
                     use_column_width=True)
        with col_info:
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} px")
            st.write(f"**Min pixel:** {img_array.min():.0f}")
            st.write(f"**Max pixel:** {img_array.max():.0f}")
            st.write(f"**Mean pixel:** {img_array.mean():.1f}")

        st.markdown("---")
        predict_tab2 = st.button(
            "🔍 Predict Malignancy", key="tab2_btn",
            use_container_width=True, type="primary")

        if predict_tab2:
            # Preprocess image
            img_resized = cv2.resize(img_array, (224, 224))
            mn, mx = img_resized.min(), img_resized.max()
            if mx > mn:
                img_norm = (img_resized - mn) / (mx - mn)
            else:
                img_norm = img_resized

            inp = torch.tensor(
                np.stack([img_norm]*3)[None],
                dtype=torch.float32)

            # Predict
            with torch.no_grad():
                out   = cnn_model(inp)
                probs = torch.softmax(out, dim=1)[0].numpy()

            prediction = int(np.argmax(probs))
            confidence = probs[prediction] * 100

            st.markdown("---")
            st.subheader("Prediction Result")

            res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

            with res_col1:
                if prediction == 1:
                    st.error("### 🔴 MALIGNANT")
                    st.markdown("The nodule image shows features "
                                "**suspicious for malignancy**.")
                else:
                    st.success("### 🟢 BENIGN")
                    st.markdown("The nodule image shows features "
                                "**consistent with a benign lesion**.")

            with res_col2:
                st.metric("Confidence", f"{confidence:.1f}%")

            with res_col3:
                st.metric("Malignancy Probability",
                          f"{probs[1]*100:.1f}%")

            # Probability bar
            fig3, ax3 = plt.subplots(figsize=(6, 2))
            bars3 = ax3.barh(
                ['Benign', 'Malignant'],
                [probs[0]*100, probs[1]*100],
                color=['#2ecc71', '#e74c3c'], height=0.5)
            ax3.set_xlim(0, 100)
            ax3.set_xlabel('Probability (%)')
            ax3.set_title('Class Probabilities')
            for bar, val in zip(bars3,
                                [probs[0]*100, probs[1]*100]):
                ax3.text(bar.get_width() + 1,
                         bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}%', va='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            # Grad-CAM
            st.markdown("---")
            st.subheader("Grad-CAM Visualization")
            st.caption(
                "Warm colors (red/yellow) show regions the model "
                "focused on when making its prediction.")

            gradcam  = GradCAM(cnn_model, cnn_model.layer4[-1])
            inp_grad = torch.tensor(
                np.stack([img_norm]*3)[None],
                dtype=torch.float32)
            cam = gradcam.generate(inp_grad, class_idx=prediction)

            fig4, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')

            axes[2].imshow(img_norm, cmap='gray')
            axes[2].imshow(cam, cmap='jet', alpha=0.45)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            st.caption(
                "⚠️ This prediction is for research purposes only "
                "and should not replace professional medical diagnosis.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px'>"
    "Lung Cancer Malignancy Prediction System | "
    "Built using LIDC-IDRI Dataset | "
    "Random Forest + ResNet18"
    "</div>",
    unsafe_allow_html=True)
