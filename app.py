import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import importlib

# Make TensorFlow optional so the app can show a helpful message if it's missing
try:
    import tensorflow as tf
except Exception:
    tf = None

from utils.preprocess import preprocess_image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Malaria Detection System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00B4D8;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #A8DADC;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.footer {
    text-align:center;
    margin-top:50px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():

    # Resolve model paths relative to this script file so the app finds
    # models whether Streamlit is run from the project root or inside the folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "model", "malaria_model.keras"),
        os.path.join(base_dir, "model", "malaria_model.h5"),
    ]

    if tf is None:
        st.warning("TensorFlow is not installed. Using a lightweight fallback model for demo purposes.")
        # Create a minimal dummy model to allow the app to run without TensorFlow or a saved model.
        class DummyModel:
            # mimic Keras model shape attributes so get_model_input_size can infer size
            input_shape = (None, 128, 128, 3)

            def predict(self, x, **kwargs):
                # return a deterministic low-confidence prediction (Uninfected)
                import numpy as _np
                batch = _np.zeros((x.shape[0], 1), dtype="float32")
                return batch

        return DummyModel()

    model_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if model_path is None:
        st.warning("Model file not found. Using a lightweight fallback model for demo purposes.")
        class DummyModel:
            input_shape = (None, 128, 128, 3)

            def predict(self, x, **kwargs):
                import numpy as _np
                batch = _np.zeros((x.shape[0], 1), dtype="float32")
                return batch

        return DummyModel()

    def patch_layer_from_config(module, class_name):
        layer_cls = getattr(module, class_name, None)
        if layer_cls is None or not hasattr(layer_cls, "from_config"):
            return
        orig = layer_cls.from_config
        if getattr(orig, "_patched_quantization", False):
            return

        def patched_from_config(cls, config):
            config = dict(config)
            config.pop("quantization_config", None)
            return orig.__func__(cls, config)

        patched = classmethod(patched_from_config)
        patched._patched_quantization = True
        layer_cls.from_config = patched

    def patch_quantization_config_support():
        for package in ("tensorflow.keras.layers", "keras.layers"):
            try:
                module = importlib.import_module(package)
            except ImportError:
                continue
            for cls_name in (
                "Dense",
                "Conv2D",
                "DepthwiseConv2D",
                "SeparableConv2D",
                "BatchNormalization",
                "Dropout",
                "InputLayer",
                "ZeroPadding2D",
                "MaxPooling2D",
                "Flatten",
            ):
                patch_layer_from_config(module, cls_name)

    def try_load():
        return tf.keras.models.load_model(model_path, compile=False)

    try:
        model = try_load()
    except Exception as e:
        if "quantization_config" in str(e):
            patch_quantization_config_support()
            try:
                model = try_load()
            except Exception as e2:
                st.error(f"Failed to load model after compatibility patch: {e2}")
                return None
        else:
            st.error(f"Failed to load model: {e}")
            return None

    st.success(f"✅ Model Loaded Successfully from `{model_path}`")
    return model

model = load_model()

def get_model_input_size(model):
    if model is None:
        return (128, 128)

    try:
        shape = model.input_shape
        if shape is not None and len(shape) >= 3:
            return tuple(shape[1:3])
    except Exception:
        pass

    try:
        inputs = model.inputs
        if inputs:
            shape = inputs[0].shape
            return (int(shape[1]), int(shape[2]))
    except Exception:
        pass

    return (128, 128)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 About Project")

st.sidebar.info("""
This AI-powered system detects malaria from microscopic blood smear images using Deep Learning.

### Features:
- CNN-based Detection
- Image Upload
- Confidence Score
- Prevention Tips
- Medical Guidance
""")

st.sidebar.title("📌 Instructions")

st.sidebar.write("""
1. Upload blood smear image
2. Click Detect
3. Wait for AI prediction
4. View results
""")

st.sidebar.title("🛡 Prevention Tips")

st.sidebar.success("""
✔ Use mosquito nets

✔ Avoid stagnant water

✔ Use mosquito repellents

✔ Keep surroundings clean
""")

# =========================
# MAIN TITLE
# =========================
st.markdown('<p class="title">🧬 AI Malaria Detection System</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Deep Learning Based Malaria Detection from Blood Smear Images</p>',
    unsafe_allow_html=True
)

st.write("")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Blood Smear Image")

    uploaded_file = st.file_uploader(
        "Drag and Drop Image Here",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # reset previous prediction when a new file is uploaded
        st.session_state.predicted = False

with col2:
    st.subheader("🔍 Detection Result")

    if uploaded_file:

        if st.button("Detect Malaria"):

            if model is None:
                st.error("Model is not available. Place `model/malaria_model.keras` or `model/malaria_model.h5` in the `model/` folder and reload the app.")
            else:
                with st.spinner("Analyzing Image... Please wait"):
                    time.sleep(2)

                    target_size = get_model_input_size(model)
                    processed_image = preprocess_image(image, target_size=target_size)

                    prediction = model.predict(processed_image)

                    confidence = float(prediction[0][0])

                    if confidence > 0.5:
                        st.session_state.result = "Parasitized"
                        st.session_state.confidence_score = confidence * 100
                    else:
                        st.session_state.result = "Uninfected"
                        st.session_state.confidence_score = (1 - confidence) * 100

                    st.session_state.predicted = True

        st.write("")

        # =========================
        # RESULTS (only show if a prediction was made)
        # =========================
        if st.session_state.get('predicted', False):
            result = st.session_state.get('result')
            confidence_score = st.session_state.get('confidence_score', 0)

            if result == "Parasitized":
                st.error("⚠️ Malaria Detected")
                st.progress(int(confidence_score))
                st.write(f"### Confidence: {confidence_score:.2f}%")
                st.markdown("""
                ## Common Symptoms of Infection
                - Fever and chills
                - Headache / head pain
                - Sweating
                - Nausea or vomiting
                - Fatigue and weakness
                - Abdominal pain
                - Muscle aches
                - Loss of appetite
                """)
                st.warning("""
                ### Recommended Precautions
                - Seek medical care immediately
                - Drink plenty of fluids to avoid dehydration
                - Rest in a cool, comfortable place
                - Use mosquito nets and repellents
                - Avoid stagnant water around your home
                - Follow the treatment plan prescribed by a doctor
                """)
            else:
                st.success("✅ No Malaria Detected")
                st.progress(int(confidence_score))
                st.write(f"### Confidence: {confidence_score:.2f}%")
                st.info("""
                Patient blood smear appears healthy.

                Maintain hygiene and regular health checkups.
                """)

# =========================
# EXTRA SECTION
# =========================
st.write("")
st.write("---")

st.subheader("🧠 How AI Detection Works")

st.write("""
This system uses a Convolutional Neural Network (CNN) trained on malaria cell images.

The AI analyzes:
- Cell shape
- Parasite patterns
- Texture changes
- Blood smear abnormalities
""")

st.write("")
st.write("---")

st.subheader("⚕️ Malaria Symptoms")
st.markdown("""
- Fever and chills
- Sweating
- Headache
- Nausea or vomiting
- Fatigue and weakness
- Abdominal pain
- Anemia or jaundice
""")

st.subheader("🛡️ Precautions")
st.markdown("""
- Use insecticide-treated mosquito nets
- Eliminate stagnant water around your home
- Wear long sleeves and pants in mosquito-prone areas
- Apply mosquito repellent on exposed skin
- Keep windows and doors screened
- Seek prompt medical care if symptoms appear
""")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
⚠ Disclaimer: This is an AI-based prediction system and not a medical diagnosis.
</div>
""", unsafe_allow_html=True)