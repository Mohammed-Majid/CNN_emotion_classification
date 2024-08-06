import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2

# Load the trained model
model = load_model("custom_model_v3.h5")

# Define the emotion classes
emotion_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# Function to preprocess the image
def preprocess_image(image):
    size = (48, 48)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image


# Function to predict emotion
def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = emotion_classes[predicted_class]
    return predicted_emotion


class EmotionClassifier(VideoTransformerBase):
    def __init__(self):
        self.prediction = None
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.prediction = predict_emotion(img_pil)
        return frame


st.title("Emotion Recognition using CNN")
st.write(
    "Use your webcam to capture an image or upload an image file to predict the emotion."
)

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Webcam")
    webrtc_ctx = webrtc_streamer(
        key="example", video_transformer_factory=EmotionClassifier
    )
    webcam_image = None
    if webrtc_ctx.video_transformer:
        if webrtc_ctx.video_transformer.prediction:
            st.write(
                f"Predicted Emotion from Webcam: {webrtc_ctx.video_transformer.prediction}"
            )

with col2:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        if st.button("Classify Uploaded Image"):
            predicted_emotion = predict_emotion(uploaded_image)
            st.write(f"Predicted Emotion from Uploaded Image: {predicted_emotion}")

# Button to classify from webcam
if st.button("Classify Webcam Image") and webrtc_ctx.video_transformer:
    if webrtc_ctx.video_transformer.frame is not None:
        st.image(
            webrtc_ctx.video_transformer.frame,
            caption="Captured Image.",
            use_column_width=True,
        )
        predicted_emotion = webrtc_ctx.video_transformer.prediction
        st.write(f"Predicted Emotion from Webcam: {predicted_emotion}")
    else:
        st.write("Please capture an image using the webcam.")
