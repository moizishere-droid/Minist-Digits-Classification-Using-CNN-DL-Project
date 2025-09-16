import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load trained model
MODEL = load_model("mnist_model.h5", compile=False, safe_mode=False

# Labels
LABEL = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

st.title("🖊️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the canvas below and click **Predict**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",   # Fill color
    stroke_width=10,      # Thickness of pen
    stroke_color="white", # Drawing color
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Get image from canvas
        img = canvas_result.image_data

        # Convert RGBA → grayscale
        gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

        # Resize to 28x28 like MNIST
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert colors: MNIST digits are white on black
        processed = cv2.bitwise_not(resized)

        # Normalize
        processed = processed.astype("float32") / 255.0
        processed = processed.reshape(1, 28, 28, 1)

        # Predict
        pred = MODEL.predict(processed)
        label = np.argmax(pred)

        st.write(f"### Prediction: {LABEL[label]}")
        st.bar_chart(pred[0])
    else:
        st.warning("Please draw a digit before predicting!")

