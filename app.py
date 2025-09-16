import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load trained model
MODEL = load_model("mnist_model.keras")

# Labels
LABEL = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

st.title("🖊️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) in the canvas below and click **Predict**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",   # Background is black
    stroke_width=10,      # Thickness of pen
    stroke_color="white", # White digit
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess(img):
    # Convert RGBA → grayscale
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Threshold (remove noise, make sure it's binary)
    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = gray[y:y+h, x:x+w]
    else:
        digit = gray

    # Resize to fit in 20x20 box
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Create 28x28 black canvas and put digit in the center
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    # Normalize
    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)
    return canvas

if st.button("Predict"):
    if canvas_result.image_data is not None:
        processed = preprocess(canvas_result.image_data)

        # Predict
        pred = MODEL.predict(processed)
        label = np.argmax(pred)

        st.write(f"### Prediction: {LABEL[label]}")
        st.bar_chart(pred[0])
    else:
        st.warning("Please draw a digit before predicting!")
