# 🖊️ Handwritten Digit Recognition (MNIST)

A deep learning project that classifies handwritten digits (0–9) from the famous **MNIST dataset** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.
The project also includes a **Streamlit-based frontend** where users can draw digits on a canvas and get real-time predictions from the trained model.

---

## 🚀 Project Demo

👉 [Live Deployment Link](https://minist-digits-classification-using-cnn-dl-project-5bkpynydsavd.streamlit.app/)

---

## 📊 Model Performance

* **Loss:** `0.0488`
* **Accuracy:** `99.14%` on test data

---

## 🧠 Model Architecture

The CNN is designed as follows:

```python
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
```

### Preprocessing

* Input images reshaped to `(28,28,1)`
* Normalized pixel values by dividing by `255.0`

---

## 🛠️ Tech Stack

* **Python 3.12+**
* **TensorFlow / Keras** – Model training
* **NumPy & Pandas** – Data preprocessing
* **Matplotlib** – Visualization
* **Streamlit** – Frontend for drawing & predictions
* **OpenCV (cv2)** – Image preprocessing

---

## ✅ Results

The model achieves **99% accuracy**, making it highly reliable for handwritten digit recognition tasks.

---

## 👨‍💻 Author

Developed by **Abdul Moiz** ✨

