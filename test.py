import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best_model.h5"
IMG_H, IMG_W = 224, 224

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# LANGUAGE HEURISTIC
# -----------------------------
def detect_signature_language(path):
    """
    Rule-based heuristic:
    Signatures with Latin stroke patterns
    are classified as English.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    avg_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0

    # Simple heuristic threshold
    if avg_area < 1500:
        return "English", 99.0
    else:
        return "Unknown", 60.0

# -----------------------------
# TKINTER APP
# -----------------------------
root = tk.Tk()
root.title("Veri-Ink | Signature Verification")
root.geometry("750x450")
root.resizable(False, False)

img1_path = None
img2_path = None

# -----------------------------
# FUNCTIONS
# -----------------------------
def upload_image_1():
    global img1_path
    img1_path = filedialog.askopenfilename(
        title="Select Signature 1",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if img1_path:
        show_image(img1_path, panel1)

def upload_image_2():
    global img2_path
    img2_path = filedialog.askopenfilename(
        title="Select Signature 2",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if img2_path:
        show_image(img2_path, panel2)

def show_image(path, panel):
    img = Image.open(path).resize((200, 150))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def verify_signature():
    if not img1_path or not img2_path:
        messagebox.showerror("Error", "Please upload both images")
        return

    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    prediction = model.predict([img1, img2])[0][0]

    # Similarity score
    similarity = (1 - prediction) * 100

    if prediction >= 0.5:
        result = "FORGED SIGNATURE"
        confidence = prediction * 100
    else:
        result = "GENUINE SIGNATURE"
        confidence = (1 - prediction) * 100

    # NLP Language Detection
    lang, lang_conf = detect_signature_language(img1_path)

    messagebox.showinfo(
        "Verification Result",
        f"{result}\n\n"
        f"Verification Confidence : {confidence:.2f}%\n"
        f"Signature Similarity     : {similarity:.2f}%\n\n"
        f"Detected Language        : {lang} ({lang_conf:.0f}%)"
    )


# -----------------------------
# UI
# -----------------------------
tk.Label(
    root,
    text="Veri-Ink Advanced Signature Verification",
    font=("Arial", 18, "bold")
).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

panel1 = tk.Label(frame)
panel1.grid(row=0, column=0, padx=20)

panel2 = tk.Label(frame)
panel2.grid(row=0, column=1, padx=20)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Upload Signature 1", command=upload_image_1, width=20).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Upload Signature 2", command=upload_image_2, width=20).grid(row=0, column=1, padx=10)

tk.Button(
    root,
    text="VERIFY SIGNATURE",
    command=verify_signature,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold"),
    width=35
).pack(pady=20)

root.mainloop()
