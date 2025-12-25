import os, sqlite3
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = "veri_ink_secret_key"

UPLOAD_FOLDER = "static/uploads"
DB_PATH = "database.db"
MODEL_PATH = "best_model.h5"
IMG_H, IMG_W = 224, 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully")

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- NLP / LANGUAGE HEURISTIC ----------------
def detect_signature_language(path):
    """
    Rule-based NLP heuristic for signatures.
    Latin-style strokes => English
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Unknown", 50

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "Unknown", 50

    avg_area = np.mean([cv2.contourArea(c) for c in contours])

    # Empirical threshold
    if avg_area < 1500:
        return "English", 99
    else:
        return "Unknown", 60

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                (name, email, phone, password)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "Email already exists!"

    return render_template("register.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[4], password):
            session["user"] = user[0]
            session["name"] = user[1]
            session["email"] = user[2]
            print(" User logged in:", email)
            print(f"User logged in: {user[1]} ({user[2]})")

            
            return redirect(url_for("predict"))

        return "Invalid login credentials"

    return render_template("login.html")

# -------- PREDICT --------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    result = similarity = None
    language = lang_conf = None

    if request.method == "POST":
        img1 = request.files["img1"]
        img2 = request.files["img2"]

        path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
        path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))

        img1.save(path1)
        img2.save(path2)

        i1 = preprocess_image(path1)
        i2 = preprocess_image(path2)

        pred = model.predict([i1, i2])[0][0]

        # Similarity score
        similarity = round((1 - pred) * 100, 2)

        result = "GENUINE SIGNATURE" if pred < 0.5 else "FORGED SIGNATURE"

        #  NLP Language Detection
        language, lang_conf = detect_signature_language(path1)

    return render_template(
        "predict.html",
        result=result,
        similarity=similarity,
        language=language,
        lang_conf=lang_conf
    )

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
