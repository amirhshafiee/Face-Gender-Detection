# Face Detection & Gender Classification

A simple project for **face detection** and **gender classification** using **OpenCV**, **MTCNN**, and **Scikit-learn**.  
It allows you to run real-time detection using your webcam.

---

## 📌 Features
- Detects faces using **MTCNN**
- Classifies gender as **male** or **female**
- Real-time detection with webcam
- Lightweight model using **SGDClassifier**

---

## 📂 Project Structure
```
.
├── learning.py        # Train the gender classification model (optional)
├── Detection.py       # Run real-time face & gender detection
└── gender_classifier.z # Saved trained model
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python numpy scikit-learn mtcnn joblib
   ```

---

## 📊 Usage

### 1. (Optional) Train the model with your own dataset
Organize your images like this:
```
gender/
├── male/
│   ├── image1.jpg
│   ├── image2.jpg
│   ...
└── female/
    ├── image1.jpg
    ├── image2.jpg
    ...
```

Then run:
```bash
python learning.py
```
This will train a gender classifier and save it as `gender_classifier.z`.

### 2. Run real-time detection
Once `gender_classifier.z` is generated, run:
```bash
python Detection.py
```
Press **q** to quit the webcam window.

---

## 🖼️ Example Output

When running the detection script, faces will be highlighted with a bounding box and labeled as **male** (green) or **female** (red).

---

## 📧 Author
**Amirhosien Shafiee**  
- Telegram: [@amirhshafiee](https://t.me/amirhshafiee)  
- Email: amirshafiee266@yahoo.com  
