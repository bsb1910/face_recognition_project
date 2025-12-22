
# 🧠 Face Recognition System using OpenCV & Deep Learning

This project implements a **face recognition pipeline** using **OpenCV’s DNN module**, **OpenFace embeddings**, and **Machine Learning (SVM)**.
It supports:

* 📸 Face recognition from **images**
* 🎥 Real-time **webcam face recognition**
* 🏷️ Training custom identities from your own dataset

---

## 📂 Project Folder Structure

```
Face-Recognition-OpenCV/
│
├── dataset/
│   ├── person1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │
│   ├── person2/
│       ├── img1.jpg
│       ├── img2.jpg
│
├── face_detection_model/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── openface_nn4.small2.v1.t7
│
├── output/
│   ├── embeddings.pickle
│   ├── recognizer.pickle
│   └── le.pickle
│
├── extract_embeddings.py
├── train_model.py
├── recognize.py
├── recognize_video.py
│
├── requirements.txt
└── README.md
```

---

## 🔄 Project Flow (Step-by-Step)

### 1️⃣ Dataset Preparation

* Each **person has a separate folder**
* Folder name = **Person name / label**
* Images should contain **one clear face**

```
dataset/
 └── Elon/
     ├── 1.jpg
     ├── 2.jpg
```

---

### 2️⃣ Face Detection

* Uses **OpenCV DNN SSD face detector**
* Model files:

  * `deploy.prototxt`
  * `res10_300x300_ssd_iter_140000.caffemodel`

✔ Detects faces
✔ Filters weak detections using confidence threshold

---

### 3️⃣ Face Embedding Extraction

* Uses **OpenFace neural network**
* Converts each detected face into a **128-D numerical vector**
* Stored as a `.pickle` file

Script:

```
extract_embeddings.py
```

---

### 4️⃣ Model Training

* Uses **Support Vector Machine (SVM)**
* Trained on face embeddings
* Label encoding converts names → numbers

Script:

```
train_model.py
```

---

### 5️⃣ Face Recognition

Two modes supported:

| Mode              | Script               |
| ----------------- | -------------------- |
| Image recognition | `recognize.py`       |
| Real-time webcam  | `recognize_video.py` |

✔ Predicts identity
✔ Displays name + confidence

---

## ⚙️ Important Variables & Their Impact

| Variable            | File            | Impact                       |
| ------------------- | --------------- | ---------------------------- |
| `confidence`        | All scripts     | Filters weak face detections |
| `embeddings.pickle` | extract/train   | Stores face vectors          |
| `recognizer.pickle` | train/recognize | Trained ML model             |
| `le.pickle`         | train/recognize | Label encoder                |
| `128-D vector`      | OpenFace        | Face identity representation |

---

## 🛠️ Tech Stack Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Core language         |
| OpenCV (cv2) | Face detection & DNN  |
| OpenFace     | Face embeddings       |
| scikit-learn | SVM classifier        |
| imutils      | Image/video utilities |
| NumPy        | Numerical operations  |
| Pickle       | Model serialization   |

---

## 🚀 How to Use This Project

### Step 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Face-Recognition-OpenCV.git
cd Face-Recognition-OpenCV
```

---

### Step 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3️⃣ Extract Face Embeddings

```bash
python extract_embeddings.py \
--dataset dataset \
--embeddings output/embeddings.pickle \
--detector face_detection_model \
--embedding-model openface_nn4.small2.v1.t7
```

---

### Step 4️⃣ Train the Recognition Model

```bash
python train_model.py \
--embeddings output/embeddings.pickle \
--recognizer output/recognizer.pickle \
--le output/le.pickle
```

---

### Step 5️⃣ Recognize Face from Image

```bash
python recognize.py \
--image images/test.jpg \
--detector face_detection_model \
--embedding-model openface_nn4.small2.v1.t7 \
--recognizer output/recognizer.pickle \
--le output/le.pickle
```

---

### Step 6️⃣ Real-Time Face Recognition (Webcam)

```bash
python recognize_video.py \
--detector face_detection_model \
--embedding-model openface_nn4.small2.v1.t7 \
--recognizer output/recognizer.pickle \
--le output/le.pickle
```

Press **`q`** to quit webcam.

---

## 📦 requirements.txt

Create a file named **`requirements.txt`** and add:

```
numpy
opencv-python
imutils
scikit-learn
pickle-mixin
```

> ⚠️ If you face OpenCV DNN issues, use:

```
opencv-python-headless
```

---

## 🎯 Applications

* Attendance systems
* Identity verification
* Surveillance systems
* College / academic mini-projects
* AI & ML learning projects

---

## ⚠️ Limitations

* Works best with **frontal faces**
* Performance depends on **lighting**
* Accuracy improves with **more images per person**

---

## 📌 Future Improvements

* Add face alignment
* Use deep CNN classifiers
* Add database integration
* Improve low-light performance

---

If you want, I can also:

* ✅ Convert this into **PDF**
* ✅ Add **diagrams / flowcharts**
* ✅ Optimize code for **production**
* ✅ Help you write **project report / viva answers**

Just tell me 👍
