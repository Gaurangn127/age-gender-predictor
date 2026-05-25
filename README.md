# 🧑 Age & Gender Predictor

A deep learning model that predicts a person's **age** and **gender** from a face image — built with PyTorch Lightning and served via a Gradio web app on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Gaurangn127/age-gender-predictor)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)

---

## 🚀 Live Demo

Try the app directly on Hugging Face Spaces:  
👉 **[https://huggingface.co/spaces/Gaurangn127/age-gender-predictor](https://huggingface.co/spaces/Gaurangn127/age-gender-predictor)**

Upload any face image and get back:
- 🔢 Predicted **age** (continuous value)
- 🚻 Predicted **gender** (Male / Female)

---

## 📌 Project Overview

This project was built for the **Kaggle Competition: Age & Gender Prediction from Face Images** (`sep-25-dl-gen-ai-nppe-1`).

The task is a multi-output prediction problem:
- **Age** → Regression (evaluated via normalized RMSE)
- **Gender** → Binary Classification (evaluated via Macro F1)

Final score is the **Harmonic Mean** of both metrics.

---

## 🏗️ Model Architecture

The model uses a **fine-tuned EfficientNet-V2-S** backbone with two separate prediction heads:

```
EfficientNet-V2-S (backbone)
        │
        ▼
  Shared Features
     ┌──┴──┐
     │     │
Age Head   Gender Head
Linear(128) → ReLU → Dropout(0.4) → Linear(1)
```

- **Backbone:** `EfficientNet-V2-S` (pretrained on ImageNet, fine-tuned)
- **Age Head:** Regression → raw scalar output
- **Gender Head:** Classification → sigmoid applied at inference for binary output
- **Framework:** PyTorch Lightning (`pl.LightningModule`)

---

## 🔄 Inference Pipeline

```
Input Image
    │
    ▼
Resize to 300×300
    │
    ▼
Normalize (ImageNet stats)
    │
    ▼
EfficientNet-V2-S Backbone
    │
    ▼
┌───────────────┐
│  Age Head     │ → Predicted Age (clipped to ≥ 0)
│  Gender Head  │ → Sigmoid → 0/1 → "Female"/"Male"
└───────────────┘
```

---

## 📁 Repository Structure

```
age-gender-predictor/
├── app.py                  # Gradio app — model loading & inference
├── model.ckpt              # Trained model checkpoint (PyTorch Lightning)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Local Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/age-gender-predictor.git
cd age-gender-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`.

> **Note:** `model.ckpt` (330 MB) is stored via Git LFS. Make sure you have [Git LFS](https://git-lfs.com/) installed before cloning.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Core deep learning framework |
| `torchvision` | EfficientNet backbone & image transforms |
| `pytorch-lightning` | Model training & checkpoint management |
| `gradio` | Interactive web UI |
| `Pillow` | Image loading and processing |

---

## 📊 Evaluation Metric

The competition uses the **Harmonic Mean of F1 and nRMSE**:

| Target | Metric |
|--------|--------|
| Age | `age_score = 1 - clamp(RMSE, 30) / 30` |
| Gender | `gender_score = Macro F1` |
| **Final** | `2 × (gender_score × age_score) / (gender_score + age_score)` |

---

## 🤗 Hugging Face Space

The model is hosted as a Gradio app on Hugging Face Spaces.  
Space: [`Gaurangn127/age-gender-predictor`](https://huggingface.co/spaces/Gaurangn127/age-gender-predictor)

---

## 📝 License

This project is for educational purposes as part of a Deep Learning / Gen AI course assignment.