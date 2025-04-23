# Detecting Helmet‑Rule Violations for Motorcyclists

*(Challenge Track 5 – Computer Vision)*

> **Safer roads begin with smarter vision.**
> Group ID‑4 (**Group 5**)

---

## 1  Problem Statement

Motorcyclists riding without helmets face a sharply higher risk of fatal injury.
Our mission is to **detect in real‑time any rider (or pillion) missing a helmet in traffic‑camera footage**, enabling authorities to enforce safety regulations and ultimately reduce crash‑related deaths.

---

## 2  Team

| Role             | Name                                  | GitHub                                    |
| ---------------- | ------------------------------------- | ----------------------------------------- |
| Developer        | **Berke Özkır**              | [@berkeozkir](https://github.com/berkeozkir) |
| Developer        | **Kaan Özarslan**             | [@kaanoz1](https://github.com/kaanoz1)       |
| Faculty Advisor | **Dr. Mehmet Kılıçarslan** | –                                        |

---

## 3  High‑Level Objectives

1. **Data Pipeline** – curate or annotate a balanced dataset (helmet / no‑helmet) and implement robust data loaders & augmentations.
2. **Modeling** – establish a baseline detector (e.g. **YOLOv8** or **EfficientDet**) and iterate with multi‑task heads or attribute classifiers.
3. **Evaluation** – track *mAP\@0.5*, precision/recall and false‑positive rate per frame, stressing edge cases (night, occlusion, rain).
4. **Deployment Demo** – build a real‑time inference script/notebook and an optional ONNX/TensorRT engine for edge devices.

---

## 4  Repository Layout *(initial)*

```
├── data/               # raw & processed datasets (git‑ignored)
├── notebooks/          # exploratory EDA & experiments
├── src/
│   ├── datamodules/    # dataset + augmentations
│   ├── models/         # detection & classification nets
│   ├── utils/          # helpers (visualisation, metrics, etc.)
│   └── inference.py    # entry‑point for real‑time demo
├── requirements.txt    # to be filled iteratively
└── README.md
```

---

## 5  Getting Started - How to contribute

```bash
# 1 Clone the repo
git clone https://github.com/berkeozkir/estu_project_cv.git
cd <repo>

# 2 Create & activate a virtual env
python -m venv .venv && source .venv/bin/activate

# 3 Install requirements
pip install -r requirements.txt
```

> **Note:** Large datasets and checkpoint files go in `data/` and are tracked with **Git LFS**

---

## 6  License

Unless course policy dictates otherwise, this project will be released under the **MIT License**.

---
