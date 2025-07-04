# Detecting Helmet‑Rule Violations for Motorcyclists

*(Challenge Track 5 – Computer Vision)*

> **Safer roads begin with smarter vision.**
> Group ID‑4 (**Group 5**)

---

## 1  Problem Statement

Motorcyclists riding without helmets face a sharply higher risk of fatal injury.
Our mission is to **detect any rider (or pillion) missing a helmet in traffic‑camera footage**, enabling authorities to enforce safety regulations and ultimately reduce crash‑related deaths.

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
4. **Deployment Demo** – build a inference script/notebook and an optional ONNX/TensorRT engine for edge devices.

---

## 4  Repository Layout *(initial)*

```
├── data/               # raw & processed datasets (git‑ignored)
├── src/
│   ├── datamodules/    # dataset + augmentations          (TBD)
│   ├── models/         # detection & classification nets  (TBD)
│   └── inference.py    # entry‑point for demo             (TBD)
│   └── gt_viz.py       # Video ground_truth visualizer
├── requirements.txt    # to be filled iteratively
└── README.md
``` 
---

## 5  Getting Started

```bash
# 1 Clone the repo
git clone https://github.com/berkeozkir/estu_project_cv.git
cd estu_project_cv

# 2 Create & activate a virtual env
python -m venv .venv && source .venv/bin/activate

# 3 Install requirements
pip install -r requirements.txt

```

## 5.1 Obtaining Dataset

Download this dataset: https://drive.google.com/file/d/1GZj6l84L5OD_ClQlh3uw17glbyPVmxhg/view  <br/>
Extract the zip and rename the folder as "data"

> **Note:** Large datasets and checkpoint files go in `data/` and are tracked with **Git LFS**


## 5.2 Exploratory Data Analysis

To visualize dataset and ground truth labels with classes and run simple YOLO inference please run the following python script

```python 
python src/gt_viz.py
```

![Alt text](assets/exploreGUI.png)

---

## 6  License

Unless course policy dictates otherwise, this project will be released under the **MIT License**.

---
