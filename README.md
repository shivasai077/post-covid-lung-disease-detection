# Post COVID Lung Disease Detection

This project is a web application for detecting COVID-19, Pneumonia, or Normal conditions from chest X-ray images using a deep learning model (Xception). The app is built with Flask and provides precautionary advice based on the prediction.

## Features

- Upload chest X-ray images via a web interface.
- Predicts one of three classes: **COVID-19**, **Normal**, or **Pneumonia**.
- Displays recommended precautions for each class.
- Uses a transfer learning model (Xception) trained on your dataset.

## Folder Structure

```
post covid lung disease detection/
│
├── app.py
├── main.py
├── requirements.txt
├── lung_covid_pneumonia_detection_xception.keras  # (created after training)
│
├── dataset/
│   ├── train/
│   │   ├── covid/
│   │   ├── normal/
│   │   └── pneumonia/
│   ├── val/
│   │   ├── covid/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── test/
│       ├── covid/
│       ├── normal/
│       └── pneumonia/
│
├── static/
│   └── uploads/
│
├── templates/
│   └── upload.html
│
└── venv/
```

## Setup Instructions

### 1. Clone the Repository

Download or clone this folder to your local machine.

### 2. Create and Activate a Virtual Environment

```sh
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Prepare the Dataset

- Place your chest X-ray images in the appropriate folders under `dataset/train`, `dataset/val`, and `dataset/test`.
- Each class (`covid`, `normal`, `pneumonia`) should have its own subfolder.

## Execution Steps

Follow these steps to train the model and run the web application:

### 1. Train the Model

```sh
python main.py
```
- This will train the model and save it as `lung_covid_pneumonia_detection_xception.keras` in the project directory.

### 2. Run the Web Application

```sh
python app.py
```
- Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage

1. On the web page, upload a chest X-ray image.
2. The app will predict the class and display precautionary advice.
3. You can upload another image as needed.

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies:
- tensorflow
- flask
- opencv-python
- numpy
- Pillow

## Notes

- Make sure the model file (`lung_covid_pneumonia_detection_xception.keras`) exists before running the web app.
- The app uses the Xception model with transfer learning; you can modify `main.py` for other architectures if needed.
- For best results, use a balanced and clean dataset.

---

**Developed by CSMA15 | Guided by Mr. P. Sai Kumar**