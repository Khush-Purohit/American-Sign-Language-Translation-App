
# American Sign Language translation App

This Python application detects American Sign Language (ASL) gestures and translates them into English text in real-time. It utilizes a machine learning classification model based on Random Forest to accurately recognize hand signs and map them to their corresponding English letters or words, enabling seamless communication between sign language users and non-signers.

- Primary language: Python


## Features
- Real-time ASL gesture detection and translation to text
- End-to-end pipeline:
  - Collect training images from a webcam
  - Build a dataset and extract features
  - Train and evaluate a machine learning model (Random Forest)
  - Run an interactive application for live predictions
- Pretrained artifacts included (`model.pkl`, `scaler.pkl`, `data.pickle`) for quick start

- ### Prerequisites
- Python 3.x
- A working webcam
- pip


### Installation
```bash
# Clone the repository
git clone https://github.com/Khush-Purohit/American-Sign-Language-Translation-App.git
cd American-Sign-Language-Translation-App

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# To run the application
python main_application.py
```

## Project Structure

- [main_application.py](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/main_application.py) — Launches the real-time ASL recognition application.
- [collect_imgs.py](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/collect_imgs.py) — Captures training images from a webcam for dataset creation.
- [create_dataset.py](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/create_dataset.py) — Builds the dataset and serializes features to `data.pickle`.
- [train_test_model.py](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/train_test_model.py) — Trains and evaluates the Random Forest model; outputs `model.pkl` and `scaler.pkl`.
- [requirements.txt](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/requirements.txt) — Python dependencies.
- [data.pickle](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/data.pickle) — Serialized dataset (generated).
- [model.pkl](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/model.pkl) — Trained model artifact.
- [scaler.pkl](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/scaler.pkl) — Feature scaler artifact.
- [README.md](https://github.com/Khush-Purohit/American-Sign-Language-Translation-App/blob/main/README.md) — Project documentation.


## Tips

- If your webcam is not detected, you may need to adjust the camera index in the scripts (e.g., change `0` to `1`).


![Screenshot 2025-10-30 225002](https://github.com/user-attachments/assets/248a96a1-4c4c-48a8-8fd5-2518e2b816c7)

