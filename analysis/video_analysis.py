import cv2
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from datetime import datetime
import pickle

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from analysis.features import extract_features
 

# ======================
# CHARGER LE MODÈLE ML
# ======================
MODEL_PATH = "models/tremblement_model_improved.pkl"
ml_model = pickle.load(open(MODEL_PATH, "rb"))


# ======================
# FILTRE BANDE PASSANTE
# ======================
def bandpass_filter(signal, fs, low=3, high=8, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)


# ======================
# ANALYSE VIDEO
# ======================
def analyze_video(video_path, patient_age=60, result_dir="static/results"):

    if not os.path.exists(video_path):
        raise FileNotFoundError("Vidéo introuvable")

    os.makedirs(result_dir, exist_ok=True)

    # Mediapipe Hand
    base_options = python.BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    y_positions = []
    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(mp_image, int(timestamp_ms))

        if result.hand_landmarks:
            wrist = result.hand_landmarks[0][0]
            y_positions.append(wrist.y)

        timestamp_ms += 1000 / fps

    cap.release()

    # Sécurité
    if len(y_positions) < 30:
        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "file": os.path.basename(video_path),
            "disorder": "Non détectable",
            "interpretation": "Main non détectée",
            "graph": None
        }

    # ======================
    # SIGNAL
    # ======================
    signal = bandpass_filter(np.array(y_positions), fps)

    # ======================
    # FEATURES
    # ======================
    features = extract_features(signal, fps)
    tremor_type = ml_model['model'].predict([features])[0]

    severity_score = features[0] * features[4]  # amplitude × fréquence

    if severity_score < 0.002:
      severity = "Faible"
    elif severity_score < 0.005:
        severity = "Moyenne"
    else:
        severity = "Élevée"
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(fft_vals), 1 / fps)
    dominant_freq = abs(freqs[np.argmax(fft_vals)])
    amplitude = np.std(signal)

    # ======================
    # GRAPH
    # ======================
    plt.figure()
    plt.plot(signal)
    plt.title("Mouvement vertical du poignet")
    plt.xlabel("Frame")
    plt.ylabel("Position normalisée")

    graph_path = os.path.join(
        result_dir,
        f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(graph_path)
    plt.close()

    # ======================
    # INTERPRÉTATION
    # ======================
    interpretation = {
        "Normal": "Mouvement physiologique normal.",
        "Hyperkinetique": "Mouvement excessif – tremblement pathologique possible.",
        "Hypokinetique": "Mouvement lent et de faible amplitude.",
        "Ataxique": "Mouvement désorganisé et non coordonné."
    }.get(tremor_type, "Indéterminé")

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file": os.path.basename(video_path),
        "amplitude": round(amplitude, 4),
        "frequency": round(dominant_freq, 2),
        "tremor_type": tremor_type,
        "severity": severity,
        "interpretation": interpretation,
        "graph": "/" + graph_path.replace("\\", "/")
    }
