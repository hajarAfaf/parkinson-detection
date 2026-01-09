import numpy as np
from scipy.fft import fft
from scipy.stats import entropy
def extract_features(signal, fs):
    """
    Extrait des features biomécaniques du mouvement
    """
    signal = np.array(signal)

    # Sécurité
    if len(signal) < fs:
        return None

    mean_val = np.mean(signal)
    # Amplitude
    amplitude = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2))    # Énergie
    peak_to_peak = np.ptp(signal)            # Amplitude max
    zcr = np.mean(np.diff(np.sign(signal)) != 0)
    # Vitesse
    velocity = np.diff(signal)
    mean_velocity = np.mean(np.abs(velocity))

    # Accélération
    acceleration = np.diff(velocity)
    mean_acceleration = np.mean(np.abs(acceleration))

    # Jerk (variation de l'accélération)
    jerk = np.diff(acceleration)
    mean_jerk = np.mean(np.abs(jerk))

    # Fréquence dominante
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(fft_vals), 1/fs)

    positive_freqs = freqs[freqs > 0]
    positive_fft = fft_vals[freqs > 0]

    dominant_freq = positive_freqs[np.argmax(positive_fft)]
    spectral_energy = np.sum(positive_fft**2)

    # ========= ENTROPIE =========
    norm_fft = positive_fft / np.sum(positive_fft)
    spectral_entropy = entropy(norm_fft)

    # Régularité (Parkinson = régulier)
    regularity = np.std(velocity)

    features = [
    # Biomécanique
    amplitude,
    mean_velocity,
    mean_acceleration,
    mean_jerk,
    regularity,

    # Signal
    rms_val,
    peak_to_peak,
    zcr,
    dominant_freq,
    spectral_energy,
    spectral_entropy
]
    return features
