import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from analysis.features import extract_features

np.random.seed(42)


def generate_normal_tremor(fs=30, duration=5, severity=0.5):
    """Signal normal: bruit blanc léger"""
    t = np.linspace(0, duration, int(fs * duration))
    # Bruit blanc gaussien faible
    signal = np.random.normal(0, 0.001 * severity, len(t))
    return signal


def generate_parkinsonian_tremor(fs=30, duration=5, severity=1.0):
    """Tremor parkinsonien: 4-6 Hz + rigidité"""
    t = np.linspace(0, duration, int(fs * duration))
    freq = np.random.uniform(4, 6) * severity  # 4-6 Hz typique
    amplitude = 0.003 * severity
    
    # Signal sinusoïdal + harmoniques + bruit
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal += 0.0005 * amplitude * np.sin(2 * np.pi * freq * 2 * t)  # Harmonique
    signal += np.random.normal(0, 0.0005 * severity, len(t))
    
    return signal


def generate_essential_tremor(fs=30, duration=5, severity=1.0):
    """Tremor essentiel: 8-12 Hz, plus haute fréquence"""
    t = np.linspace(0, duration, int(fs * duration))
    freq = np.random.uniform(8, 12) * severity  # 8-12 Hz
    amplitude = 0.002 * severity
    
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal += 0.0003 * amplitude * np.sin(2 * np.pi * freq * 1.5 * t)
    signal += np.random.normal(0, 0.0003 * severity, len(t))
    
    return signal


def generate_hyperkinetic_tremor(fs=30, duration=5, severity=1.0):
    """Mouvements involontaires: multi-fréquences irrégulières"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Plusieurs fréquences mélangées
    signal = 0.004 * severity * np.sin(2 * np.pi * 3 * t)
    signal += 0.003 * severity * np.sin(2 * np.pi * 5.5 * t)
    signal += 0.002 * severity * np.sin(2 * np.pi * 8 * t)
    signal += np.random.normal(0, 0.001 * severity, len(t))
    
    # Ajouter des pics aléatoires (mouvements involontaires)
    for _ in range(int(len(t) * 0.1)):
        idx = np.random.randint(0, len(t))
        signal[idx] += np.random.uniform(-0.003, 0.003) * severity
    
    return signal


def generate_hypokinetic_tremor(fs=30, duration=5, severity=1.0):
    """Bradykinésie: amplitude très réduite, mouvements lents"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Fréquence très basse, amplitude très réduite
    signal = 0.0003 * severity * np.sin(2 * np.pi * 1.5 * t)
    signal += np.random.normal(0, 0.0002 * severity, len(t))
    
    return signal


def generate_ataxic_tremor(fs=30, duration=5, severity=1.0):
    """Ataxie: incoordination avec irrégularités"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Signal irrégulier avec amplitude variable
    signal = np.random.uniform(-0.004, 0.004, len(t)) * severity
    
    # Ajouter des oscillations irrégulières
    freq = np.random.uniform(2, 4)
    signal += 0.002 * severity * np.sin(2 * np.pi * freq * t)
    
    # Variation d'amplitude (caractéristique de l'ataxie)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    signal *= envelope
    
    return signal


def generate_physiological_tremor(fs=30, duration=5, severity=1.0):
    """Tremor physiologique: 8-12 Hz (involontaire normal)"""
    t = np.linspace(0, duration, int(fs * duration))
    freq = np.random.uniform(8, 12)
    
    signal = 0.0005 * severity * np.sin(2 * np.pi * freq * t)
    signal += np.random.normal(0, 0.0002 * severity, len(t))
    
    return signal


# ============================================
# 2. GENERATION DU DATASET
# ============================================

X = []
y = []

fs = 30
n_samples_per_class = 300  # Plus d'exemples
severity_levels = [0.5, 1.0, 1.5]  # Différentes sévérités

anomaly_generators = {
    "Normal": generate_normal_tremor,
    "Parkinsonien": generate_parkinsonian_tremor,
    "Tremor_Essentiel": generate_essential_tremor,
    "Hyperkinetique": generate_hyperkinetic_tremor,
    "Hypokinetique": generate_hypokinetic_tremor,
    "Ataxie": generate_ataxic_tremor,
    "Physiologique": generate_physiological_tremor
}

print("📊 Génération du dataset...")
for anomaly_type, generator in anomaly_generators.items():
    for i in range(n_samples_per_class):
        # Varier la sévérité
        severity = np.random.choice(severity_levels)
        
        # Générer le signal
        signal = generator(fs=fs, duration=5, severity=severity)
        
        # Extraire les features
        features = extract_features(signal, fs)
        
        X.append(features)
        y.append(anomaly_type)
        
        if (i + 1) % 100 == 0:
            print(f"  {anomaly_type}: {i + 1}/{n_samples_per_class}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Dataset généré: {X.shape} samples, {X.shape} features")
print(f"Classes: {np.unique(y)}")

# ============================================
# 3. SPLIT TRAIN/TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================
# 4. ENTRAINEMENT DU MODELE
# ============================================

print("\n🤖 Entraînement du modèle...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Modèle entraîné!")

# ============================================
# 5. EVALUATION
# ============================================

print("\n📋 EVALUATION DU MODELE\n")

# Prédictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracies
train_acc = (y_pred_train == y_train).mean()
test_acc = (y_pred_test == y_test).mean()

print(f"Accuracy Train: {train_acc:.4f}")
print(f"Accuracy Test:  {test_acc:.4f}")

# Rapport détaillé
print("\n📊 Classification Report (Test Set):\n")
print(classification_report(y_test, y_pred_test))

# Matrice de confusion
print("\n🔍 Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# ============================================
# 6. FEATURE IMPORTANCE
# ============================================

print("\n⭐ Top 10 Features:")
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]

for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. Feature {idx}: {feature_importance[idx]:.4f}")

# ============================================
# 7. SAUVEGARDE
# ============================================

print("\n💾 Sauvegarde du modèle...")
with open("models/tremblement_model_improved.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'classes': model.classes_,
        'n_features': X.shape,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }, f)

print("✅ Modèle sauvegardé avec succès!")
