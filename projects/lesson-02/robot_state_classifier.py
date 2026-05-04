"""
Robot State Classifier
Generate synthetic IMU data for 3 robot states, train a Random Forest
to classify them, and evaluate with a confusion matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. Generate synthetic IMU data ----------
np.random.seed(42)
N_PER_CLASS = 500
FEATURES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
CLASSES = ["driving", "slipping", "tipped"]

# Class 0: driving — gravity in +z, very little movement
driving = np.random.normal(
    loc=[0, 0, 9.8, 0, 0, 0],
    scale=0.3,
    size=(N_PER_CLASS, 6),
)

# Class 1: slipping — wheels spinning, lots of horizontal accel & gyro noise
slipping = np.random.normal(
    loc=[0, 0, 9.8, 0, 0, 0],
    scale=[1.2, 1.2, 0.5, 1.5, 1.5, 0.3],
    size=(N_PER_CLASS, 6),
)

# Class 2: tipped — gravity vector rotates, z-accel reduced
tipped = np.random.normal(
    loc=[5, 5, 5, 0, 0, 0],
    scale=0.5,
    size=(N_PER_CLASS, 6),
)

X = np.vstack([driving, slipping, tipped])
y = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS + [2] * N_PER_CLASS)

print(f"Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
print(f"Class distribution: {np.bincount(y)}")

# ---------- 2. Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------- 3. Fit ----------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---------- 4. Predict ----------
y_pred = clf.predict(X_test)

# ---------- 5. Evaluate ----------
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Overfitting check
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test  accuracy: {test_acc:.3f}")
  

  
print(f"Gap (overfit indicator): {train_acc - test_acc:.3f}")

# Feature importance — which sensor channels mattered most?
print("\nFeature importance:")
for feat, imp in sorted(
    zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {feat:10s} {imp:.3f}")

# ---------- 6. Visualize confusion matrix ----------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Robot State Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()