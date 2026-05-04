# Lesson 2: ML Refresher with Real Robot Data

**Phase 1 · Lesson 2 of 27**
**Estimated time:** 3–5 hours
**Status:** 🟡 In Progress
**Delivered:** April 27, 2026
**Prerequisite:** Lesson 1 complete. Active venv with `numpy` and `matplotlib`.

---

## What you'll know by the end

- The basic ML vocabulary: features, labels, supervised vs unsupervised, classification vs regression
- The five-step ML workflow (data → split → fit → predict → evaluate)
- Why **train/test split** exists and what it protects you from (overfitting)
- How to read a **confusion matrix** and a `classification_report`
- How to train a real classifier on robot-shaped sensor data using **scikit-learn**

## The Big Idea

Machine learning is about learning a function from examples. Instead of writing rules ("if accel_z < 5, the robot has tipped"), you show the computer thousands of labeled examples and let it find the rules. That's it.

For a robotics ML engineer, the most common ML task is **classification on sensor data**: given the readings from this robot's IMU/lidar/camera, what state is it in? Is it driving, slipping, tipped? Is the human in front of it a pedestrian or a mannequin? Is this lidar pulse a wall, a person, or noise?

You won't always use deep learning for this. **Random forests and gradient boosted trees still beat neural nets on small tabular sensor datasets** — every working roboticist should know how to spin one up in 20 lines. That's what this lesson does.

The most important mental shift in this lesson: **always evaluate on data the model has never seen.** Training accuracy is meaningless. Test accuracy is the only number that tells you whether your model will work in the real world.

---

## Read

### The five-step ML workflow

Every supervised ML project, no matter how fancy, has the same five steps:

1. **Data:** features `X` (the inputs, shape `(n_samples, n_features)`) and labels `y` (the answers, shape `(n_samples,)`).
2. **Split:** carve off a chunk of data the model will never see during training — the **test set**. Typical split: 80% train / 20% test.
3. **Fit:** show the model the training data. It adjusts its internal parameters to minimize error.
4. **Predict:** ask the model what it thinks the labels are for the test set.
5. **Evaluate:** compare predictions to true labels. Compute metrics.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
```

That's the entire skeleton of supervised ML. Everything else — neural nets, transformers, GR00T — is variations on this same shape.

### Features and labels

A **feature** is one number describing one example. A **label** is the answer for that example.

For a wheeled robot's state classifier, features might be:

| accel_x | accel_y | accel_z | gyro_x | gyro_y | gyro_z | label |
|---|---|---|---|---|---|---|
| 0.1 | -0.2 | 9.7 | 0.0 | 0.1 | 0.0 | driving |
| 0.0 | 0.1 | 9.8 | 0.0 | -0.1 | 0.2 | driving |
| 1.4 | -1.2 | 9.4 | 1.8 | -1.1 | 0.5 | slipping |
| 5.2 | 5.1 | 4.8 | 0.1 | 0.0 | 0.1 | tipped |

Six features per row, one label per row. `X` is the table of features as a NumPy array; `y` is the column of labels.

### Why train/test split matters — overfitting

Imagine I asked you to memorize 100 specific photos and their labels, then tested you on those exact same photos. You'd get 100%. But you wouldn't have *learned* anything about photos in general — you'd just have memorized.

That's **overfitting**. A model that memorizes training data perfectly but fails on new data. The only way to detect it: hold out data the model never sees during training.

Rule: **train accuracy is for debugging. Test accuracy is the truth.** If train >> test, your model is overfitting.

### The confusion matrix

When you have more than 2 classes, raw "accuracy" hides a lot. The confusion matrix shows *which* classes get confused with which.

For 3 classes (driving, slipping, tipped), a confusion matrix is a 3×3 grid: rows are true labels, columns are predicted labels. Diagonal = correct. Off-diagonal = mistakes.

```
                    Predicted
                 driving  slipping  tipped
True  driving  [   95        4        1   ]   ← 100 actual driving
      slipping [   12       80        8   ]   ← 100 actual slipping
      tipped   [    0        2       98   ]   ← 100 actual tipped
```

Reading this: the model nails "driving" (95/100) and "tipped" (98/100) but confuses "slipping" with "driving" 12% of the time. That's a real, actionable insight that one accuracy number wouldn't tell you.

### scikit-learn

The library you use for classical (non-deep) ML in Python. Stable, fast, beautifully designed API. Every model has the same `.fit()` / `.predict()` / `.score()` interface, so swapping models is a one-line change.

You'll install it in a moment.

---

## Watch (~30 min, your choice)

- **StatQuest — "Random Forests, Clearly Explained"** (~10 min) — best intuition for the model you'll use
- **StatQuest — "Confusion Matrix Clearly Explained"** (~5 min) — locks in metrics
- **scikit-learn quickstart video** by sentdex or Keith Galli (~15 min) — optional sklearn intro

---

## Build — Mini-Project: Robot State Classifier

You'll generate synthetic IMU data for a wheeled robot in three states (driving, slipping, tipped), train a Random Forest to tell them apart, and evaluate it properly.

### Setup

In your `(venv)` PowerShell (still in `C:\robotics-course`):

```powershell
pip install scikit-learn seaborn
```

`seaborn` is for prettier confusion matrix plots. `scikit-learn` is the ML library.

### Code — `projects/lesson-02/robot_state_classifier.py`

Make a new folder and file:

```powershell
mkdir C:\robotics-course\projects\lesson-02
```

Then create `robot_state_classifier.py` in VS Code (same flow as last lesson) and paste:

```python
"""
Robot State Classifier
Generate synthetic IMU data for 3 robot states, then train a Random Forest
to classify them. Evaluate with a confusion matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. Generate synthetic IMU data ----------
np.random.seed(42)
N_PER_CLASS = 500  # 500 samples per state
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

# Compare train vs test accuracy — overfitting check
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
```

Run:

```powershell
python robot_state_classifier.py
```

### What you should see

- ~1500 total samples, 500 per class
- Train/test sizes (1200/300)
- A `classification_report` with precision, recall, f1-score per class
- A 3×3 confusion matrix
- Train and test accuracy printed (both should be high — likely 0.97–1.00)
- Feature importance ranking
- A heatmap window showing the confusion matrix in color

### Stretch

1. **Reduce the noise gap:** make the three classes harder to distinguish (e.g., set slipping `scale=[0.5, 0.5, 0.4, 0.6, 0.6, 0.2]`). Re-run. Watch test accuracy drop. Watch the confusion matrix get messier.
2. **Compare models:** swap `RandomForestClassifier` for `LogisticRegression` (also from sklearn). Same `.fit()` / `.predict()` API. Which is better on this data?
3. **Real data instead of synthetic:** download the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — IMU data from real phones. Same 5-step workflow, real-world accuracy.

---

## Reflect — Answer these in your next message

1. In your own words, what's the difference between **classification** and **regression**? Give one example of each from a real robotics task.
2. Why does the code use `stratify=y` in `train_test_split`? What would go wrong without it?
3. Your model probably hit ~99% test accuracy. Is that impressive? Why or why not? (Hint: think about how you generated the data.)
4. What does **feature importance** tell you, and why might a robotics engineer care?
5. If your model had train accuracy 0.99 and test accuracy 0.55, what's happening, and what would you do about it?

---

## Reference

- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- StatQuest YouTube channel: https://www.youtube.com/@statquest
- UCI HAR dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

**On lesson complete:** Update `01-progress-tracker.md` — mark Lesson 2 ✅, fill in date, append session log entry, prep `lesson-03-linear-algebra.md`.
