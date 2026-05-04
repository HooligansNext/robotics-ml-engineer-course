"""
Neural-net version of the robot state classifier from Lesson 2.
Same data, same problem, but trained as a PyTorch MLP on the GPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------- 1. Data — same as Lesson 2 ----------
np.random.seed(42)
N_PER_CLASS = 500
CLASSES = ["driving", "slipping", "tipped"]

driving = np.random.normal([0, 0, 9.8, 0, 0, 0], 0.3, (N_PER_CLASS, 6))
slipping = np.random.normal([0, 0, 9.8, 0, 0, 0], [1.2, 1.2, 0.5, 1.5, 1.5, 0.3], (N_PER_CLASS, 6))
tipped = np.random.normal([5, 5, 5, 0, 0, 0], 0.5, (N_PER_CLASS, 6))

X = np.vstack([driving, slipping, tipped]).astype(np.float32)
y = np.array([0]*N_PER_CLASS + [1]*N_PER_CLASS + [2]*N_PER_CLASS)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. Move to GPU ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)
X_test_t  = torch.from_numpy(X_test).to(device)
y_test_t  = torch.from_numpy(y_test).to(device)

# ---------- 3. Define the model ----------
class IMUClassifier(nn.Module):
    def __init__(self, in_features=6, hidden=32, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = IMUClassifier().to(device)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------- 4. Train ----------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
n_epochs = 200

train_losses, test_accs = [], []

for epoch in range(n_epochs):
    model.train()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).argmax(dim=1)
        test_acc = (test_pred == y_test_t).float().mean().item()

    train_losses.append(loss.item())
    test_accs.append(test_acc)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f} | test acc {test_acc:.4f}")

# ---------- 5. Final evaluation ----------
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).argmax(dim=1).cpu().numpy()
    test_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

train_acc = (train_pred == y_train).mean()
test_acc = (test_pred == y_test).mean()
print(f"\nFinal train acc: {train_acc:.3f}")
print(f"Final test  acc: {test_acc:.3f}")
print(f"Gap: {train_acc - test_acc:.3f}")

print("\nClassification report:")
print(classification_report(y_test, test_pred, target_names=CLASSES))

cm = confusion_matrix(y_test, test_pred)

# ---------- 6. Visualize ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(train_losses); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
axes[0].set_title("Training loss"); axes[0].grid(True)
axes[1].plot(test_accs); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Test accuracy over epochs"); axes[1].grid(True)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True"); axes[2].set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("imu_nn_results.png")
plt.show()