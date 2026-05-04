# Lesson 4: PyTorch Fundamentals

**Phase 1 · Lesson 4 of 27**
**Estimated time:** 4–6 hours
**Status:** 🟡 In Progress
**Delivered:** April 27, 2026
**Prerequisite:** Lesson 3 complete. Comfortable with NumPy and dot products.

---

## What you'll know by the end

- What a **tensor** is (it's a NumPy array with two superpowers: GPU + autograd)
- How to put data on your **RTX 4070** and do math 30–100× faster
- What **autograd** does — automatic gradient calculation that powers all of training
- The four-step **PyTorch training loop** that every model from MLPs to GR00T uses
- How to build, train, and evaluate a neural network classifier — and compare it to the Random Forest from Lesson 2

## The Big Idea

**PyTorch is NumPy for the GPU, plus the ability to compute gradients automatically.** That's it. Everything else — neural networks, transformers, training, deployment — is built on those two ideas.

You already know NumPy. A `torch.Tensor` is the same shape, same math, same indexing. The two differences:

1. **Tensors can live on the GPU.** Move a tensor to CUDA → operations run on thousands of cores in parallel. This is why a model that takes a week to train on CPU takes 6 hours on your 4070.
2. **PyTorch tracks operations.** Tell it "I want to know how `loss` changes if I tweak this weight," and it gives you the answer for *every* weight at once. That's autograd. Without it, training a neural net is computational suicide; with it, it's `loss.backward()`.

Mental shift this lesson: stop thinking of training as math you do. Start thinking of it as **a four-step loop you call**. Every PyTorch model — including 7-billion-parameter foundation models — trains via the same loop you'll write today.

---

## Read

### Tensors

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # like np.array
x.shape         # torch.Size([2, 2])
x.dtype         # torch.float32
x @ x           # matmul, same as NumPy
x.sum()         # tensor(10.)
```

If you can write NumPy, you can write PyTorch tensor code. The API is ~95% identical.

### Devices — moving to the GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1000, 1000, device=device)   # built directly on GPU
y = torch.randn(1000, 1000, device=device)
z = x @ y                                     # runs on GPU, very fast
```

Two rules:

- **All tensors in an operation must be on the same device.** Mixing CPU and CUDA tensors in one op = error.
- **Move data to GPU once, do many ops, move results back.** Constant CPU↔GPU transfers kill performance.

`x.to("cuda")` and `x.to("cpu")` move tensors between devices.

### Autograd — the magic ingredient

Tell PyTorch to track gradients on a tensor with `requires_grad=True`:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 5 * x       # y = x² + 5x
y.backward()             # compute dy/dx
print(x.grad)            # tensor(11.)  — d/dx(x² + 5x) = 2x + 5 → 2(3)+5 = 11
```

You wrote a function. PyTorch built the computation graph behind the scenes. `backward()` walked back through it and gave you the derivative for free. **This is what makes training neural nets feasible.**

> **Interactive widget — Autograd Playground** (request in chat: "show me autograd widget")
> Slide x, watch y and dy/dx update live.

### Neural networks — `nn.Module`

A neural network is a stack of operations with learnable parameters. You define it as a class.

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyClassifier(nn.Module):
    def __init__(self, in_features=6, hidden=32, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)   # weights + bias
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))     # nonlinear activation
        return self.fc2(x)          # raw logits (one number per class)

model = TinyClassifier()
```

`nn.Linear(6, 32)` is just a learnable matrix `W` of shape `(32, 6)` and a bias `b` of shape `(32,)`. Forward pass: `y = x @ W.T + b`. Same matmul you saw in Lesson 3, with a `relu` (max(0, x)) in between.

### The training loop — four steps, forever

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    # 1. Forward — compute predictions
    logits = model(X_train)
    loss = criterion(logits, y_train)

    # 2. Zero the old gradients
    optimizer.zero_grad()

    # 3. Backward — compute new gradients via autograd
    loss.backward()

    # 4. Step — nudge weights in the direction that reduces loss
    optimizer.step()
```

That's the entire shape of training a neural network. Every. Single. Time. Whether you're training a 4-parameter linear regression or a 7-billion-parameter transformer, this loop is the heart of it.

> **Interactive widget — Training Loop** (request in chat: "show me training loop widget")
> Step through one iteration at a time. See the loss go down.

---

## Watch (~30 min, your choice)

- **Daniel Bourke — "PyTorch in 100 Seconds"** (~2 min, fastest possible overview)
- **PyTorch Lightning — "Tensors and Autograd"** (~10 min) — locks in autograd
- **Andrej Karpathy — "Neural Networks: Zero to Hero, Part 1"** (the spreadsheet one, ~25 min) — best mental model of training that exists

---

## Build — Mini-Project: Neural Net Classifier on IMU Data

You'll re-do Lesson 2's robot state classifier — but with a PyTorch neural network instead of scikit-learn's Random Forest. Same data, same problem. **Did the neural net beat the forest?** That's the question this project answers.

### Setup

In your `(venv)` PowerShell:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

This installs PyTorch with CUDA 12.4 wheels (compatible with your CUDA 13.2 driver — PyTorch ships fewer CUDA versions than NVIDIA's drivers support). Will be a ~2.5 GB download. If it fails, try `cu121` instead, or use the selector at https://pytorch.org/get-started/locally/.

Verify after install:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see your **RTX 4070** in the output. If `CUDA available: False`, we'll debug.

### Code — `projects/lesson-04/imu_neural_net.py`

```python
"""
Neural-net version of the robot state classifier from Lesson 2.
Same synthetic IMU data, same 3 classes, but trained as a PyTorch MLP on the GPU.
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

axes[0].plot(train_losses)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
axes[0].set_title("Training loss")
axes[0].grid(True)

axes[1].plot(test_accs)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Test accuracy over epochs")
axes[1].grid(True)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("Confusion matrix")

plt.tight_layout()
plt.savefig("imu_nn_results.png")
plt.show()
```

Run:

```powershell
python imu_neural_net.py
```

### What you should see

- A model summary (3 Linear layers totaling ~1,400 parameters)
- "Training on: cuda"
- Loss decreasing every 20 epochs, test accuracy climbing toward 0.95+
- Final accuracy comparable to (or slightly better than) the Random Forest from Lesson 2
- A 3-panel matplotlib figure: loss curve, accuracy curve, confusion matrix

### Stretch

1. **Try a bigger / smaller model.** Set `hidden=8` (tiny) or `hidden=128` (bigger). What happens to test accuracy and training time?
2. **Train longer.** Run 1000 epochs. Does it overfit? (Watch the gap.)
3. **Compare timing.** Print `time.time()` before and after the training loop. Move the data to CPU (`device = "cpu"`) and re-run. How much slower is it?

---

## Reflect — Answer in your next message

1. **What's a tensor**, in 2 sentences? How is it different from a NumPy array?
2. The training loop has four lines: `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, `loss = criterion(logits, y)`. **Put them in the correct order** for one training iteration.
3. What does `loss.backward()` actually compute? (Use words from Lesson 3 if helpful.)
4. Your neural net probably hit ~0.97 test accuracy on this data. Is this proof that neural nets are better than Random Forests? Why or why not?
5. If you saw `RuntimeError: Expected all tensors to be on the same device`, what's wrong and how would you fix it?

---

## Reference

- PyTorch official tutorial: https://pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch CUDA install: https://pytorch.org/get-started/locally/
- Karpathy's micrograd (autograd from scratch): https://github.com/karpathy/micrograd

---

**On lesson complete:** Update `01-progress-tracker.md` — mark Lesson 4 ✅, fill in date, append session log, prep `lesson-05-containers-docker.md`. **Phase 1 (Foundations) is then done.** 🎉
