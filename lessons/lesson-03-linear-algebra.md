# Lesson 3: Linear Algebra for ML & Robotics

**Phase 1 · Lesson 3 of 27**
**Estimated time:** 3–4 hours
**Status:** 🟡 In Progress
**Delivered:** April 27, 2026
**Prerequisite:** Lesson 2 complete. NumPy comfortable.

---

## What you'll know by the end

- What a **vector**, a **matrix**, and a **tensor** actually are (geometrically and in code)
- The two linear algebra operations that account for ~90% of all ML compute: **dot product** and **matrix multiplication**
- How a **2D rotation matrix** turns a "rotate this point cloud" task into a single multiply
- Why the dot product is the soul of the **attention mechanism** that powers GR00T, GPT, and basically every modern model

## The Big Idea

Linear algebra is the language ML and robotics speak. Every neural network is a stack of matrix multiplies. Every transform between coordinate frames (camera → robot base, end-effector → world) is a matrix. Every point cloud is a matrix. Every batch of images is a tensor.

The good news: you only need a small handful of operations to get extremely far. Not eigenvalues today, not SVD, not Jordan form. Just:

1. **Vectors** as arrows in space (and as columns of numbers).
2. **Dot product** — the single most important operation in ML.
3. **Matrix-vector multiply** — how transformations are applied.
4. **Rotation matrix** — the physical/geometric meaning of a matrix.

The mental shift: stop thinking "math equation." Start thinking **"a transform is a function, a matrix is the function written down, and matmul is calling it."**

---

## Read

### Vectors

A vector is a list of numbers. Geometrically, it's an arrow from the origin to a point.

```python
import numpy as np
v = np.array([3, 4])     # 2D vector — points 3 right, 4 up
v.shape                  # (2,)
np.linalg.norm(v)        # 5.0  — length (Pythagoras)
```

In robotics, vectors are everywhere: a position `(x, y, z)`, a velocity `(vx, vy, vz)`, an RGB color `(r, g, b)`, a feature vector for an ML model.

> **Interactive widget — Vector Playground** (request in chat: "show me vector playground")
> Drag two vector heads. Watch addition and dot product update live.

### Dot product

Two vectors of the same length. Multiply element-by-element, sum the result.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)             # 1*4 + 2*5 + 3*6 = 32
```

What it *means*: the dot product measures **alignment**. Two vectors pointing the same direction have a large positive dot product. Perpendicular vectors have dot product 0. Opposite vectors have a large negative dot product.

```
a · b = |a| * |b| * cos(angle between them)
```

This is why dot products power attention in transformers: "how aligned is this query with each key?" High alignment → high attention weight.

### Matrices

A matrix is a 2D grid of numbers. Shape `(rows, cols)`. In NumPy:

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
M.shape                  # (2, 3)
```

Useful intuitions:

- A 2×3 matrix is **two rows of length 3** — or **three columns of length 2** — or **a function from 3D vectors to 2D vectors** (more on this in matrix-vector multiply).
- A camera image `(H, W, 3)` is technically a 3D **tensor**. A point cloud `(N, 3)` is a matrix.

### Matrix-vector multiplication

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
v = np.array([10, 20, 30])
M @ v                    # array([140, 320])
```

The `@` operator means "matrix multiply." Mechanically: each row of `M` does a dot product with `v` to produce one output number.

The deep insight: **`M @ v` is a function call.** `M` is a function from R³ → R². `v` is the input. The result is the output. This is how every neural network layer works — a matrix `W` of learned weights times an input vector.

> **Interactive widget — Matrix-Vector Multiply** (request in chat: "show me matmul widget")
> Edit `M` and `v` in a grid, see the output update.

### Rotation matrix — the physical meaning

A 2D rotation matrix rotates a vector by angle θ counterclockwise:

```
R(θ) = [ cos θ, -sin θ ]
       [ sin θ,  cos θ ]
```

Apply it: `R @ v` gives `v` rotated by θ. Apply it to every point in a point cloud: `R @ points.T` rotates the entire cloud.

This is THE workhorse of robotics. Every transform between coordinate frames (camera-to-base, end-effector-to-world) is a rotation matrix combined with a translation. ROS 2's TF library is essentially "manage thousands of these in a tree."

> **Interactive widget — Rotation Playground** (request in chat: "show me rotation widget")
> Slide an angle, watch a robot's point cloud rotate around the origin.

---

## Watch (~30 min, your choice)

- **3Blue1Brown — "Essence of Linear Algebra" Ch. 1, 3, 4** (~30 min total) — *the* visual intro, irreplaceable
- **3Blue1Brown — "Visual intro to dot products"** (~10 min) — locks in the alignment intuition

---

## Build — Mini-Project: Rotate a Point Cloud + Hand-Coded Attention

Two small builds. The first is robotics-flavored (rotate a point cloud), the second is ML-flavored (implement scaled dot-product attention by hand).

### Setup

In your `(venv)` PowerShell:

```powershell
mkdir C:\robotics-course\projects\lesson-03
cd C:\robotics-course\projects\lesson-03
```

No new pip installs — NumPy and matplotlib only.

### File 1 — `rotate_point_cloud.py`

```python
"""
Generate a 2D point cloud shaped like a robot, then rotate it by a sequence
of angles using a rotation matrix. Plot before and after.
"""
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix_2d(theta):
    """Returns the 2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Make a "robot" point cloud (an arrow shape)
points = np.array([
    [0, 0], [2, 0], [2, 0.5], [3, 0.5], [3, -0.5], [2, -0.5], [2, -1], [0, -1],
])  # shape (8, 2) — 8 points, each (x, y)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
angles = [0, np.pi / 4, np.pi / 2, np.pi]  # 0°, 45°, 90°, 180°

for ax, theta in zip(axes, angles):
    R = rotation_matrix_2d(theta)
    rotated = points @ R.T  # apply rotation to every point at once
    ax.plot(*rotated.T, 'o-')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_title(f"θ = {np.degrees(theta):.0f}°")

plt.tight_layout()
plt.savefig("rotated_robot.png")
plt.show()

print("Verification — rotating by 90° then 90° more should equal 180°:")
R90 = rotation_matrix_2d(np.pi / 2)
R180 = rotation_matrix_2d(np.pi)
test_point = np.array([1, 0])
print(f"R90 @ R90 @ [1, 0] = {R90 @ R90 @ test_point}")
print(f"R180     @ [1, 0] = {R180 @ test_point}")
```

Run it. You should see four panels — robot shape rotated 0°, 45°, 90°, 180° — and a verification print showing two ways of getting to 180° give (almost exactly) the same answer.

### File 2 — `manual_attention.py`

This is the heart of every transformer. You'll implement it in ~15 lines using only NumPy.

```python
"""
Scaled dot-product attention from scratch.
Given queries Q, keys K, values V, compute:
    attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # how aligned each Q is with each K
    weights = softmax(scores, axis=-1)      # turn into probabilities
    return weights @ V, weights              # weighted sum of values + the weights themselves

# Tiny example: 3 tokens, each represented by a 4-dim vector
np.random.seed(0)
Q = np.random.randn(3, 4)  # 3 queries
K = np.random.randn(3, 4)  # 3 keys
V = np.random.randn(3, 4)  # 3 values

output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention output shape:", output.shape)
print("\nAttention weights (rows sum to 1):")
print(np.round(weights, 3))
print("\nRow sums (should all be 1.0):", weights.sum(axis=1))
```

Run it. The output is the contextualized representation of each token — a weighted blend of the values, where the weights come from how each query aligns with each key. This exact pattern is what makes a transformer work.

### Stretch

1. **3D rotation:** write `rotation_matrix_z(theta)` for a 3D z-axis rotation. Apply it to a 3D point cloud.
2. **Compose transforms:** rotate by 30°, then translate by `(2, 1)`. Show that order matters (rotate-then-translate ≠ translate-then-rotate).
3. **Attention with masking:** in a transformer decoder, you don't want token #2 attending to token #5 (it can't see the future). Add a mask that sets future scores to `-inf` before the softmax.

---

## Reflect — Answer in your next message

1. **In English**, what does the dot product `a · b` *measure*? (Two-sentence intuition.)
2. If `M` has shape `(4, 5)` and `v` has shape `(5,)`, what's the shape of `M @ v`? Why?
3. The rotation matrix uses `cos` and `sin`. Why does that work — what's the geometric reason `cos θ` and `sin θ` show up in the matrix?
4. In the attention code, why divide `Q @ K.T` by `sqrt(d_k)` before the softmax? (Hint: think about what happens to softmax when its inputs get very large.)
5. What part felt mechanical and what part felt magical? (No wrong answer.)

---

## Reference

- 3Blue1Brown — Essence of Linear Algebra: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- NumPy linear algebra docs: https://numpy.org/doc/stable/reference/routines.linalg.html
- Attention is All You Need (the original transformer paper): https://arxiv.org/abs/1706.03762

---

**On lesson complete:** Update `01-progress-tracker.md` — mark Lesson 3 ✅, fill in date, append session log, prep `lesson-04-pytorch-fundamentals.md`.
