# Lesson 1: Python Foundations for ML & Robotics

**Phase 1 · Lesson 1 of 27**
**Estimated time:** 2–4 hours
**Status:** 🟡 In Progress
**Delivered:** April 27, 2026
**Prerequisite:** Lesson 0 (Environment Setup) — Python 3.11+, VS Code, Git working in a terminal.

---

## What you'll know by the end

- How Python actually works (variables, types, control flow, functions)
- The four core data structures (list, dict, tuple, set)
- Why NumPy exists and why every ML library uses it
- How to *think in arrays* — the mental shift from hobbyist to engineer

## The Big Idea

Python is the lingua franca of ML and robotics. It's not the fastest language — C++ is — but it's **glue**. PyTorch, ROS 2, Isaac Sim, and NVIDIA's DLI labs all expose Python APIs. Your job as an ML engineer isn't to write GPU kernels; it's to orchestrate them. Python is how you do that.

The most important mental shift in this lesson: **stop thinking in loops, start thinking in arrays.** `for i in range(1000000)` is slow Python. `np.array(...) * 2` runs on optimized C under the hood. That difference is everything in ML.

---

## Read

### Variables & Types

Python figures out types automatically. No declarations.

```python
name = "Joey"
age = 25
gpu_temp = 67.4
is_training = True
```

### The four core data structures

```python
# LIST — ordered, changeable
sensors = ["lidar", "camera", "imu"]
sensors.append("radar")

# DICT — key-value pairs
robot = {"name": "R2", "battery": 0.87, "online": True}

# TUPLE — ordered, UNchangeable
position = (3.4, 2.1, 0.0)

# SET — unique items, no order
unique_classes = {"person", "car", "person", "dog"}
```

### Functions

```python
def calculate_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

dist = calculate_distance((0, 0), (3, 4))  # 5.0
```

### NumPy — the array mindset

NumPy gives Python fast, multi-dimensional arrays. Robotics is *all* arrays: a camera frame is `(H × W × RGB)`, a point cloud is `(N × 3)`, neural net weights are arrays.

```python
import numpy as np

sensor_readings = np.array([23.1, 23.5, 22.9, 24.0, 23.7])
mean_temp = sensor_readings.mean()
celsius_to_fahrenheit = sensor_readings * 9/5 + 32  # vectorized, no loop

image = np.zeros((480, 640))
image[100:200, 100:200] = 255  # paint a white square
print(image.shape)  # (480, 640)
```

---

## Watch (~25 min, your choice)

- "Python in 100 Seconds" by Fireship (2 min, fastest overview)
- "NumPy Tutorial for Beginners" by Keith Galli (first 25 min, *the* NumPy intro)

---

## Build — Mini-Project: Sensor Data Analyzer

### Setup (one-time)

1. Open PowerShell or your WSL2 terminal
2. `cd C:\robotics-course` (or your chosen course root)
3. `python -m venv venv`
4. Activate: `venv\Scripts\activate` (PowerShell) or `source venv/bin/activate` (WSL2)
5. `pip install numpy matplotlib`

### Code — `projects/lesson-01/sensor_analyzer.py`

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
hours = np.linspace(0, 24, 1440)
temperature = 22 + 3 * np.sin(hours * np.pi / 12) + np.random.normal(0, 0.5, 1440)

print(f"Mean temperature: {temperature.mean():.2f} °C")
print(f"Max temperature: {temperature.max():.2f} °C")
print(f"Min temperature: {temperature.min():.2f} °C")
print(f"Standard deviation: {temperature.std():.2f} °C")

overheating_mask = temperature > 24
print(f"Minutes spent overheating: {overheating_mask.sum()}")

plt.figure(figsize=(10, 4))
plt.plot(hours, temperature, label="Temperature")
plt.axhline(y=24, color='r', linestyle='--', label="Overheat threshold")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (°C)")
plt.title("Joey's Robot — 24h Sensor Data")
plt.legend()
plt.savefig("sensor_plot.png")
plt.show()
```

Run: `python sensor_analyzer.py`

### Stretch

Add a battery voltage sensor (12.6V → slowly decaying) and plot both on twin y-axes. Search "matplotlib twin axes" only after struggling 15 minutes.

---

## Reflect — Answer these in your next message

1. Difference between a **list** and a **tuple**, and when would you choose one over the other?
2. Why is `np.array([1,2,3]) * 2` faster than `[x*2 for x in [1,2,3]]`?
3. In the project, what does `temperature > 24` actually return?
4. If a camera image is `shape = (1080, 1920, 3)`, what does each number mean?
5. What confused you, surprised you, or didn't click?

---

## Reference

- NumPy quickstart: https://numpy.org/doc/stable/user/quickstart.html
- Python tutorial: https://docs.python.org/3/tutorial/

---

**On lesson complete:** Update `01-progress-tracker.md` — mark Lesson 1 ✅, fill in date, append session log entry, and prep `lesson-02-ml-refresher.md`.
