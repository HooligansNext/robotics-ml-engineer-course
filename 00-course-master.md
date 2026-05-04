# Robotics ML Engineer Course — Master

**Owner:** Joey
**Goal:** Become a world-class Machine Learning Engineer specializing in Robotics
**Started:** April 27, 2026
**Format:** Lesson-by-lesson, multimodal, project-based
**Hardware:** Windows PC with NVIDIA GPU
**Time commitment:** Flexible

---

## Course Overview

A custom 6-phase curriculum to become a Robotics ML Engineer, built around NVIDIA's free Deep Learning Institute (DLI) self-paced courses, Isaac Sim, Isaac Lab, ROS 2, and the GR00T foundation model stack.

### Learning principles

- **Build-first:** Every concept paired with a mini-project
- **Multimodal:** Read + Watch + Build + Reflect + Reference for every lesson
- **Earn certifications** as natural milestones
- **Flexible pace** — quality over speed

---

## Full Curriculum Map (27 lessons + capstone)

### Phase 0 — Environment Setup (~3–5 days)

*Goal: A working dev environment so Phase 1 isn't blocked on tooling.*

| # | Lesson | Mini-Project |
|---|---|---|
| 0 | Windows + WSL2 + Python + Git + VS Code | Push a "hello world" repo to GitHub |

### Phase 1 — Foundations (~5–7 weeks)

*Goal: Real Python fluency, the math under the hood, and an applied ML refresher.*

| # | Lesson | Mini-Project |
|---|---|---|
| 1 | Python Foundations for ML & Robotics | Sensor Data Analyzer |
| 2 | ML Refresher with Real Robot Data | Train classifier on sensor data |
| 3 | Linear Algebra for ML & Robotics | Implement transforms + dot-product attention by hand |
| 4 | PyTorch Fundamentals | First neural net on GPU |

### Phase 2 — GPU & Accelerated Computing (~3–4 weeks)

*Goal: Master your GPU.*

| # | Lesson | Cert Target |
|---|---|---|
| 5 | Containers, Docker, NGC Catalog | — |
| 6 | CUDA Basics with Python | DLI: Fundamentals of Accelerated Computing with CUDA Python |
| 7 | Mixed Precision & TensorRT Intro | — |

### Phase 3 — Deep Learning for Perception (~5–7 weeks)

*Goal: Make robots see.*

| # | Lesson | Cert Target |
|---|---|---|
| 8 | Computer Vision Foundations (CNNs) | DLI: Getting Started with Deep Learning |
| 9 | Object Detection & Segmentation | DLI: Computer Vision for Industrial Inspection |
| 10 | Real-time Webcam Detection | — |
| 11 | Custom Segmentation on Synthetic Data | — |

### Phase 4 — Robotics Fundamentals (~7–9 weeks)

*Goal: Learn the actual robotics stack, the math that makes it work, and how robots reason about the world.*

| # | Lesson | Cert Target |
|---|---|---|
| 12 | ROS 2 (Humble/Jazzy) Basics | — |
| 13 | Coordinate Frames, Transforms, Kinematics | — |
| 14 | Building Your First Robot in Isaac Sim | DLI: Building Your First Robot in Isaac Sim |
| 15 | Sensors & Driving in Isaac Sim | — |
| 16 | URDF + OpenUSD Intro | NCP-OUSD: OpenUSD Development (Professional) |
| 17 | Probability & Statistics for Robotics ML | Estimate sensor noise; Bayesian update |
| 18 | State Estimation / SLAM Intro (Kalman, occupancy grids) | 1D Kalman filter on noisy lidar |

### Phase 5 — Physical AI & Reinforcement Learning (~6–8 weeks)

*Goal: Train robots to learn.*

| # | Lesson | Cert Target |
|---|---|---|
| 19 | RL Foundations (PPO, SAC) with PyTorch | DLI: Reinforcement Learning |
| 20 | Isaac Lab 3.0 — Train Policies in Sim (Newton physics) | — |
| 21 | Imitation Learning + Synthetic Data Generation (Replicator) | — |
| 22 | Sim-to-Real Concepts + Manipulation Project | — |
| 23 | NVIDIA GR00T (N1.7+) + VLA Foundation Models | — |

### Phase 6 — Cloud, SDKs, Deployment (~4–5 weeks)

*Goal: Deploy and scale.*

| # | Lesson | Cert Target |
|---|---|---|
| 24 | Cloud GPU Basics + NVIDIA Brev | — |
| 25 | Triton Inference Server | NCA-AIIO: AI Infrastructure & Operations |
| 26 | Safety & Reliability for Real Robots (failure modes, e-stops, eval harnesses) | — |
| 27 | (Optional) Jetson Deployment | — |

### Capstone (~4 weeks)

**Sim-to-real pick-and-place:** Train a robot arm in Isaac Lab to do pick-and-place, evaluate it on held-out scenes, document the methodology, and publish to GitHub as portfolio work.

---

## Certification Roadmap

| Phase | Certification | Cost | Status |
|---|---|---|---|
| 2 | DLI: Fundamentals of Accelerated Computing with CUDA Python | Free | ⬜ |
| 3 | DLI: Getting Started with Deep Learning | Free | ⬜ |
| 3 | DLI: Computer Vision for Industrial Inspection | Free | ⬜ |
| 4 | DLI: Building Your First Robot in Isaac Sim | Free | ⬜ |
| 4 | NCP-OUSD: OpenUSD Development (Professional) | Paid exam (~$400) | ⬜ |
| 5 | DLI: Reinforcement Learning | Free | ⬜ |
| 6 | NCA-AIIO: AI Infrastructure & Operations (Associate) | Paid exam (~$135) | ⬜ |

---

## Key NVIDIA Resources (Bookmarked)

- DLI Self-Paced Courses: https://www.nvidia.com/en-us/training/self-paced-courses/
- NVIDIA Certifications: https://www.nvidia.com/en-us/learn/certification/
- Robotics Learning Path: https://www.nvidia.com/en-us/learn/learning-path/robotics/
- Physical AI Learning Hub: https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/index.html
- Isaac Developer Hub: https://developer.nvidia.com/isaac
- Isaac Sim: https://developer.nvidia.com/isaac/sim
- Isaac Lab (open source): https://developer.nvidia.com/isaac/lab
- Isaac GR00T: https://developer.nvidia.com/isaac/gr00t
- Isaac-GR00T GitHub: https://github.com/NVIDIA/Isaac-GR00T
- Learn OpenUSD: https://docs.nvidia.com/learn-openusd/latest/index.html
- NCA-AIIO exam page: https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/
- NCP-OUSD exam page: https://www.nvidia.com/en-us/learn/certification/openusd-development-professional/
- NVIDIA Developer Program: https://developer.nvidia.com/

---

## Changelog

- **2026-04-27 (v2)** — Side quests inlined into main path: Linear Algebra (Lesson 3), Probability & Stats (Lesson 17), State Estimation/SLAM (Lesson 18), Safety & Reliability (Lesson 26). Total 27 lessons + capstone. Phase timings adjusted.
- **2026-04-27 (v1)** — Course package created. OpenUSD cert corrected (Professional, not Associate); Phase 0 environment-setup lesson added; containers moved before CUDA; transforms/kinematics lesson inserted; Isaac Lab 3.0 + GR00T N1.7 references applied; imitation learning added to Phase 5.
