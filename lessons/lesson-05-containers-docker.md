# Lesson 5: Containers, Docker, and the NGC Catalog

**Phase 2 · Lesson 5 of 27**
**Estimated time:** 2–3 hours (mostly waiting for Docker Desktop to install)
**Status:** 🟡 In Progress
**Delivered:** April 27, 2026
**Prerequisite:** Phase 1 complete. WSL2 enabled (you confirmed it earlier).

---

## What you'll know by the end

- What a **container** is and why every modern ML/robotics project ships as one
- The difference between an **image** (the recipe) and a **container** (a running instance)
- How to pull a pre-built **NVIDIA NGC** image and run PyTorch inside it
- How to pass your **GPU** through to a container (`--gpus all`)
- Why this skill is non-negotiable for Isaac Sim, Isaac Lab, Triton, and any cloud GPU work

## The Big Idea

A container is a way to take an entire Linux environment — OS libraries, Python version, ML framework versions, CUDA toolkit, your code — bundle it up, and run it on **any** machine with Docker installed. Identically. Every time.

Compare it to your last week:

- **Without containers** (what you did): install Python carefully, fight with the MS Store version, set up a venv, install PyTorch with the right CUDA wheels, hope it all works.
- **With containers**: `docker pull nvcr.io/nvidia/pytorch:25.04-py3` → a fully working Python + PyTorch + CUDA environment, GPU-ready, identical for everyone.

This is why every NVIDIA product — Isaac Sim, Isaac Lab, Triton, GR00T — ships as a container. NVIDIA's engineers built it once and tested it once, and you run *that exact thing*.

The mental shift this lesson: stop thinking of "installing software" as the path. Start thinking of containers as **shipping an entire computer in a file.** When your robot perception model goes to production, it goes inside a container. When your sim-to-real RL training runs in the cloud, it runs in a container. This is the actual unit of work in ML/robotics.

---

## Read

### Image vs Container — the bakery metaphor

- An **image** is a *recipe + ingredients*. Static. Portable. You can have many copies.
- A **container** is a *cake in the oven*. A running instance of an image. You can run many containers from the same image at the same time.

```
Image (PyTorch 25.04)  →  docker run  →  Container #1 (running)
                                      →  Container #2 (running)
                                      →  Container #3 (stopped)
```

You pull images. You run containers. Most images come from a **registry** — a server that hosts them.

### Registries — Docker Hub vs NGC

- **Docker Hub** — the general-purpose default registry. Most open-source images live here.
- **NVIDIA NGC Catalog** (`nvcr.io`) — NVIDIA's registry. Hosts GPU-optimized containers: PyTorch with the latest CUDA, TensorRT, RAPIDS, Isaac Sim, Triton, etc. Free for individual use.

You'll use NGC for everything in this course beyond Phase 2.

### Dockerfile — the recipe

A Dockerfile is a text file describing how to build an image. Tiny example:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3   # start from NVIDIA's PyTorch image
COPY my_model.py /app/                  # copy your code in
RUN pip install opencv-python            # add any extra deps
CMD ["python", "/app/my_model.py"]      # what to run when the container starts
```

You won't write Dockerfiles today — you'll just *pull* and *run* a pre-built one. Writing them is a Phase 6 skill.

### GPU passthrough — `--gpus all`

By default, containers can't see your host GPU. Add `--gpus all` to `docker run` and your RTX 4070 Ti SUPER becomes visible inside the container, complete with CUDA. This is how Isaac Sim and Isaac Lab access your GPU — they're running inside containers.

> **Interactive widget — Container Lifecycle** (request: "show me container widget")
> Visualize the image → run → container → exec flow.

---

## Watch (~20 min, your choice)

- **Fireship — "Docker in 100 Seconds"** (~2 min, fastest possible overview)
- **Docker official intro** (~10–15 min) on YouTube
- **NVIDIA NGC overview** at https://catalog.ngc.nvidia.com/

---

## Build — Mini-Project: Run NVIDIA PyTorch in a Container

You'll install Docker Desktop, pull the official NVIDIA PyTorch container, run it with GPU access, and verify it sees your RTX 4070 Ti SUPER.

### Setup — install Docker Desktop

1. Download **Docker Desktop for Windows** at https://www.docker.com/products/docker-desktop/
2. Run the installer. **When asked: keep "Use WSL 2 instead of Hyper-V" CHECKED.**
3. After install, restart Windows when prompted.
4. Launch Docker Desktop. Accept the EULA.
5. Wait for the Docker engine to finish starting (whale icon in system tray turns from animated to steady).

Docker Desktop on Windows ships with NVIDIA GPU support **built in** (no separate NVIDIA Container Toolkit needed in 2026). It just works.

### Verify

In a fresh PowerShell:

```powershell
docker --version
docker info
```

You should see Docker version 27.x or later and `info` should print without errors. If it errors with "Docker daemon not running," wait another minute for Desktop to finish booting.

### Project — pull and run NVIDIA PyTorch

```powershell
docker pull nvcr.io/nvidia/pytorch:25.04-py3
```

(That's a ~9 GB download — go grab water.)

When it finishes, run it interactively with GPU access:

```powershell
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3 bash
```

What this command says:
- `docker run` — start a container
- `--gpus all` — pass through every GPU on the host
- `-it` — interactive + attach a terminal
- `--rm` — delete the container when I exit (no leftover junk)
- `nvcr.io/nvidia/pytorch:25.04-py3` — which image to use
- `bash` — what to run inside

You'll land at a shell prompt that looks like `root@<hash>:/workspace#` — that's a fresh Ubuntu container running on your machine. Let's verify the GPU works inside it:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

You should see your **RTX 4070 Ti SUPER** detected inside the container, just like outside it. PyTorch and CUDA are pre-installed (you're getting NVIDIA's official build).

Type `exit` to leave the container. With `--rm`, it disappears.

### Stretch

1. **Run a Python script from the host inside the container** by mounting a folder:

```powershell
docker run --gpus all -it --rm -v C:\robotics-course\projects\lesson-04:/work nvcr.io/nvidia/pytorch:25.04-py3 bash
# inside container:
cd /work
python imu_neural_net.py
```

The container reads files from your host. This is the canonical way to run code inside containers.

2. **Find an Isaac Sim image** at https://catalog.ngc.nvidia.com/ — search for "isaac-sim". You don't need to pull it (it's huge, ~50 GB), just look at the size and tags. This is what you'll use in Phase 4.

3. **List running containers, then images:**
```powershell
docker ps        # running
docker ps -a     # all (including stopped)
docker images    # local images
```

---

## Reflect — Answer in your next message

1. **Image vs container** — explain the difference in two sentences.
2. Why do you think NVIDIA ships Isaac Sim as a container instead of a regular installer?
3. What does `--gpus all` do, and what would happen if you forgot it when running a PyTorch training job?
4. You have two ML projects with conflicting Python dependencies (one needs PyTorch 1.13, the other needs PyTorch 2.6). How do containers solve this?
5. **Compare** — running `python imu_neural_net.py` directly on your host vs running it inside the NVIDIA PyTorch container. What's the same? What's different?

---

## Reference

- Docker docs: https://docs.docker.com/
- NGC Catalog: https://catalog.ngc.nvidia.com/
- NVIDIA PyTorch container release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

---

**On lesson complete:** Update `01-progress-tracker.md`, fill in `portfolio/lesson-05/README.md` (it'll be a short one — "set up Docker and ran NVIDIA PyTorch container with GPU passthrough"), prep `lesson-06-cuda-python.md`. Lesson 6 is the cert prep lesson.
