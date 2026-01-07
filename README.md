# DD2360-RayTracing

A CUDA-based GPU ray tracer developed for **DD2360 Applied GPU Programming (KTH)**. 
This project follows a profiling-driven optimisation workflow starting from a brute-force baseline and progressing to two acceleration structures: an octree and a GPU-oriented Linear Bounding Volume Hierarchy (LBVH).

Based on the CUDA implementation from 
https://github.com/rogerallen/raytracinginoneweekendincuda by Roger Allen.

---

## Branches

This repository is organised into three main branches:

- `baseline` 
  Brute-force CUDA ray tracer without acceleration structures.

- `octree` 
  Ray tracer extended with an **octree** spatial subdivision structure to prune ray–object intersection tests.

- `lbvh` 
  Ray tracer extended with an **LBVH** (Linear BVH) using Morton codes and a linear memory layout for GPU-friendly construction and traversal.

The `lbvh` branch introduces additional headers (e.g. `aabb.h`, `lbvh.h`, `morton.h`) that are not present in `baseline` and `octree`.

---

## Requirements

- CUDA Toolkit 11.0+
- C++ compiler supported by your CUDA toolkit
- A CUDA-capable NVIDIA GPU

---

## Build and Render (Makefile)

```bash
make
make out.ppm        # render and write output to out.ppm
eog out.ppm         # open the image with Eye of GNOME
```

- `make` builds the CUDA executable (`./cudart`)
- `make out.ppm` runs the renderer and saves the result to `out.ppm`
- `eog out.ppm` opens the rendered image

---

## Profiling and Benchmarking

### Nsight Systems (end-to-end timeline)

```bash
nsys profile -o lbvh --stats=true ./cudart > /dev/null
nsys-ui lbvh.nsys-rep
```

- Records a full system timeline and kernel breakdown
- Identifies dominant kernels such as `render`, `create_world`, and `free_world`
- `> /dev/null` suppresses image output during profiling

---

### Nsight Compute (single-kernel analysis)

```bash
ncu -k render --set basic --replay-mode kernel --target-processes all \
./cudart > ncu_kernel_basic.txt 2>&1
```

- Profiles only the `render` kernel
- Collects metrics such as occupancy, memory throughput, and compute throughput
- Outputs results to `ncu_kernel_basic.txt`

---

## Runtime Parameters (configured in `main.cu`)

This project does **not** provide a command-line interface. 
All experimental parameters are configured directly in `main.cu`.

```cpp
int nx = 1200;
int ny = 800;
int ns = 10;
int tx = 8;
int ty = 8;

const int GRID_RADIUS = 11;
const int NUM_EXTRA_SPHERES = 3;
const int NUM_GROUND = 1;

const int NUM_GRID_SPHERES = (2 * GRID_RADIUS) * (2 * GRID_RADIUS);
const int NUM_HITABLES     = NUM_GRID_SPHERES + NUM_EXTRA_SPHERES + NUM_GROUND;
```

- `nx`, `ny`: image resolution 
- `ns`: samples per pixel 
- `tx`, `ty`: CUDA block dimensions 
- `GRID_RADIUS`: controls scene complexity (number of spheres)

---

## Project Structure

Core source file:
- `main.cu` 
  Entry point and rendering logic (kernels, configuration, scene construction)

Acceleration and precision (branch-dependent):
- `lbvh.h` – LBVH construction and traversal 
- `morton.h` – Morton code generation 
- `aabb.h` – Axis-aligned bounding boxes 

Rendering components (mostly inherited from the base implementation):
- `camera.h` – Camera model (depth of field support) 
- `hitable.h`, `hitable_list.h` – Abstract ray–object interface and scene container 
- `material.h` – Material models (Lambertian, metal, dielectric) 
- `sphere.h` – Sphere primitive 
- `ray.h` – Ray definition 
- `vec3.h` – Vector mathematics

---

## Profiling Artifacts

The repository may contain profiling and analysis outputs such as:

- `*.nsys-rep` (Nsight Systems reports) 
- `ncu_*.txt` (Nsight Compute logs) 
- `out.ppm` (rendered images)

These are used to support the performance evaluation presented in the final report.

---

## Contributors

- Yuanqing Wang 
- Yuxuan Sun 
- Yiyao Zhang 
