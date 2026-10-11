# GPU Fluid & Particle Simulator

 A high-performance, real-time Smoothed Particle Hydrodynamics (SPH) fluid and granular dynamics simulation running on the GPU using Taichi and Python.

## Features
 - Real-time SPH fluid mechanics running on GPU (Apple Metal / CUDA / Vulkan).
 - Spatial grid hashing for accelerated particle neighborhood search.
 - Pressure, viscosity, and surface tension force solvers.
 - High-framerate interactive GUI with velocity and density shader color-coding.

## Installation

 ```bash
 git clone https://github.com/Seyo2412/gpu-fluid-particle-simulator.git
 cd gpu-fluid-particle-simulator
 python3 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
 ```

 ## Usage

 ```bash
 python main.py
 ```

## Roadmap

 - [ ] Core SPH physics kernel (density, pressure, viscosity)
 - [ ] Spatial grid hashing for neighbor search
 - [ ] Interactive GPU viewport with velocity-based shaders
 - [ ] Obstacle collision and container interaction
 - [ ] Automated tests & demo GIF export