# NumbaDiffusion2D

This is a simple GPU-accelerated Python script for simulating diffusion in 2D using the discretized heat equation. It was completed as the assignment for HPC133, a graduate course on GPU-accelerated programming through SciNet at the University of Toronto (https://scinet.courses). 

It uses Numba's just-in-time CUDA kernel compiling function to define a kernel that evolves a 2D array using the Laplace equation. 

The GPU-accelerated version is diff2d_gpu.py, as compared to the original unaltered diff2d.py provided as a CPU-driven example. 300-400x acceleration is obtained by using the GPU compared to pure Python.
