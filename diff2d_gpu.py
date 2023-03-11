#!/usr/bin/env python
#
# diff2d_gpu.py - Simulates two-dimensional diffusion on a square domain, with GPU acceleration.
#
# Original creator of CPU code: Ramses van Zon
# SciNetHPC, 2016
#
# Modified by: Noel Garber
#

import numpy as np
import cupy as cp
import time
from numba import cuda, int32, float64
from diff2dplot import plotdens

'''
Define a CUDA kernel that evolves the density array. As input, it takes the following variables, stored in device memory: 
    input_dens_dev:         the 2D density array for the previous time step
    output_laplacian_dev:   a blank 2D array of equal shape to the density array; 
                            note that the laplacian from the previous time step is not taken as an argument and thus not required
    output_densnext_dev:    a blank 2D array of equal shape to the density array; used to output the newly evolved density
    nrows_dev:              the number of rows to iterate over
    ncols_dev:              the number of cols to iterate over
    D_dev, dx_dev, dt_dev:  input parameters
'''

'''
@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], int32, int32, float64, float64, float64)')
def evolve_density_kernel(input_dens_dev, output_laplacian_dev, output_densnext_dev, nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev):
    for i in range(1, nrows_dev + 1):
        for j in range(1, ncols_dev + 1):
            output_laplacian_dev[i][j] = input_dens_dev[i+1][j] + input_dens_dev[i-1][j] + input_dens_dev[i][j+1] + input_dens_dev[i][j-1] - 4*input_dens_dev[i][j]
    for i in range(1, nrows_dev + 1):
        for j in range(1, ncols_dev + 1):
            output_densnext_dev[i][j] = input_dens_dev[i][j] + (D_dev/dx_dev**2)*dt_dev*output_laplacian_dev[i][j]
'''


@cuda.jit('void(float64[:,:], float64[:,:], int32, int32)', device=True)
def laplacian_kernel(input_dens_dev, output_laplacian_dev, nrows_dev, ncols_dev):
    # Define shared memory
    shared_input = cuda.shared.array(shape=(16, 16), dtype=float64)
    shared_output = cuda.shared.array(shape=(16, 16), dtype=float64)

    # Load shared memory with input density values
    i, j = cuda.grid(2)
    si, sj = cuda.gridsize(2)
    for si_idx in range(si):
        for sj_idx in range(sj):
            shared_input[si_idx, sj_idx] = input_dens_dev[i + si_idx, j + sj_idx]
    cuda.syncthreads()

    # Compute the laplacian using shared memory
    if i > 0 and i < nrows_dev + 1 and j > 0 and j < ncols_dev + 1:
        shared_output[cuda.threadIdx.x, cuda.threadIdx.y] = (
                shared_input[cuda.threadIdx.x + 1, cuda.threadIdx.y] +
                shared_input[cuda.threadIdx.x - 1, cuda.threadIdx.y] +
                shared_input[cuda.threadIdx.x, cuda.threadIdx.y + 1] +
                shared_input[cuda.threadIdx.x, cuda.threadIdx.y - 1] -
                4 * shared_input[cuda.threadIdx.x, cuda.threadIdx.y]
        )

    cuda.syncthreads()

    # Write shared output back to global memory
    for si_idx in range(si):
        for sj_idx in range(sj):
            if i + si_idx > 0 and i + si_idx < nrows_dev + 1 and j + sj_idx > 0 and j + sj_idx < ncols_dev + 1:
                output_laplacian_dev[i + si_idx, j + sj_idx] = shared_output[si_idx, sj_idx]


@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], int32, int32, float64, float64, float64)')
def evolve_density_kernel(input_dens_dev, output_laplacian_dev, output_densnext_dev, nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev):
    # Compute the laplacian
    laplacian_kernel[(nrows_dev + 14) // 16, (ncols_dev + 14) // 16](input_dens_dev, output_laplacian_dev, nrows_dev, ncols_dev)

    # Compute the new density
    i, j = cuda.grid(2)
    if i > 0 and i < nrows_dev + 1 and j > 0 and j < ncols_dev + 1:
        output_densnext_dev[i][j] = input_dens_dev[i][j] + (D_dev / dx_dev ** 2) * dt_dev * output_laplacian_dev[i][j]


# driver routine
def main():
    print("Beginning simulation...")

    # Script sets the parameters D, x1, x2, runtime, dx, outtime, and graphics:
    from diff2dparams import D, x1, x2, runtime, dx, outtime, graphics

    # Compute derived parameters.
    nrows     = int((x2-x1)/dx) # number of x points
    ncols     = nrows           # number of y points, same as x in this case.
    npnts     = nrows + 2       # number of x points including boundary points
    dx        = (x2-x1)/nrows   # recompute (previous dx may not fit in [x1,x2])
    dt        = 0.25*dx**2/D    # time step size (edge of stability)
    nsteps    = int(runtime/dt) # number of steps of that size to reach runtime
    nper      = int(outtime/dt) # how many steps between snapshots
    if nper==0: nper = 1

    print("\tNumber of steps:", nsteps)
    print("\tSteps per snapshot:", nper)

    # Allocate arrays.
    x         = [x1+((i-1)*(x2-x1))/nrows for i in range(npnts)] # x values (also y values)
    dens      = [[0.0]*npnts for i in range(npnts)] # time step t
    densnext  = [[0.0]*npnts for i in range(npnts)] # time step t+1

    # Initialize.
    simtime=0*dt
    for i in range(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in range(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens[i][j] = a*b

    # Output initial signal.
    print(simtime)
    if graphics:
        plotdens(dens, x[0], x[-1], first=True)

    # copy the initialized density array; becomes 64-bit float array in device memory
    dens_dev = cp.array(dens)

    # define block and grid dimensions to fit the data
    blockDim = (32, 32)
    gridDim = ((nrows + blockDim[0] - 1) // blockDim[0], (ncols + blockDim[1] - 1) // blockDim[1])

    print("Block size:", blockDim[0] * blockDim[1])
    print("Grid size:", gridDim[0] * gridDim[1])

    # perform the computationally heavy component on the GPU as follows

    for s in range(nsteps):
        print("Step", s, "of", nsteps)
        t1 = time.time()

        # redefine output arrays as blanks
        output_laplacian_dev = cp.zeros((npnts, npnts))  # temporary array to hold the laplacian; default 64-bit floats
        output_densnext_dev = cp.zeros((npnts, npnts))  # temporary array to hold the newly evolved density; default 64-bit floats

        t2 = time.time()
        print("\ttime making input arrays:", t2 - t1, "s")

        # call the kernel function with optimized block and grid sizes
        evolve_density_kernel[gridDim, blockDim](dens_dev, output_laplacian_dev, output_densnext_dev, nrows, ncols, D, dx, dt)
        dens_dev = output_densnext_dev
        simtime += dt  # update simulation time

        t3 = time.time()
        print("\ttime executing the kernel:", t3 - t2, "s")

        # output snapshot if necessary
        if (s + 1) % nper == 0:
            print(simtime)
            if graphics:
                dens = dens_dev.get()
                plotdens(dens, x[0], x[-1])

if __name__ == '__main__':
    main()
