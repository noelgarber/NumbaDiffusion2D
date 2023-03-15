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
import sys
import time
from numba import cuda, int32, float64
from diff2dplot import plotdens

'''
Define the density update kernel using the heat equation, where: 
    T  = 2D numpy array of heat values to be evolved
    D  = thermal diffusivity coefficient
    dx = space step
    dt = time step
'''
@cuda.jit
def update_density(T, T_out, D, dx, dt, N_x, N_y):
    i, j = cuda.grid(2)
    if i >= 1 and i < N_x-1 and j >= 1 and j < N_y-1:
        laplacian_value = (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - 4*T[i][j]) / dx**2
        T_new = T[i,j] + (D*dt)*laplacian_value
        T_out[i,j] = T_new



'''
# Naive kernel with a new approach. Speed issues may be due to inefficient grid and block size, or inefficient
# memory management (global memory is slow compared to shared memory), or some combination of these factors.
@cuda.jit('void(float32[:,:], float32[:,:], int32, int32, float32, float32, float32)')
def update_density(input_dens_dev, output_densnext_dev, nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev):
    for i in range(1, nrows_dev + 1):
        for j in range(1, ncols_dev + 1):
            laplacian_ij = input_dens_dev[i + 1][j] + input_dens_dev[i - 1][j] + input_dens_dev[i][j + 1] + input_dens_dev[i][j - 1] - 4 * input_dens_dev[i][j]
            output_densnext_dev[i][j] = input_dens_dev[i][j] + (D_dev / dx_dev ** 2) * dt_dev * laplacian_ij



@cuda.jit('void(float64[:,:], float64[:,:], int32, int32, int32, int32, float64, float64, float64)')
def update_density(input_dens_dev, output_densnext_dev, nrows_dev, ncols_dev, block_dim_x, block_dim_y, D_dev, dx_dev, dt_dev):
    # Obtain D, dx, and dt, and block dimensions from constant memory as read-only pointers
    D_const = cuda.const.array_like(D_dev)
    dx_const = cuda.const.array_like(dx_dev)
    dt_const = cuda.const.array_like(dt_dev)
    bdimx_const = cuda.const.array_like(block_dim_x)
    bdimy_const = cuda.const.array_like(block_dim_y)

    i, j = cuda.grid(2)
    if i > 0 and i < nrows_dev+1 and j > 0 and j < ncols_dev+1:
        s_input_dens = cuda.shared.array((bdimx_const+2, bdimy_const+2), dtype=float64)
        s_input_dens[i % bdimx_const + 1, j % bdimy_const + 1] = input_dens_dev[i, j]
        if i % bdimx_const == 0 and i > 0:
            s_input_dens[0, j % bdimy_const + 1] = input_dens_dev[i-1, j]
        if i % bdimx_const == bdimx_const - 1 and i < nrows_dev:
            s_input_dens[bdimx_const + 1, j % bdimy_const + 1] = input_dens_dev[i+1, j]
        if j % bdimy_const == 0 and j > 0:
            s_input_dens[i % bdimx_const + 1, 0] = input_dens_dev[i, j-1]
        if j % bdimy_const == bdimx_const - 1 and j < ncols_dev:
            s_input_dens[i % bdimx_const + 1, bdimy_const + 1] = input_dens_dev[i, j+1]
        cuda.syncthreads()

        laplacian_ij = s_input_dens[i % bdimx_const + 2, j % bdimy_const + 1] + s_input_dens[i % bdimx_const, j % bdimy_const + 1] + s_input_dens[i % bdimx_const + 1, j % bdimy_const + 2] + s_input_dens[i % bdimx_const + 1, j % bdimy_const] - 4 * s_input_dens[i % bdimx_const + 1, j % bdimy_const + 1]
        output_densnext_dev[i, j] = input_dens_dev[i, j] + (D_const / dx_const ** 2) * dt_const * laplacian_ij
'''


'''
# TODO Use the cuda grid function etc. from here in the new kernel above
# Define the main kernel to evolve the density using the diffusion equations; first computes the laplacian and then evolves the density
@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], int32, int32, float64, float64, float64)')
def evolve_density_kernel(input_dens_dev, output_laplacian_dev, output_densnext_dev, nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev):
    # Define block and grid dimensions for the Laplacian kernel launched from within this kernel below
    blockDim_laplacian = (32, 32)
    gridDim_laplacian = ((nrows_dev + blockDim_laplacian[0] - 1) // blockDim_laplacian[0], (ncols_dev + blockDim_laplacian[1] - 1) // blockDim_laplacian[1])

    # Compute the laplacian
    laplacian_kernel[gridDim_laplacian, blockDim_laplacian](input_dens_dev, output_laplacian_dev, nrows_dev, ncols_dev)

    # Compute the new density
    i, j = cuda.grid(2)
    if i > 0 and i < nrows_dev + 1 and j > 0 and j < ncols_dev + 1:
        output_densnext_dev[i][j] = input_dens_dev[i][j] + (D_dev / dx_dev ** 2) * dt_dev * output_laplacian_dev[i][j]
'''

# driver routine
def main(verbose = False):
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

    print("\tNumber of steps:", nsteps, "\n\tSteps per snapshot:", nper) if verbose else None

    # Allocate arrays.
    x            = [x1+((i-1)*(x2-x1))/nrows for i in range(npnts)] # x values (also y values)
    blank_array  = np.zeros((npnts, npnts))
    dens_initial = blank_array.copy() # time step t

    # Initialize.
    print("Calculating initial density state... ", end="") if verbose else None
    simtime=0*dt
    for i in range(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in range(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens_initial[i][j] = a*b
    print("done") if verbose else None

    # Output initial signal.
    print("Simulation time:", simtime)
    if graphics:
        plotdens(dens_initial, x[0], x[-1], first=True)

    #---------------------- Begin GPU Processing Section ----------------------#

    # Set up the CUDA kernel grid and block sizes
    threadsperblock = (16, 16)
    blockspergrid_x = (nrows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ncols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # copy the initialized density array to device memory
    dens_dev = cuda.to_device(dens_initial)
    densnext_dev = cuda.to_device(blank_array)

    for s in range(nsteps):
        print("Step", s, "of", nsteps)
        t0 = time.time()

        # call the kernel function; results are passed into densnext_dev
        update_density[blockspergrid, threadsperblock](dens_dev, densnext_dev, D, dx, dt, nrows, ncols)
        t1 = time.time()
        print("\tkernel execution time:", (t1-t0)/1000, "ms") if verbose else None

        # copy results (densnext_dev at t+1) back to host memory
        dens_host = densnext_dev.copy_to_host()
        t2 = time.time()
        print("\tresults retrieval time:", (t2-t1)/1000, "ms")

        # update the simulation time
        simtime += dt  # update simulation time

        # prepare for next round
        dens_dev = cuda.to_device(dens_host)
        densnext_dev = cuda.to_device(blank_array)
        t3 = time.time()
        print("\ttime spent copying arrays to device for next round:", (t3-t2)/1000, "ms") if verbose else None

        # output snapshot if necessary
        if (s + 1) % nper == 0:
            print(simtime)
            if graphics:
                plotdens(dens_host, x[0], x[-1])

if __name__ == '__main__':
    main(verbose = True)
