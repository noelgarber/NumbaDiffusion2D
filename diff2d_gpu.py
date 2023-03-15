#!/usr/bin/env python
#
# diff2d_gpu.py - Simulates two-dimensional diffusion on a square domain, with GPU acceleration.
#
# Original creator of CPU code: Ramses van Zon
# SciNetHPC, 2016
#
# Modified by: Noel Garber
# Course: HPC133, University of Toronto
# Instructor: Yohai Meiron
#

import numpy as np
import time
from numba import cuda
from diff2dplot import plotdens

'''
Define the density update kernel using the heat equation, where: 
    T  = 2D numpy array of heat values to be evolved
    D  = thermal diffusivity coefficient
    dx = space step
    dt = time step
'''
@cuda.jit('void(float64[:,:], float64[:,:], float64, float64, float64, int32, int32)')
def update_density(T, T_out, D, dx, dt, N_x, N_y):
    i, j = cuda.grid(2)
    if i >= 1 and i < N_x-1 and j >= 1 and j < N_y-1:
        laplacian_value = (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1] - 4*T[i][j]) / dx**2
        T_new = T[i,j] + (D*dt)*laplacian_value
        T_out[i,j] = T_new

# driver routine
def main(verbose = False):
    print("Beginning simulation...")
    start_time = time.time()

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

    # Allocate arrays.
    x            = [x1+((i-1)*(x2-x1))/nrows for i in range(npnts)] # x values (also y values)
    blank_array  = np.zeros((npnts, npnts))
    dens_initial = blank_array.copy() # time step t

    # Initialize.
    simtime=0*dt
    for i in range(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in range(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens_initial[i][j] = a*b

    # Output initial signal.
    print("Simulation time:", simtime)
    if graphics:
        plotdens(dens_initial, x[0], x[-1], first=True)

    #---------------------- Begin GPU Processing Section ----------------------#

    # Start CUDA routine
    cuda_start_time = time.time()

    # Set up the CUDA kernel grid and block sizes
    threadsperblock = (16, 16)
    blockspergrid_x = (nrows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ncols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # copy the initialized density array to device memory
    dens_dev = cuda.to_device(dens_initial)
    densnext_dev = cuda.to_device(blank_array)
    blank_array_dev = cuda.to_device(blank_array)

    for s in range(nsteps):
        # call the kernel function; results are passed into densnext_dev
        update_density[blockspergrid, threadsperblock](dens_dev, densnext_dev, D, dx, dt, nrows, ncols)

        # assign results (densnext_dev at t+1) to replace original input, and update the simulation time
        dens_dev = densnext_dev
        simtime += dt  # update simulation time

        # prepare for next round by blanking the receiving array
        densnext_dev = cuda.to_device(blank_array_dev)

        # output snapshot if necessary
        if (s + 1) % nper == 0:
            print("Simulation time:", simtime)
            if graphics:
                dens_host = dens_dev.copy_to_host()
                plotdens(dens_host, x[0], x[-1])

    # Report elapsed time for CUDA routine and host processes
    end_time = time.time()
    print("CUDA processing time:", end_time - cuda_start_time, "s")
    print("Host processing time:", cuda_start_time - start_time, "s")

if __name__ == '__main__':
    main()
