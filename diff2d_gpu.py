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
from numba import cuda
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
@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], int32, int32, float64, float64, float64)')
def evolve_density_kernel(input_dens_dev, output_laplacian_dev, output_densnext_dev, nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev):
    for i in range(1, nrows_dev + 1):
        for j in range(1, ncols_dev + 1):
            output_laplacian_dev[i][j] = input_dens_dev[i+1][j] + input_dens_dev[i-1][j] + input_dens_dev[i][j+1] + input_dens_dev[i][j-1] - 4*input_dens_dev[i][j]
    for i in range(1, nrows_dev + 1):
        for j in range(1, ncols_dev + 1):
            output_densnext_dev[i][j] = input_dens_dev[i][j] + (D_dev/dx_dev**2)*dt_dev*output_laplacian_dev[i][j]

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

    # perform the computationally heavy component on the GPU as follows
    for s in range(nsteps):
        print("Step", s, "of", nsteps)

        # redefine output arrays as blanks
        output_laplacian_dev = cp.zeros((npnts, npnts)) # temporary array to hold the laplacian; default 64-bit floats
        output_densnext_dev = cp.zeros((npnts, npnts)) # temporary array to hold the newly evolved density; default 64-bit floats

        # define block and grid dimensions to fit the data
        blockDim = (32, 32)
        gridDim = ((nrows + blockDim[0] - 1) // blockDim[0], (ncols + blockDim[1] - 1) // blockDim[1])

        # execute the kernel
        evolve_density_kernel[gridDim, blockDim](dens_dev, output_laplacian_dev, output_densnext_dev, nrows, ncols, D, dx, dt)

        # copy the results to become the new density array on device memory
        dens_dev = output_densnext_dev.copy()
        simtime += dt # update the simulation time

        # free up device memory before the next round
        del output_densnext_dev, output_laplacian_dev

        # Plot and report time.
        if (s+1)%nper == 0:
            print(simtime)
            if graphics:
                print("Plotting graphics...")
                dens_host = cp.asnumpy(dens_dev)
                plotdens(dens_host, x[0], x[-1])

if __name__ == '__main__':
    main()
