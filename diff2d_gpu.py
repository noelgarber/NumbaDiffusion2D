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

# cuda kernel
@cuda.jit('void(int32, int32, float32, float32, float32, float32[:,:], float32[:,:], float32[:,:])')
def update_density(gpu_nrows, gpu_ncols, gpu_D, gpu_dx, gpu_dt, gpu_laplacian, gpu_dens, gpu_densnext):
    i, j = cuda.grid(2)
    if i >= 1 and i <= gpu_nrows and j >= 1 and j <= gpu_ncols:
        gpu_laplacian_ij = gpu_dens[i + 1, j] + gpu_dens[i - 1, j] + gpu_dens[i, j + 1] + gpu_dens[i, j - 1] - 4 * gpu_dens[i, j]
        gpu_laplacian[i,j] = gpu_laplacian_ij
        gpu_densnext[i,j] = gpu_dens[i,j] + (gpu_D / gpu_dx ** 2) * gpu_dt * gpu_laplacian_ij

# driver routine
def main():
    t1 = time.time()

    # Script sets the parameters D, x1, x2, runtime, dx, outtime, and graphics:
    from diff2dparams import D, x1, x2, runtime, dx, outtime, graphics

    # Compute derived parameters.
    nrows     = int((x2-x1)/dx) # number of x points
    ncols     = nrows           # number of y points, same as x in this case.
    npnts     = nrows + 2       # number of x points including boundary points
    dx_recalc = (x2-x1)/nrows   # recompute (previous dx may not fit in [x1,x2])
    dt        = 0.25*dx_recalc**2/D    # time step size (edge of stability)
    nsteps    = int(runtime/dt) # number of steps of that size to reach runtime
    nper      = int(outtime/dt) # how many step s between snapshots
    if nper==0: nper = 1

    t2 = time.time()
    print("Time taken computing derived parameters:", t2 - t1, "s")

    # Initialize general device variables
    nrows_dev = cp.int32(nrows)
    ncols_dev = cp.int32(ncols)
    D_dev = cp.float32(D)
    dx_dev = cp.float32(dx_recalc)
    dt_dev = cp.float32(dt)

    # Declare device parameters
    max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    threadsperblock = max_threads_per_block
    blockspergrid = (nrows_dev + threadsperblock - 1) // threadsperblock

    t3 = time.time()
    print("Time taken computing general GPU-related variables:", t3 - t2, "s")

    # Allocate arrays.
    x = np.array([x1+((i-1)*(x2-x1))/nrows for i in range(npnts)]) # x values (also y values)
    dens = np.zeros((npnts, npnts)) # time step t
    densnext = np.zeros((npnts, npnts)) # time step t+1

    # Initialize.
    simtime=0*dt
    for i in np.arange(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in np.arange(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens[i][j] = a*b

    t4 = time.time()
    print("Time spent during initialization:", t4 - t3, "s")

    # Output initial signal.
    print(simtime)
    if graphics:
        plotdens(dens, x[0], x[-1], first=True)

    t5 = time.time()
    print("Time taken outputting initial signal:", t5 - t4, "s")

    # Initialize step-specific device variables to input into the kernel.
    laplacian_dev = cp.zeros((npnts, npnts), dtype=np.float32)  # temporary device array to hold the laplacian
    dens_dev = cp.asarray(dens, dtype=cp.float32)
    densnext_dev = cp.asarray(densnext, dtype=cp.float32)

    t6 = time.time()
    print("Time taken transferring arrays to device memory:", t6 - t5, "s")

    #TODO: The erasure of dens/dens_dev occurs at some point after this

    for s in np.arange(nsteps):
        print("Current Step:", s)
        t01 = time.time()

        # Take one step to produce new density.
        update_density[blockspergrid, threadsperblock](nrows_dev, ncols_dev, D_dev, dx_dev, dt_dev,
                                                       laplacian_dev, dens_dev, densnext_dev)

        t02 = time.time()
        #print("\tTime taken executing the update_density GPU kernel:", t02 - t01, "s")

        # Retrieve the new density at t+1 and assign so it becomes the new t, and update simulation time.
        dens_dev = densnext_dev
        dens = cp.asnumpy(dens_dev)
        simtime += dt
        # Plot and report time.
        if (s+1)%nper == 0:
            print(simtime)
            if graphics:
                plotdens(dens, x[0], x[-1])

        t03 = time.time()
        #print("\tTime taken preparing for another step:", t03 - t02, "s")

if __name__ == '__main__':
    main()



'''
for i in range(1,nrows+1):
    for j in range(1,ncols+1):
        laplacian[i][j] = dens[i+1][j]+dens[i-1][j]+dens[i][j+1]+dens[i][j-1]-4*dens[i][j]
for i in range(1,nrows+1):
    for j in range(1,ncols+1):
        densnext[i][j] = dens[i][j] + (D/dx**2)*dt*laplacian[i][j]
'''