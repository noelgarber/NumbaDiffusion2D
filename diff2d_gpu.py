#!/usr/bin/env python
#
# diff2d_gpu.py - Simulates two-dimensional diffusion on a square domain, with GPU acceleration.
#
# Original creator of CPU code: Ramses van Zon
# SciNetHPC, 2016
#
# Modified by: Noel Garber
#

# import plotdens function
from diff2dplot import plotdens
from numba import cuda

# kernel function for updating density
@cuda.jit
def update_density(dens, densnext, laplacian, nrows, ncols, D, dx, dt):
    i, j = cuda.grid(2)
    if 1 <= i < nrows + 1 and 1 <= j < ncols + 1:
        laplacian[i][j] = dens[i+1][j] + dens[i-1][j] + dens[i][j+1] + dens[i][j-1] - 4*dens[i][j]
        densnext[i][j] = dens[i][j] + (D/dx**2)*dt*laplacian[i][j]

# driver routine
def main():
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
    x         = [x1+((i-1)*(x2-x1))/nrows for i in range(npnts)] # x values (also y values)
    dens      = cuda.to_device([[0.0]*npnts for i in range(npnts)]) # time step t
    densnext  = cuda.to_device([[0.0]*npnts for i in range(npnts)]) # time step t+1
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
        plotdens(dens.copy_to_host(), x[0], x[-1], first=True)
    # temporary array to hold laplacian
    laplacian = cuda.to_device([[0.0]*npnts for i in range(npnts)])
    for s in range(nsteps):
        # Take one step to produce new density.
        update_density[(nrows, ncols), (16, 16)](dens, densnext, laplacian, nrows, ncols, D, dx, dt)
        # Swap array pointers so t+1 becomes the new t, and update simulation time.
        dens, densnext = densnext, dens
        simtime += dt