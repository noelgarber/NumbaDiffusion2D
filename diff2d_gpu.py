#!/usr/bin/env python
#
# diff2d_gpu.py - Simulates two-dimensional diffusion on a square domain, with GPU acceleration.
#
# Original creator of CPU code: Ramses van Zon
# SciNetHPC, 2016
#
# Modified by: Noel Garber
#

import time

# import plotdens function
from diff2dplot import plotdens

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
    nper      = int(outtime/dt) # how many step s between snapshots
    if nper==0: nper = 1

    # Allocate arrays.
    x         = [x1+((i-1)*(x2-x1))/nrows for i in range(npnts)] # x values (also y values)
    dens      = [[0.0]*npnts for i in range(npnts)] # time step t
    densnext  = [[0.0]*npnts for i in range(npnts)] # time step t+1

    # Up to this point, processing time is negligible, less than a second. The bottleneck is ahead.

    # Initialize.
    simtime=0*dt
    for i in range(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in range(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens[i][j] = a*b

    # Processing time up to this point is also fairly negligible.

    # Output initial signal.
    print(simtime)
    if graphics:
        plotdens(dens, x[0], x[-1], first=True)

    # temporary array to hold laplacian
    laplacian = [[0.0]*npnts for i in range(npnts)]
    t0 = time.time()
    t1 = time.time()
    total_time_taken = 0
    # The following loop takes the vast majority of the time for this program.
    for s in range(nsteps):
        # Take one step to produce new density.
        for i in range(1,nrows+1):
            for j in range(1,ncols+1):
                laplacian[i][j] = dens[i+1][j]+dens[i-1][j]+dens[i][j+1]+dens[i][j-1]-4*dens[i][j]
        for i in range(1,nrows+1):
            for j in range(1,ncols+1):
                densnext[i][j] = dens[i][j] + (D/dx**2)*dt*laplacian[i][j]
        t2 = time.time()
        print("core calculations processing time:", t2 - t1, "s")
        # Swap array pointers so t+1 becomes the new t, and update simulation time.
        dens, densnext = densnext, dens
        simtime += dt
        # Plot and report time.
        if (s+1)%nper == 0:
            print(simtime)
            if graphics:
                plotdens(dens, x[0], x[-1])
        t3 = time.time()
        print("post-hoc computations time:", t3 - t2, "s")
        print("laplacian calculation time at s =", s, "is", t3 - t1, "s")
        total_time_taken = total_time_taken + (t2 - t1)
        print("Elapsed time calculating laplacians:", total_time_taken, "s")
        t1 = time.time()
    print("total time taken for calculating laplacians:", t1 - t0, "s")

if __name__ == '__main__':
    main()

