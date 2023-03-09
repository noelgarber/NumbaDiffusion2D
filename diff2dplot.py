# plotdens.py 
# Ramses van Zon
# SciNetHPC, 2016
#
_plotdens_vmin = -10.0
_plotdens_vmax = 10.0
_plotdens_image_number = 1
def plotdens(dens,x1,x2,first=False):
    '''plot a 2d density, using matplotlib's imshow. The density should be
    in a square array "dens", with x and y values ranging from x1 to
    x2 and density values plotted with a colormap as if they range
    from d1 to d2. "first" should be set to True upon first call,
    which will clear the figure and add the colorbar.
    '''
    import os
    if 'DISPLAY' in os.environ:
        import matplotlib.pyplot as plt
        global _plotdens_vmin
        global _plotdens_vmax
        global _plotdens_image_number
        global _im
        if first:
            _plotdens_vmin = min(min(d) for d in dens)
            _plotdens_vmax = 1.#max(max(d) for d in dens)
        if first:
            plt.clf()
            plt.ion()
            _im = plt.imshow(dens,interpolation='none',aspect='equal',extent=(x1,x2,x1,x2),vmin=_plotdens_vmin,vmax=_plotdens_vmax,cmap='nipy_spectral')
        else:
            _im.set_data(dens)
        if first:
            plt.colorbar()
        plt.show()
        plt.pause(0.5)
        #plt.savefig(f"image{_plotdens_image_number:03d}.png")
        _plotdens_image_number += 1
    else:
        print ("WARNING: No DISPLAY found for graphics")

