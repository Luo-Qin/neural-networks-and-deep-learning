"""
valley
~~~~~~

Plots a function of two variables to minimize.  The function is a
fairly generic valley function."""

#### Libraries
# Third party
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy

fig = plt.figure()
ax = fig.gca(projection='3d')
X = numpy.arange(-1, 1, 0.1)
Y = numpy.arange(-1, 1, 0.1)
X, Y = numpy.meshgrid(X, Y)
Z = X**2 + Y**2

colortuple = ('w', 'b')
colors = numpy.empty(X.shape, dtype=str)
for x in xrange(len(X)):
    for y in xrange(len(Y)):
        colors[x, y] = colortuple[(x + y) % 2]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=True)

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0, 2)
ax.w_xaxis.set_major_locator(LinearLocator(3))
ax.w_yaxis.set_major_locator(LinearLocator(3))
ax.w_zaxis.set_major_locator(LinearLocator(3))

plt.show()
