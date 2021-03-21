import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2

# np.arange(start, stop, step) to give us smoother line
x = np.arange(0, 5, 1e-3)
y = f(x)

plt.figure()
plt.plot(x, y)

# the point and the 'close enough' point
p2_delta = 1e-4
x1 = 2
x2 = x1 + p2_delta
y1 = f(x1)
y2 = f(x2)

print((x1,y1), (x2,y2))

# derivative approximation and y-intercept for the tangent line
approximate_derivative = (y2-y1)/(x2-x1)
b = y2 -  approximate_derivative*x2

# we put the tangent line calculation into a function so we can call 
# it multiple times for different values of x
# approximate_derivative and b are constant for given function
# thus calculated once above this function 
def tangent_line(x):
    return approximate_derivative*x + b

# plotting the tangent line
# +- 1 to draw the tangent line on our graph 
# then we calculate the y for given x using the tangent line function 
# Matplotlib will draw a line for us through these points
to_plot = [x1-1, x1, x1+1]
plt.plot(to_plot, [tangent_line(i) for i in to_plot])
plt.savefig('graph1.png')

print('approximate derivative for f(x)', 
    f'where x = {x1} is {approximate_derivative}')

