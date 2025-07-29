# importing the modules
import numpy as np 
import matplotlib.pyplot as plt 

# data to be plotted
x = np.arange(-21, 51) 
y = x * x * x * x - 30 * x * x * x - 3 * x * x + 2 * x - 6

# plotting
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.plot(x, y, color ="red") 
plt.show()