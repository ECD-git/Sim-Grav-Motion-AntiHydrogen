import numpy as np
import matplotlib.pyplot as plot

# define params needed, all relative
sigma = 1; # beam width
r = 10000; # partical init pos rel to beam centre
N=1000000
angle = np.random.uniform(0,np.asin(10**(-4)), size = N)

b = r*np.sin(angle)
b = b[(b <= sigma)]


Fig = plot.figure()
plot.hist(b, bins=200, density=True, alpha=0.5, color='blue')
plot.title('Histogram of b, r>>$\sigma$, $N=1* 10^{6}$')
plot.xlabel('Impact Parameter')
plot.ylabel('Density')
plot.show()