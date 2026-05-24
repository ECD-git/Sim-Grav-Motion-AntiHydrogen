import numpy as np
import matplotlib.pyplot as plot


global sigma; sigma = 200*(10**(-6))
global lambd; lambd = 1200*(10**(-9))
global z_R; z_R = (np.pi*(sigma**2))/(lambd);

def BeamWdith(z):
    return sigma*np.sqrt(1+(z/z_R)**2);

x = np.linspace(-0.20,0.20,50)
y = BeamWdith(x)

f = plot.figure()

plot.plot(x,y, color="blue")
plot.plot(x,-y, color="blue")
plot.grid()



plot.xlabel("Beam Width [m]")
plot.ylabel("Axial position [m]")

plot.show()


