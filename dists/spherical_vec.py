import numpy as np
import matplotlib.pyplot as plot

N = 1000
V_tot = 1

def RandAngles(N):
    angles = np.random.random((N,2))
    angles[:,1] = angles[:,1]*2*np.pi # phi
    angles[:,0] = np.acos(1-2*angles[:,0]) # theta
    return angles

def RandUnitVectors(N):
    angles = RandAngles(N)
    unitVectors = np.zeros((N,3))
    unitVectors[:,0] = np.sin(angles[:,0])*np.cos(angles[:,1]) #x
    unitVectors[:,1] = np.sin(angles[:,0])*np.sin(angles[:,1]) #y
    unitVectors[:,2] = np.cos(angles[:,0]) #z
    return unitVectors

def TransverseVelocityTest(N):
    unitVectors = RandUnitVectors(N)
    transverseVelocityUVs = np.sqrt((unitVectors[:,0]**2) + (unitVectors[:,1]**2))
    Angles = RandAngles(N)
    transverseComp = np.sin(Angles[:,0])

    #for i in range(len(transverseVelocityUVs)):
    #    retur = [(unitVectors[i,0]), (unitVectors[i,1]), unitVectors[i,2], transverseVelocityUVs[i]]
    #    print(retur)
    
    Fig = plot.figure()
    plot.hist(transverseComp, bins=1000, density=True, alpha=0.5, color='blue')
    plot.title("Histogram of uniform random $v_t$ values, $N=${0}".format(N))
    plot.xlabel('Transverse Velocity')
    plot.ylabel('Density')
    plot.show()

def SphereVectorTest(N):
    unitVectors = RandUnitVectors(N)

    fig = plot.figure()
    ax = plot.axes(projection='3d')
    ax.scatter(unitVectors[:,0], unitVectors[:,1], unitVectors[:,2])
    ax.set_title('Uniformly Weighted Random Unit Vector Plot, N={0:3.2f}'.format(np.size(unitVectors, axis=0)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plot.show()

def AnglesTest(N):
    angles = RandAngles(N)

    figure = plot.figure()
    plot.scatter(angles[:,1]/np.pi, angles[:,0]/np.pi)
    plot.xlabel(r"$\phi$ /$\pi$")
    plot.ylabel(r"$\theta$ /$\pi$")
    plot.plot([0,2], [0.5,0.5], 'k--')
    plot.title("Uniform Rand Spherical Unit Vector Angles, N={0}".format(N))
    plot.show()

#AnglesTest(1000)
TransverseVelocityTest(100000)
#SphereVectorTest(1000)