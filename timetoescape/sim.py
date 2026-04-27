import numpy as np
import matplotlib.pyplot as plot
import math

global MASS; MASS = 1.00784*1.66*(10**-27);
global K_B; K_B = 1.380649*(10**-23);
global NUMBER; NUMBER = 10000;
global INITTEMP; INITEMP = 0.001;
global INITPE; INITPE = 0;
global TRAPSIGMA;
global TRAPD;

# RANDOM GENERATORS
def UniRandSpace(N):
    return np.random.rand(N)

def UniRandVector():
    '''
    Returns a numpy vector uniformly distributed on the surface of a unit sphere, in cartesian coords (x,y,z)
    '''
    theta = np.acos(1-2*np.random.uniform(0,1))
    phi = np.random.uniform(0,2*np.pi)
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

# MAXWELL BOLTZMAN VELOCITY DISTRIBUTION
def MaxBoltCDF(vel, temp, PE):
    '''
    Maxwell boltzmann CDF for a velocity vel, temperature temp, potential energy PE.
    '''
    E = (0.5*MASS*(vel**2)) + PE
    return math.erf((E/(K_B*temp))**0.5) - (2/(np.pi**0.5)) * ((E/(K_B*temp))**0.5) * np.exp(-E/(K_B*temp))

def MaxBoltPDF(vel,temp,PE):
    '''
    Maxwell boltzmann PDF for a velocity vel, temperature temp, potential energy PE.
    '''
    E = 0.5*MASS*(vel**2) + PE
    return ((2/np.pi)**0.5)*2*((MASS/(K_B*temp))**0.5)*(E/(K_B*temp))*np.exp(-E/(K_B*temp))

def GetDistributedVelocity(prob, temp, PE, it=20, guess=0):
    '''
    Uses newton raphson iteration, it times, to get a velocity from a uniformly distributed variable prob, of temperature temp, potential energy PE.
    '''
    if(guess == 0):
        guess = ((2*K_B*temp)/MASS)**0.5
        #print(guess)
    for i in range(it):
        guess = guess - ((MaxBoltCDF(guess, temp, PE)-prob)/MaxBoltPDF(guess, temp, PE))
        #print(guess)
    return guess

def RandVelocities(N, temp, PE):
    '''
    Returns a numpy array of N maxwell boltzman distributed velocities
    '''
    uniforms = np.random.rand(N)
    velocities = np.vectorize(GetDistributedVelocity)(uniforms, temp, PE)
    return velocities

def DrawInitVels(velocities):
    Fig = plot.figure()
    plot.hist(velocities, bins=200, density=True, alpha=0.5, color='blue')
    x = np.linspace(np.min(velocities), np.max(velocities), 100)
    y = MaxBoltPDF(x, 0.001, 0)
    plot.plot(x,y,"k--",label = "Maxwell Boltzmann PDF")
    plot.title("Randomly Distributed Velocities, $T=${0}$k$, U={1}$j$, N={2}".format(INITEMP, INITPE, NUMBER))
    plot.xlabel('velocity')
    plot.ylabel('Density')
    plot.legend()
    plot.show()

# SIMULATION LOOP


def __main__():    
    velocities = RandVelocities(NUMBER,INITEMP,INITPE)
    #DrawInitVels(velocities)

__main__()
