import numpy as np
import matplotlib.pyplot as plot
import math

# physical
global MASS; MASS = 1.00784*1.66*(10**-27);
global K_B; K_B = 1.380649*(10**-23);
# simulation
global NUMBER; NUMBER = 10000;
global INITTEMP; INITEMP = 0.001;
global INITPE; INITPE = 0;
global TIMESTEP; TIMESTEP = 0.001; # make sure to design sim so timestep can be changed at a later date
# trap
global TRAPSIGMA; TRAPSIGMA = 200*(10**-6) #[m]
global TRAPD; TRAPD = 5*0.01 #[m]
global BEAMSWITCHPERIOD; BEAMSWITCHPERIOD = 1 #[ns]

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

def UniRandVectorSpace(N):
    '''
    Returns an NP array of uniform random unit vectors 
    '''
    angles = np.random.random((N,2))
    angles[:,1] = angles[:,1]*2*np.pi # phi
    angles[:,0] = np.acos(1-2*angles[:,0]) # theta

    unitVectors = np.zeros((N,3))
    unitVectors[:,0] = np.sin(angles[:,0])*np.cos(angles[:,1]) #x
    unitVectors[:,1] = np.sin(angles[:,0])*np.sin(angles[:,1]) #y
    unitVectors[:,2] = np.cos(angles[:,0]) #z
    return unitVectors, angles

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

# PARTICLE PATHING

def MFPPDF(x):
    '''
    Returns the PDF of the probability the particle has entered the beam after a transverse path x
    Using mean free path theory, PDF(x) = 1-e^-(simga^2/d^2)x
    '''
    return 1-np.exp(-(TRAPSIGMA**2/TRAPD**2)*x)

# sim
def simulate():
    # Initialise variables of simulation
    VELOCITIES = RandVelocities(NUMBER,INITEMP,INITPE)
    FREEPATHS = np.zeros(NUMBER)
    TIME = 0

    # -- Main LOOP
    # - move particles + redist vel
    TIME += TIMESTEP # iterate by one step
    directions, angles = UniRandVectorSpace(np.size(VELOCITIES)) # get current direction of all particles
    transComps = np.sqrt(directions[:,0]**2 + directions[:,1]**2) # get all transverse components
    FREEPATHS += VELOCITIES*transComps*TIMESTEP # get the amount each particle travels by

    # - check if a particle is crossing the beam
    # ie if a random num (0,1] is less than the value of the PDF at that MFP, then its crossing the beam
    crossingBeamIndex = np.where(UniRandSpace(len(FREEPATHS)) < np.vectorize(MFPPDF)(FREEPATHS))
    # - If crossing beam
    # get impact params
    impactParams = TRAPSIGMA*np.sin(angles[:,1]) 
    # get beam path
    crossLengths = 2*np.sqrt(TRAPSIGMA**2-impactParams**2)
    # get kick numbers
    beamTimes = crossLengths/(VELOCITIES*transComps)
    nKicks = beamTimes/(BEAMSWITCHPERIOD*(10**-9))

    # reset the MFP of effected particles to 0
    FREEPATHS[crossingBeamIndex] = 0
    
    

    # REMOVE ANY ESCAPED PARTICLES
    # Important to maintain that VELOCITIES and FREEPATHS are always the same lenght



def __main__():    
    np.atan(1/0)
    #simulate()
    #DrawInitVels(velocities)
    

__main__()
