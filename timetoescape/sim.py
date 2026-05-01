import numpy as np
import matplotlib.pyplot as plot
import math

# physical
global MASS; MASS = 1.00784*1.66*(10**-27);
global K_B; K_B = 1.380649*(10**-23);
global C; C = 3*(10**8);
global ALPHA_FACTOR; ALPHA_FACTOR=6.67*(10**-31) #[m^3], alpha/4*pi*epsilon_0
# simulation
global NUMBER; NUMBER = 1000;
global INITTEMP; INITEMP = 0.001; # kelvin
global INITPE; INITPE = 0;
global THRESHOLD; THRESHOLD = 0.2; # kelvin
global TIMESTEP; TIMESTEP = 0.01; # make sure to design sim so timestep can be changed at a later date
global MAXPASSES; MAXPASSES = 10000;

# trap
global TRAPSIGMA; TRAPSIGMA = 200*(10**-6) #[m]
global TRAPD; TRAPD = 5*0.01 #[m]
global BEAMSWITCHPERIOD; BEAMSWITCHPERIOD = 1*(10**-9) #[s]
global BEAMLAMBDA; BEAMLAMBDA = 1200*(10**-9) # [m]
global BEAMPOWER; BEAMPOWER = 1 #[W]

# we are using the expectation value of the energy change per kick since there will be like 10**6 kicks per beam pass
global BEAMSTRENGTH; BEAMSTRENGTH = ALPHA_FACTOR * ((2*BEAMPOWER)/(C*(TRAPSIGMA**2)))

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
    meanFreePath = (np.pi*(TRAPD**2))/(2*TRAPSIGMA)
    return 1-np.exp(-x/meanFreePath)

def RandomMFPTime(vel_t):
    '''
    Gets a randomly distributed time until next beam pass following a MFP distribution
    '''
    meanFreePath = (np.pi*(TRAPD**2))/(2*TRAPSIGMA)
    return -(meanFreePath/vel_t)*np.log(1-np.random.random_sample())

# RANDOM WALK STATISTICS
def RandomWalkPDF(n,N):
    return (1/(np.sqrt(2*np.pi*N))) * (np.exp(-(n**2)/(2*N)))

def RandomWalkCDF(n,N):
    return 0.5*math.erf(n/np.sqrt(2*N))

def GetDistributedn(prob, N, it=20, guess=0):
    '''
    Uses newton raphson iteration, it times, to get a random walk n from a dist of size N
    '''
    if(guess == 0):
        guess = np.sqrt(N)
    for i in range(it):
        guess = guess - ((RandomWalkCDF(guess,N)-prob)/RandomWalkPDF(guess, N))
    i = np.random.randint(-1,1)
    if(i == 0): i = 1
    return guess*i

# get energy
def GetEnergyK(velocity, PE):
    '''
    returns a particles energy in kelvin
    '''
    return (0.5*MASS*(velocity**2) + PE)/K_B


# sim
def simulate():
    # Initialise variables of simulation
    VELOCITIES = RandVelocities(NUMBER,INITEMP,INITPE)
    FREEPATHS = np.zeros(NUMBER)
    TIME = 0
    escapeTimes = np.array([])

    # -- Main LOOP
    while(np.size(VELOCITIES)>0):
        # - move particles + redist vel
        TIME += TIMESTEP # iterate by one step
        directions, angles = UniRandVectorSpace(np.size(VELOCITIES)) # get current direction of all particles
        transComps = np.sqrt(directions[:,0]**2 + directions[:,1]**2) # get all transverse components
        FREEPATHS += VELOCITIES*transComps*TIMESTEP # get the amount each particle travels by

        # - check if a particle is crossing the beam
        # ie if a random num (0,1] is less than the value of the PDF at that MFP, then its crossing the beam
        crossingBeamIndex = np.where(UniRandSpace(np.size(FREEPATHS)) <= np.vectorize(MFPPDF)(FREEPATHS))[0]
        # - If crossing beam
        if(np.size(crossingBeamIndex)>0):
            # get impact params
            impactParams = TRAPSIGMA*UniRandSpace(np.size(crossingBeamIndex))
            # get beam path
            crossLengths = 2*np.sqrt(TRAPSIGMA**2-impactParams**2)
            # get kick numbers
            beamTimes = crossLengths/(VELOCITIES[crossingBeamIndex]*transComps[crossingBeamIndex])
            NKicks = beamTimes/(BEAMSWITCHPERIOD*(10**-9))
            # get a number of non-cancelled kicks, randomly invert it for negative energy 
            nKicks = np.vectorize(GetDistributedn)(np.random.uniform(0,0.5,size=np.size(crossingBeamIndex)), NKicks)
            # get a change in z-dir energy and therefor velocity for each particle
            deltaUZ = BEAMSTRENGTH * nKicks
            deltaVZ = deltaUZ/(MASS*VELOCITIES[crossingBeamIndex]*directions[crossingBeamIndex,2])
            # Apply the change in velocity to effected particles
            # this may be a little slow but works for no for recombining the ind of effect particles with the overall particle array
            for i in range(np.size(crossingBeamIndex)):
                ind = crossingBeamIndex[i]
                VELOCITIES[ind] = np.sqrt((VELOCITIES[ind]*directions[i,0])**2 + (VELOCITIES[ind]*directions[i,1])**2 + ((VELOCITIES[ind]*directions[i,2]) + deltaVZ[i])**2)
            # reset the MFP of effected particles to 0
            FREEPATHS[crossingBeamIndex] = 0

        #print(str(TIME) + ' ' + str(VELOCITIES) + ' ' + str(GetEnergyK(VELOCITIES[0],0)))
        if(np.isnan(VELOCITIES[0])):
            break;

        # - REMOVE ANY ESCAPED PARTICLES
        # Important to maintain that VELOCITIES and FREEPATHS are always the same lenght
        # get particles energies in kelvin
        energies = GetEnergyK(VELOCITIES, 0)
        escaped = np.where(energies>THRESHOLD)[0]
        if(np.size(escaped)>0):
            escapeTimes = np.append([TIME]*np.size(escaped), escapeTimes)
            # remove escaped particles 
            VELOCITIES = np.delete(VELOCITIES, escaped)
            FREEPATHS = np.delete(FREEPATHS, escaped)
            print("{0} PARTICLES ESCAPED, {1} REMAIN".format(np.size(escaped), np.size(VELOCITIES)))

    return escapeTimes

def simulate2():
    # Initialise variables of simulation
    VELOCITIES = RandVelocities(NUMBER,INITEMP,INITPE)
    TIMES = np.zeros(NUMBER)
    PASSES = np.zeros(NUMBER)
    KICKS = np.zeros(NUMBER)
    escapes = np.array([np.array([]), np.array([]), np.array([])])

    while(np.size(VELOCITIES)>0 and np.max(PASSES) < MAXPASSES):
        # distribute velocity components
        directions, angles = UniRandVectorSpace(np.size(VELOCITIES)) # get current direction of all particles
        transComps = np.sqrt(directions[:,0]**2 + directions[:,1]**2) # get all transverse components
        # Get time until next beam pass
        TIMES += np.vectorize(RandomMFPTime)(VELOCITIES*transComps)

        # For particles crossing beam (all of ones remaining lmao)
        # Impact params
        impactParams = TRAPSIGMA*UniRandSpace(np.size(VELOCITIES))
        # get beam path
        crossLengths = 2*np.sqrt(TRAPSIGMA**2-impactParams**2)
        # get crossing time and therefore total number of kicks
        beamTimes = crossLengths/(VELOCITIES*transComps)
        NKicks = beamTimes/(BEAMSWITCHPERIOD)
        # get a number of non-cancelled kicks, randomly invert it for negative energy (coin flip)
        nKicks = np.vectorize(GetDistributedn)(np.random.uniform(0,0.5,size=np.size(VELOCITIES)), NKicks)
        # get a change in z-dir energy and therefor velocity for each particle
        deltaUZs = BEAMSTRENGTH * nKicks
        deltaVZs = deltaUZs/(MASS*VELOCITIES*directions[:,2])
        # apply increase in velocity
        VELOCITIES = np.sqrt((VELOCITIES*directions[:,0])**2 + (VELOCITIES*directions[:,1])**2 + ((VELOCITIES*directions[:,2]) + deltaVZs)**2)
        
        # break and return an error if any velocities are nan
        if(np.any(np.isnan(VELOCITIES))):
            print("NAN ENCOUNTERED IN VELOCITIES, BREAKING")
            break;

        # ITERATE TRACKING ARRAYS
        PASSES += np.array([1]*np.size(PASSES))
        KICKS += nKicks

        # CHECK FOR ESCAPES
        energies = GetEnergyK(VELOCITIES, 0)
        escaped = np.where(energies>THRESHOLD)[0]
        if(np.size(escaped)>0):
            # create an array of all escape times, total passes and total number of non cancelled kicks for all escaped particles
            escapes = np.hstack((np.array([TIMES[escaped],PASSES[escaped],KICKS[escaped]]), escapes))
            # remove escaped particles 
            VELOCITIES = np.delete(VELOCITIES, escaped)
            PASSES = np.delete(PASSES, escaped)
            KICKS = np.delete(KICKS, escaped)
            TIMES = np.delete(TIMES, escaped)
            print("{0} PARTICLES ESCAPED, {1} REMAIN".format(np.size(escaped), np.size(VELOCITIES)))
    # After loop ends, its assumed all remaining particles will not escape in any capacity we care about
    stucks = np.array([TIMES, VELOCITIES, PASSES, KICKS])
    return escapes, stucks
        
def Histogram(array):
    Fig = plot.figure()
    plot.hist(array, bins=200, density=True, alpha=0.5, color='blue')
    plot.title("Escape Times, {0} Particles, $\delta t$ = {1}, {2}K -> {3}K".format(NUMBER, TIMESTEP, INITEMP, THRESHOLD))
    plot.xlabel('Escape Time /s')
    plot.ylabel('Density')
    plot.legend()
    plot.show()
    
def __main__():    
    times, bin = simulate2()
    Histogram(times[0])

    #A = np.array([np.array([1,1]), np.array([2,2]),])
    #B = np.array([np.array([3,3]), np.array([4,4]),])
    #print(np.hstack((A,B)))

__main__()
