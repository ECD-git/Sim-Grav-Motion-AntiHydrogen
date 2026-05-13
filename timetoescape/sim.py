import numpy as np
import matplotlib.pyplot as plot
import math
import csv
from ast import literal_eval
import scipy.optimize as SciPio

# physical
global MASS; MASS = 1.00784*1.66*(10**-27);
global K_B; K_B = 1.380649*(10**-23);
global C; C = 3*(10**8);
global ALPHA_FACTOR; ALPHA_FACTOR=6.67*(10**-31) #[m^3], alpha/4*pi*epsilon_0
# simulation
global NUMBER; NUMBER = 20000;
global INITTEMP; INITEMP = 0.001; # kelvin
global INITPE; INITPE = 0;
global THRESHOLD; THRESHOLD = 0.01; # kelvin
global MAXPASSES; MAXPASSES = 10000;
global MAXTIME; MAXTIME = 10000; # 10000 sim seconds
global MINESCAPES; MINESCAPES = 2000; # min number of escapes to occur before sim starts timing out particles
global SNAPSHOTTIMES; SNAPSHOTTIMES = np.array([1000,2000,3000,4000,5000]) # take dist at some interesting times, im not sure how this behaves at >MAXTIME since its uneeded

global ESCAPEVEL; ESCAPEVEL = np.sqrt(2*(THRESHOLD*K_B)/MASS)
print("ESCAPE VELOCITY: {0} ms^-1".format(ESCAPEVEL))

# trap
global TRAPSIGMA; TRAPSIGMA = 200*(10**-6) #[m]
global TRAPD; TRAPD = 5*0.01 #[m]
global BEAMSWITCHPERIOD; BEAMSWITCHPERIOD = 1*(10**-9) #[s]
global BEAMLAMBDA; BEAMLAMBDA = 1200*(10**-9) # [m]
global BEAMPOWER; BEAMPOWER = 2 #[W]

# we are using the expectation value of the energy change per kick since there will be like 10**6 kicks per beam pass
global BEAMSTRENGTH; BEAMSTRENGTH = ALPHA_FACTOR * ((4*BEAMPOWER)/(C*(TRAPSIGMA**2)))

defaultfontsize = 13
plot.rc('axes', titlesize=defaultfontsize) #fontsize of the title
plot.rc('axes', labelsize=defaultfontsize) #fontsize of the x and y labels
plot.rc('xtick', labelsize=defaultfontsize) #fontsize of the x tick labels
plot.rc('ytick', labelsize=defaultfontsize) #fontsize of the y tick labels
plot.rc('legend', fontsize=defaultfontsize) #fontsize of the legend

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

def GetDistributedVelocity(prob, temp, PE, it=100, guess=0):
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
    plot.hist(velocities, bins=100, density=True, alpha=0.5, color='blue')
    x = np.linspace(0, np.max(velocities), 100)
    y = MaxBoltPDF(x, 0.001, 0)
    plot.plot(x,y,"k--",label = "Maxwell Boltzmann PDF")
    plot.title("Randomly Distributed Velocities, $T=${0}$k$, U={1}$j$, N={2}".format(INITEMP, INITPE, np.size(velocities)))
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

def simulate2(snapshots = False):
    # Initialise variables of simulation
    VELOCITIES = RandVelocities(NUMBER,INITEMP,INITPE)
    initVel = VELOCITIES
    TIMES = np.zeros(NUMBER)
    PASSES = np.zeros(NUMBER)
    KICKS = np.zeros(NUMBER)
    escapes = np.array([np.array([]), np.array([]), np.array([])])
    stucks = np.array([np.array([]), np.array([]), np.array([]), np.array([])])
    snapshotdata = [[[0,0,0]] for _ in range(np.size(SNAPSHOTTIMES))]

    while(np.size(VELOCITIES)>0):
        # distribute velocity components
        directions, angles = UniRandVectorSpace(np.size(VELOCITIES)) # get current direction of all particles
        transComps = np.sqrt(directions[:,0]**2 + directions[:,1]**2) # get all transverse components
        # Get time until next beam pass
        deltaT = np.vectorize(RandomMFPTime)(VELOCITIES*transComps)

        # iterate time
        TIMES += deltaT

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
        
        # need to scale the kick size to the velocity of the particle
        #EnergyScaling = 1-(np.cos((np.pi*VELOCITIES*directions[:,2]*BEAMSWITCHPERIOD)/(BEAMLAMBDA)))**2
        #EnergyScaling = np.full(len(EnergyScaling), 0.5)
        #print(( np.mean(EnergyScaling)*BEAMSTRENGTH)/K_B) # prints the average beam pass energy in kelvin
        
        # get a change in z-dir energy and therefor velocity for each particle
        deltaUZs = BEAMSTRENGTH * nKicks# * EnergyScaling
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

        # CHECK FOR ESCAPES OR TIME OUT
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
            deltaT = np.delete(deltaT, escaped)
            print("{0} PARTICLES ESCAPED, {1} REMAIN".format(np.size(escaped), np.size(VELOCITIES)))

        if(snapshots):
            # check if time->time+deltatime lies over one of our snapshot times (inclusive)
            # this is a little computationally heavy but i cant think of a more efficient method
            # disable if unneeded
            # broadcast the snapshot times vertically to a matrix of length equal to TIMES
            checkTimesMatrix = np.tile(np.reshape(SNAPSHOTTIMES, (np.size(SNAPSHOTTIMES,axis=0),1)),(1, np.size(TIMES,axis=0)))
            # broadcast TIMES, TIMES+deltaT horizontally to the length of snapshottimes,
            # check if a snapshot time lies between the 2 values, and get their ind (atp TIMES has been iterated so must look to previous value)
            indSnapshot = (np.tile(TIMES-deltaT,(np.size(checkTimesMatrix, axis=0),1))<checkTimesMatrix) * (checkTimesMatrix<=np.tile(TIMES,(np.size(checkTimesMatrix, axis=0),1)))
            indSnapshot = np.where(indSnapshot)

            # get a snapshot of data if at a relevant time
            # want data of all escaped particles as above and data of all remaining particles
            # NB: you can get the escapes before each snapshot time by truncating the 'escapes' array to that time
            if (np.size(indSnapshot[0]) > 0):
                for i in range(len(indSnapshot[0])):
                    snapshotdata[indSnapshot[0][i]].append([float(VELOCITIES[indSnapshot[1][i]]), float(PASSES[indSnapshot[1][i]]), float(KICKS[indSnapshot[1][i]])])
                #print("{0} PARTICLES PASSED SOME TIME THRESHOLD".format(np.size(indSnapshot[0])))

        # check if a minimum number of particles have escaped first
        if(np.size(escapes[0]) > MINESCAPES):
            stuck = np.where(TIMES>MAXTIME)[0]
            if(np.size(stuck) > 0):
                # get vel and other profiles of stuck particles
                stucks = np.hstack((np.array([TIMES[stuck],VELOCITIES[stuck],PASSES[stuck],KICKS[stuck]]), stucks))
                VELOCITIES = np.delete(VELOCITIES, stuck)
                PASSES = np.delete(PASSES, stuck)
                KICKS = np.delete(KICKS, stuck)
                TIMES = np.delete(TIMES, stuck)
                print("{0} PARTICLES TIMEDOUT, {1} REMAIN".format(np.size(stuck), np.size(VELOCITIES)))
            # break if we have exceeded the max number of passes for any particle
            if (np.size(PASSES) != 0):
                if(np.max(PASSES) >= MAXPASSES):
                    break;

    # After loop ends, its assumed all remaining particles will not escape in any capacity we care about
    print("MAX LOOPS EXCEEDED, {0} REMAINING PARTICLES TIMED OUT".format(np.size(VELOCITIES)))
    stucks = np.hstack((np.array([TIMES,VELOCITIES,PASSES,KICKS]), stucks))
    print("{0} PARTICLES ESCAPED, {1} STILL TRAPPED".format(np.size(escapes[0]),np.size(stucks[0])))

    #remove placeholder element from the snapshot data list
    if (snapshots):
        for i in range(len(snapshotdata)):
            snapshotdata[i].pop(0)

        return initVel, escapes, stucks, snapshotdata
    else:
        return initVel, escapes, stucks


def ReadFile(filename):
    dat = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        raw = csv.reader(f)
        # eg dat[0] = escape times
        # dat[1] = num beam passes of escaped particles
        # dat[2] = total num of non cancelled kicks for escaping particles

        # [0] = Times
        # [1] = Vel of stuck particles
        # [2] = passes
        # [3] = kicks

        # row num = snapshot time index
        # column = particle
        # for each element, [velocity, num passes, num kicks]

        # for inital vel, each index is just a velocity
        for row in raw:
            dat.append(row)
    # csv reads as strings so need to convert back to floats
    try:
        for i in range(len(dat)):
            dat[i] = list(map(float, dat[i]))
        return dat
    except:
        return dat
    
def EscapeTimesModel(x,a,b):
    return b*np.exp(x/a)

def HistEscapeTimes(filename):
    dat = ReadFile(filename)

    # get mean and std of plots

    Fig = plot.figure()
    bins = plot.hist(dat[0], bins=200, density=True, alpha=0.5, color='blue', label="data")

    binx = np.array([])
    for i in range(np.size(bins[1])):
       if(i+1 != np.size(bins[1])):
        binx = np.append((bins[1][i] + bins[1][i+1])/2, binx)
    biny = np.flip(bins[0])

    coefs = SciPio.curve_fit(EscapeTimesModel, binx, biny, p0=[-5000,0.0002])
    print(coefs[0])
    y = EscapeTimesModel(binx, coefs[0][0], coefs[0][1])
    perr = np.sqrt(np.diag(coefs[1]))

    '''
    RedChiSq = 0
    for i in range(len(binx)):
        RedChiSq += (bins[0][i]- EscapeTimesModel())**2/EscapeTimesModel()
    '''

    plot.title(r"Escape Times, {0} Particles, {3} Escaped {1}K -> {2}K".format(NUMBER, INITEMP, THRESHOLD, len(dat[0])))
    plot.text(6500,0.00013,"Model $PDF(t) = b\exp[t/a + c]$\na = {0:3.2e}$\pm${2:3.2e} s \nb={1:3.2e}$\pm${3:3.2e}".format(coefs[0][0], coefs[0][1], perr[0],perr[1]), fontsize=15)
    plot.plot(binx,y,'k--', label="model")
    plot.xlabel('Escape Time /s')
    plot.ylabel('Density')
    plot.legend()
    plot.show()

def HistVelDistStucks(filename):
    dat = ReadFile(filename)

    # get mean and std of plots

    Fig = plot.figure()
    plot.hist(dat[1], bins=200, density=True, alpha=0.5, color='blue')
    plot.title(r"Velocities, Stuck Particles, {0} Particles {3} Stuck, {1}K -> {2}K".format(NUMBER, INITEMP, THRESHOLD, len(dat[1])))
    plot.xlabel('Vel /ms^-1')
    plot.ylabel('Density')
    plot.show()

def HistSnapShotVel(filename, initials, timeIndex=0):
    dat = ReadFile(filename)
    time = SNAPSHOTTIMES[timeIndex]
    # atp each row is a list of particle dat as a string, need to convert each string of list to an actual list
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            dat[i][j] = literal_eval(dat[i][j])
    velocities = [x[0] for x in dat[timeIndex]]
    nkicks = [x[2] for x in dat[timeIndex]]
    initVels = ReadFile(initials)

    Fig = plot.figure()

    x = np.linspace(0, np.max(initVels), 100)
    y = MaxBoltPDF(x, 0.001, 0)
    #plot.plot(x,y,"r--",label = "Maxwell Boltzmann PDF")

    plot.hist(initVels, bins=100, density=False, alpha=0.15, color = 'red', label="Initial Velocities")
    plot.hist(velocities, bins=100, density=False, alpha=0.65, color='blue', label = "Current Velocities")
    plot.plot([ESCAPEVEL, ESCAPEVEL], [0, 0.2], 'k--', label="Threshold Velocity")

    plot.title(r"Total num kicks, {0} Initial Particles, {4} remain at t={1}s, {2}K -> {3}K".format(NUMBER, time, INITEMP, THRESHOLD, len(velocities)))
    plot.xlabel('vel /ms^-1')
    plot.ylabel('Counts')
    plot.legend()
    plot.show()

def HistAllSnapShotVel(filename, initials):
    dat = ReadFile(filename)
    # atp each row is a list of particle dat as a string, need to convert each string of list to an actual list
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            dat[i][j] = literal_eval(dat[i][j])

    Fig = plot.figure()
    
    initVels = ReadFile(initials)
    plot.hist(initVels, bins=100, density=False, alpha=0.1, color = 'red', label="Initial Velocities")
    plot.text(np.mean(initVels), 550, "t = {0} s\nMean={1:3.2f} ms-1\nSTD={2:3.2f} ms-1".format(0, np.mean(initVels), np.std(initVels)))
    print(str(np.mean(initVels)) + ',' + '0')

    textYPos = [450,350,260,200,170]
    textXPos = [5.6, 6.8, 7.6, 9.2, 11]
    for i in range(np.size(SNAPSHOTTIMES)):
        velocities = [x[0] for x in dat[i]]
        print(str(np.mean(velocities)) + ',' + str(SNAPSHOTTIMES[i]))
        plot.hist(velocities, bins=100, density=False, alpha=0.1*(i+1), color='blue', label = "{0}s".format(SNAPSHOTTIMES[i]))
        plot.text(textXPos[i], textYPos[i], "t = {0} s\nMean={1:3.2f} ms-1\nSTD={2:3.2f} ms-1".format(SNAPSHOTTIMES[i], np.mean(velocities), np.std(velocities)))
    
    plot.plot([ESCAPEVEL, ESCAPEVEL], [0, 600], 'k--', label="Threshold Velocity")

    plot.title(r"Normalised Velocity Distributions, {0} Initial Particles, {1}K -> {2}K".format(NUMBER, INITEMP, THRESHOLD))
    plot.xlabel('vel /ms^-1')
    plot.ylabel('Counts')
    plot.legend()
    plot.show()

def VelOverTimeModel(x,a,b,d):
    return (b*np.exp((x/a))) + d

def VelOverTime():
    data = np.array([[4.579341917506838,0],
                    [6.294667660745464,1000],
                    [7.383582966421579,2000],
                    [8.21977892793488,3000],
                    [8.870650580819806,4000],
                    [9.351998770023739,5000]])
    
    coefs = SciPio.curve_fit(VelOverTimeModel, data[:,1], data[:,0], p0=[-500,-1,9])
    x = np.linspace(0,6000,50)
    print(coefs)
    y = VelOverTimeModel(x, coefs[0][0], coefs[0][1], coefs[0][2],)

    perr = np.sqrt(np.diag(coefs[1]))

    figure = plot.figure()
    plot.scatter(data[:,1], data[:,0], marker='x', label="data",zorder=1)
    plot.plot(x,y,'k--',zorder=2,label="model")
    plot.text(2000,5,"Model $v = b\exp[t/a]$ + d\na = {0:3.2f} $\pm$ {3:3.2f}s\nb={1:3.2f} $\pm$ {4:3.2f} ms-1\nd={2:3.2f}$\pm${5:3.2f} ms-1".format(coefs[0][0], coefs[0][1], coefs[0][2], perr[0],perr[1],perr[2]), fontsize=15)
    plot.xlabel(r"time \s", fontsize=13)
    plot.ylabel(r"mean velocity \ms$^{-1}$", fontsize=13)
    plot.title("Mean Velocity Over Time")
    plot.legend()
    plot.show()
    
def HistSnapShotKicks(filename, timeIndex=0):
    dat = ReadFile(filename)
    time = SNAPSHOTTIMES[timeIndex]
    # atp each row is a list of particle dat as a string, need to convert each string of list to an actual list
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            dat[i][j] = literal_eval(dat[i][j])
    velocities = [x[0] for x in dat[timeIndex]]
    nkicks = [x[2] for x in dat[timeIndex]]

    Fig = plot.figure()

    print(np.mean(nkicks))
    print(np.std(nkicks))

    plot.hist(nkicks, bins=100, density=True, alpha=1, color='blue', label = "Current Velocities")

    plot.title(r"Total num kicks, {0} Initial Particles, {4} remain at t={1}s, {2}K -> {3}K".format(NUMBER, time, INITEMP, THRESHOLD, len(velocities)))
    plot.xlabel('n kicks')
    plot.ylabel('Density')
    plot.legend()
    plot.show()

def DrawTestVels(num, temp, PE):
    velocities = RandVelocities(num,temp,PE)
    x = np.linspace(0, np.max(velocities), 100)
    y = MaxBoltPDF(x, temp, PE)

    Fig = plot.figure()
    bins = plot.hist(velocities, bins=100, density=True, alpha=0.5, color='blue', label="Random Variables")

    ModelMean = 2 * np.sqrt(K_B*temp/MASS) * np.sqrt(2/np.pi)
    mean = np.mean(velocities)
    std = np.std(velocities)

    # get middle value of all bins
    binx = np.array([])
    for i in range(np.size(bins[1])):
       if(i+1 != np.size(bins[1])):
        binx = np.append((bins[1][i] + bins[1][i+1])/2, binx)

    RedChiSq = 0
    for i in range(len(binx)):
        RedChiSq += (bins[0][i]-MaxBoltPDF(binx[i], temp, PE))**2/MaxBoltPDF(binx[i], temp, PE)

    plot.text(200,0.005,r"Model Mean = {0:3.2f},\nMean = {1:3.2f} $\pm$ {2:3.2f}".format(ModelMean, mean, std))
    plot.plot(x,y,"k--",label = "Maxwell Boltzmann PDF")
    plot.title("Randomly Distributed Velocities, $T=${0}k, U={1}j, N={2}".format(temp, PE, np.size(velocities)))
    plot.xlabel('Velocity / ms$^{-1}$')
    plot.ylabel('Density')
    plot.legend()
    plot.show()
  
def __main__():
    initvel, escapes, stucks, snapshotdata = simulate2(snapshots=True)
    
    # this completely overwrites previous file data, so for consecutive runs the data should be combined into a 'master array' before being written
    with open("lastrunTimes.csv", "w", newline="", encoding="utf-8") as fe:
        writer = csv.writer(fe)
        writer.writerows(escapes)

    with open("lastrunStucks.csv", "w", newline="", encoding="utf-8") as fs:
        writer = csv.writer(fs)
        writer.writerows(stucks)

    with open("lastrunSnapshot.csv", "w", newline="", encoding="utf-8") as fss:
        writer = csv.writer(fss)
        writer.writerows(snapshotdata)

    with open("initvelocities.csv", "w", newline="", encoding="utf-8") as fi:
        writer = csv.writer(fi)
        writer.writerow(initvel)

    print("CSV file written successfully.")
    

#__main__()
#HistVelDistStucks("lastrunStucks.csv")
HistEscapeTimes("lastrunTimes.csv")
#HistSnapShotVel("lastrunSnapshot.csv", "initvelocities.csv", timeIndex=2)
#HistAllSnapShotVel("lastrunSnapshot.csv", "initvelocities.csv")
#VelOverTime()
#HistSnapShotKicks("lastrunSnapshot.csv", timeIndex=3)

#DrawTestVels(10000,0.5,0)