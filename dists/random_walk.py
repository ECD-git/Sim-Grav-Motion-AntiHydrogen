import numpy as np
import matplotlib.pyplot as plot

def PDF(n,N):
    return 1/(np.sqrt(2*np.pi*N)) * np.exp(-(n**2)/(2*N))

Nspecies = 10000
N=10000

species = np.zeros(Nspecies)

for i in range(N):
    delta = np.random.randint(-1,1,size=Nspecies)
    delta[np.where(delta==0)] += 1
    species += delta

Fig = plot.figure()
plot.hist(species, bins=200, density=True, alpha=0.5, color='blue')

mean = np.mean(species)
std = np.std(species)

plot.text(140,0.003,"Model Mean $\pm$ STD = {0} $\pm$ {1} ,\nMean $\pm$ STD = {2:3.2f} $\pm$ {3:3.2f}".format(0, np.sqrt(N), mean, std))

n = np.linspace(np.min(species), np.max(species), 100)
y = PDF(n,N)
plot.plot(n,y,"k--",label = "Random Walk PDF")

plot.title("Histogram of Simulated Random Walk, {0} tests, N = {1}".format(Nspecies,N))
plot.xlabel('n')
plot.ylabel('Density')
plot.legend()
plot.show()