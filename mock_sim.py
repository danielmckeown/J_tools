%matplotlib inline
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import matplotlib as mpl

#define 2 functions, one to simply create an NFW density profile, the other to create a particle distribution based
# on this NFW density profile, called M_within_r
def rho(r, R_s = 12.52, rho_0 = 3.0e7 ):
    rho = rho_0 / (   r/R_s * ( 1 + r/R_s ) **2  )
    return rho

# The probability distribution is the mass within R 
# This is a derived function from a NFW so it gives the particles in each bin
# thus it gives the number of particles per bin
#calculate the Mass profile


def M_within_r(r, rho_0 = 3.0e7, R_s = 12.52):
    M = 4.0 * np.pi * rho_0 * R_s **3.0 * ( np.log ( (R_s + r)/ R_s ) - r/(R_s + r) )
    return M

#=====================  0. some parameters  =====================
# Important
save_data = False #do you want to save the data, SET IT TO TRUE IF YOU ALREADY GENERATE ONE SET
read_data = False #do you want to read the existing data, SET IT TO TRUE THE 1ST TIME YOU GENERATE THE DATA
                # SO IT SAVES THE DATA YOU GENERATE

#path you wanna save/read the data
path = "/Users/danmckeown/" # path
file_name = "particle_daniel.npy" #file name
plot_path = '/Users/danmckeown/' # where you want to save your data
plot_name = 'test.png'

#=====================  1.Get the reverse function of M_normed  =====================
r_min = 0.01
r_max = 300
n_parts = 500

rad = np.linspace(r_min,r_max,n_parts) #generate radius

rho_test = rho(rad) # the density from the density profile given (creates the orange line)

M_max = M_within_r(r_max) # the mass at r_max : used to normalized M_within_r

print ('printing M max')
print (np.log10(M_max))

M = M_within_r(rad)/M_max # normalize (from 0-1) M_within_r so that at r_max it equals 1 (this is the CDF)

print(M)
# So now here I am saying, what function would fit the inverse of my y coord (M within R ) and my x coord r itself
interp = interp1d(M,rad,kind ='nearest') #inverse function of M_normed (this is the QDF)

# So interp is now my inverse function which I will use later

#=====================  2.Generate particles  =====================
#If you already generate the set of particles, then just read the data you generate
#instead of generating another set of particles, which will increase the time this program takes
if read_data:
    data = np.load(path+"/"+file_name)
else:
    #number of particles you want
    num_of_particles = 10**7
    mass_particle = M_max/num_of_particles #mass of a single particle
    
    print ('printing mass per particle')
    print ( mass_particle)
    #Generate uniform random number between 0.0 - 1.0 for Mass
    random_mass_0to1 = np.random.uniform(low = 0.0,high = 1.0,size=num_of_particles)

    random_r = interp(random_mass_0to1) # generate the radius

    data = np.zeros((num_of_particles,4))

    for i in range(0, num_of_particles):
        phi = 2 * np.pi * np.random.uniform(low = 0.0, high = 1.0)
        theta = np.arccos(1 - 2 * np.random.uniform(low = 0.0, high = 1.0))
        data[i,0] = random_r[i]*np.sin(theta) * np.cos(phi)
        data[i,1] = random_r[i]*np.sin(theta) * np.sin(phi)
        data[i,2] = random_r[i]*np.cos(theta)
        data[i,3] = random_r[i]

    #if you wanna save the particle data you generate above
    if save_data:
        np.save(path+"/"+file_name,data)


#=====================  3.Plot the data  =====================
fig, ax = plt.subplots(1, 1)

lower = 0.0
upper = 300.0
bins_num = 500
step = (upper - lower)/bins_num
weight_particles = mass_particle*1.0/(4.0*np.pi*np.square(data[:,3]))/step
# /the weight particles is just an estimate
# read doc for plot.hist or np.histogram


# if you use numpy.histogram to do the histogram, use the following part
'''
hist = np.histogram(data[:,3], bins=bins_num, weights = weight_stars, normed = False)
den = hist[0]
d = ( (hist[1])[0:bins_num] + (hist[1])[1:bins_num+1] ) * 0.5
plt.scatter(d,den,color = 'blue',s = 0.5)
'''

plt.plot(rad,rho_test,color = 'orange', linestyle='--', linewidth=3)

#if you use plt.hist to do the histogram, use the following part
plt.hist(data[:,3], bins=bins_num, weights = weight_particles,histtype='step', normed = False)



plt.yscale('log')
plt.xscale('log')
plt.xlim(0.01,300)
plt.ylim(5.0e-4,1.0e10)
fig.savefig(plot_path+plot_name,dpi=400)
plt.show()
plt.hist2d(data[:,1], data[:,2], norm=mpl.colors.LogNorm(), bins= 64)
plt.show() 


