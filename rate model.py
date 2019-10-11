import numpy as np

n = 1 #carrier_concentration

tau_s = 1 #spontaneous_emmision_life_time

dn = 0 #rate_of_change_of_carrier_concentration

I = 1 #current Amps
e = 1.9*10**(-19) #e constant
V = 1 # Volume
g = 1 #gain contant
n_0 = 1  #intial carrier conc
P = 1 #photon conc


def dn(t):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*P


t_stop = 50
delta_t = 0.01
t = 0
while t < t_stop:

    n_t1 = n_t + dn(t)*delta_t