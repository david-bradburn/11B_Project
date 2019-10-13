import numpy as np
import matplotlib.pyplot as plt

#n = 1 #carrier_concentration

tau_s_default = 1 #spontaneous_emmision_life_time

dn = 0 #rate_of_change_of_carrier_concentration

I_initial = 1 #current Amps
E = 1.9*10**(-19) #e constant
volume = 1 # Volume
gain = 1 #gain contant
n_0_const = 1  #transparent carrier density
P = 1 #photon conc

t_stop = 20
delta_t_default = 0.001




def dn(n, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain, n_0 = n_0_const, p = P):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p


def n_steady_state(p = P, tau_s = tau_s_default, I = I_initial, e = E, V = volume, g = gain, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t)*delta_t

        t += delta_t

        if abs((n_t1 - n_t)/ n_t1) < 0.001:
            return n_t1

        n_t = n_t1

p_ar = np.logspace(0, 100, 10)
for p in p_ar:
    print(n_steady_state(p))
