import numpy as np
import matplotlib.pyplot as plt

#n = 1 #carrier_concentration

tau_s = 1 #spontaneous_emmision_life_time

dn = 0 #rate_of_change_of_carrier_concentration

I = 1 #current Amps
e = 1.9*10**(-19) #e constant
V = 1 # Volume
g = 1 #gain contant
n_0 = 1  #transparent carrier density
P = 1 #photon conc

n_initial_array = np.logspace(15, 22, 10)


def dn(n):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*P


for n_initial in n_initial_array:
    t_stop = 7
    delta_t = 0.001
    t = 0

    n_out = []
    n_t = n_initial

    n_out += [n_t]

    t_out = []
    t_out += [t]

    while t < t_stop:

        n_t1 = n_t + dn(n_t)*delta_t
        print(t, n_t)

        t += delta_t

        n_out += [n_t1]
        t_out += [t]

        n_t = n_t1

    plt.semilogy(t_out, n_out)

plt.show()