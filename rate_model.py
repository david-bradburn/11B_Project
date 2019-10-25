import numpy as np
import matplotlib.pyplot as plt

#n = 1 #carrier_concentration

tau_s_default = 3*10**(-9)# 3*10**(-9) #ns #spontaneous_emmision_life_time

dn = 0 #rate_of_change_of_carrier_concentration

I_initial = 1# 50*10**(-3) #current Amps
E = 1.9*10**(-19) #e constant

Length = 280 * 10**(-4) #cm
Width = 3 * 10**(-4) #cm
Depth = 0.2 * 10**(-4) #cm

volume = Length * Width * Depth # Volume

gain = 3 * 10 **(-7) #cm^3/s #gain contant
n_0_const = 1.1 * 10**(18)  #cm^-3#transparent carrier density
P = 1#1*10**10 #photon conc

t_stop = 0.00000001
delta_t_default = t_stop/1000



def dn(n, p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain, n_0 = n_0_const):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p


def n_steady_state(p = P, I = I_initial, tau_s = tau_s_default , e = E, V = volume, g = gain, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t, p, I)*delta_t

        t += delta_t

        if abs((n_t1 - n_t)/ n_t1) < 0.0001:
            return n_t1

        n_t = n_t1


I = np.linspace(1, 10, 5)


def G_P_I():
    for i in I:
        p_ar = np.logspace(9, 17.8, 100)
        g_out = []
        for p_0 in p_ar:
            g_out += [gain*(n_steady_state(p_0, i) - n_0_const)]

        plt.semilogy(np.log10(p_ar), g_out, Label = i)
        plt.xlabel("Log(P)")
        plt.ylabel("n")
        plt.title("Carrier Conc Vs P")

    plt.legend()
    plt.show()

G_P_I()