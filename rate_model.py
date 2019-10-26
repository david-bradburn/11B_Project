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
P_on = 1*10**15 # note variation in pk-pk power in
P_off = 1* 10**12

t_stop = 0.00000001
delta_t_default = 1*10**(-10)

#print(delta_t_default)

def dn(n, p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain, n_0 = n_0_const):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p


def n_steady_state(p = P, I = I_initial, tau_s = tau_s_default , e = E, V = volume, g = gain, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t, p, I)*delta_t

        #t += delta_t

        if abs((n_t1 - n_t)/ n_t1) < 0.00001:
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

#G_P_I()


def generate_random_sequence(length, p_on = P_on, p_off = P_off):
    assert type(length) == int
    return np.random.choice([p_off, p_on], length)


def P_out_for_sequence():
    sequence = generate_random_sequence(100, P_on)
    f = 100 * 10 ** 6     # B/s

    time = [0]

    t_step = min(1/(1000*f), 1*10**(-10))
    bit = 0

    n_t0 = n_steady_state(sequence[0])
    #print(n_t0)
    n_o = [n_t0]
    gain_ar = [gain*(n_t0 - n_0_const)]
    P_o_ar = [gain_ar[-1] * sequence[0]]

    time_0 = time[0]
    while time[-1] < (1/f * len(sequence)):

        n_t1 = n_t0 + dn(n_t0, sequence[bit]) * t_step

        #print(time[-1], n_t0)

        gain_ar += [gain*(n_t1 - n_0_const)]
        n_o += [n_t1]
        n_t0 = n_t1

        P_o_ar += [gain_ar[-1]*sequence[bit]]

        time += [time[-1] + t_step]

        if time[-1] - time_0 >= 1/f:
            bit += 1
            time_0 = time[-1]
            if bit > len(sequence):
                raise ValueError


    #print(gain_ar)
    #print(time)
    #print(n_o)
    plt.plot(time, n_o)
    plt.title('Carrier Density Against Time')
    plt.show()
    plt.plot(time, gain_ar)
    plt.title('Gain against time')
    plt.show()
    plt.plot(time, P_o_ar)
    plt.title('Power Out against time')
    plt.show()
P_out_for_sequence()
