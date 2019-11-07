import numpy as np
import matplotlib.pyplot as plt

#n = 1 #carrier_concentration

tau_s_default = 3*10**(-9)# 3*10**(-9) #ns #spontaneous_emmision_life_time

dn = 0 #rate_of_change_of_carrier_concentration

I_initial = 0.005# 50*10**(-3) #current Amps
E = 1.9*10**(-19) #e constant

Length = 10 * 10**(-6) #m
Width = 3 * 10**(-6) #m
Depth = 0.2 * 10**(-6) #m

volume = Length * Width * Depth # Volume
#print(volume)

gain_coefficient = 3 * 10 ** (-20) #m^3/s #gain contant
n_0_const = 1.1 * 10**(24)  #m^-3#transparent carrier density
P = 1#1*10**10 #photon conc
P_on = 1*10**27 # note variation in pk-pk power in
P_off = 1* 10**26




t_stop = 0.0001
delta_t_default = 1*10**(-11)

#print(delta_t_default)

def dn(n, p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p


def n_steady_state(p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t, p, I)*delta_t

        #t += delta_t

        if abs((n_t1 - n_t)/ n_t1) < 0.0000001:
            return n_t1

        n_t = n_t1


I = np.linspace(0.001, 0.01, 5)


def G_P_I():
    for i in I:
        p_ar = np.logspace(24, 29.5, 100)
        g_out = []
        for p_0 in p_ar:
            g_out += [np.exp(gain_coefficient * (n_steady_state(p_0, i) - n_0_const) * Length)]

        plt.plot(np.log10(p_ar), 10*np.log10(g_out), Label = i)
        plt.xlabel("Log(P)")
        plt.ylabel("Gain /dB")
        plt.title("Gain Vs P")

    plt.legend()
    plt.show()

#G_P_I()


def generate_random_sequence(length, p_on = P_on, p_off = P_off):
    assert type(length) == int
    return np.random.choice([p_off, p_on], length) #need to add rise and fall time


def generate_random_full_sequence_without_risetime(length, f, step_time, p_on = P_on, p_off = P_off):
    assert type(length) == int

    total_time = (length * 1/f)
    sequence = generate_random_sequence(length, p_on, p_off)

    bit = 0
    t = 0
    t_0 = 0
    op = []

    while t < total_time:
        op += [sequence[bit]]
        t += step_time

        if t - t_0 >= 1/f:

            bit += 1
            t_0 = t
            if bit > len(sequence):
                raise ValueError

    # plt.plot(op)
    # plt.show()
    return op
#
# op2 = generate_random_full_sequence_without_risetime(10, 0.1, 0.01)
#


def generate_random_full_sequence_with_risetime(length, f, step_time, rise_time, fall_time, p_on = P_on, p_off = P_off):
    assert type(length) == int

    total_time = (length * 1/f)
    sequence = generate_random_sequence(length, p_on, p_off)

    bit = 0
    t = 0
    t_0 = 0
    op = []

    max_to_min = np.linspace(p_on, p_off, int(fall_time/step_time))
    min_to_max = np.linspace(p_off, p_on, int(rise_time/step_time))

    while t < total_time:
        op += [sequence[bit]]
        t += step_time

        if t - t_0 >= 1/f:

            bit += 1
            t_0 = t
            if bit > len(sequence):
                raise ValueError

            if sequence[bit - 1] == sequence[bit]:
                pass
            elif sequence[bit - 1] < sequence[bit]:
                #print(op, min_to_max)
                for i in min_to_max:
                    op += [i]
                t += len(min_to_max) * step_time
            elif sequence[bit - 1] > sequence[bit]:
                for i in max_to_min:
                    op += [i]
                t += len(max_to_min) * step_time

    return op
    #print(op)
    #plt.plot(op)
    #plt.show()

#
# op1 = generate_random_full_sequence_with_risetime(10, 0.1, 0.01, 1, 1)
#
# plt.plot(op1)
# plt.plot(op2)
# plt.show()


def P_out_for_sequence():
    sequence = generate_random_sequence(100, P_on, P_off)
    f = 10 * 10 ** 6     # B/s min is roughly 1 MHz before sim breaks down
    rise_time = 1/(20*f)

    time = [0]

    t_step = min(1/(1000*f), 1*10**(-10))
    bit = 0

    n_t0 = n_steady_state(sequence[0])
    #print(n_t0)
    n_o = [n_t0]
    gain_ar = [gain_coefficient * (n_t0 - n_0_const)]
    P_o_ar = [gain_ar[-1] * sequence[0]]

    time_0 = time[0]
    while time[-1] < (1/f * len(sequence)):

        n_t1 = n_t0 + dn(n_t0, sequence[bit]) * t_step

        #print(time[-1], n_t0)

        gain_ar += [np.exp(gain_coefficient * (n_t1 - n_0_const)*Length)]

        # if time[-1] == 0:
        #     print(gain_coefficient * (n_t1 - n_0_const)*Length)
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
    plt.plot(time[1:], n_o[1:])
    plt.title('Carrier Density Against Time')
    plt.show()
    plt.plot(time[1:], gain_ar[1:])
    plt.title('Gain against time')
    plt.show()
    plt.plot(time[1:], P_o_ar[1:])
    plt.title('Power Out against time')
    plt.show()

#P_out_for_sequence()

#Gonna have to do a redesign of the system so I can feed the out of one into the other
