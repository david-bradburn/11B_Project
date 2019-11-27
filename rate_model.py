import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

                                    #n = 1 #carrier_concentration

tau_s_default = 3*10**(-9)          # 3*10**(-9) #ns #spontaneous_emmision_life_time

dn = 0                              #rate_of_change_of_carrier_concentration

I_initial = 0.005                   # 50*10**(-3) #current Amps
E = 1.9*10**(-19)                   #e constant

Length = 10 * 10**(-6)              #m
Width = 3 * 10**(-6)                #m
Depth = 0.2 * 10**(-6)              #m

volume = Length * Width * Depth     # Volume
                                    #print(volume)


gain_coefficient = 3 * 10 ** (-20)  #m^3/s #gain contant
n_0_const = 1.1 * 10**(24)          #m^-3#transparent carrier density
P = 1                               #1*10**10 #photon conc
P_on = 0.5*10**28                   # note variation in pk-pk power in
P_off = 0.5* 10**27


                                    #bottom 0.5*10**26 - 0.5*10**27
                                    #middle 0.5*10**27 - 0.5*10**28
                                    #top 0.5*10**28 - 0.5*10**29

t_stop = 0.0001
delta_t_default = 1*10**(-11)

########################################################################################################################
#    Assuming that the laser is InP then Eg = 1.35 eV -> wavelength = 915nm

h = sc.h
c = sc.c
wavelength = 915*10**(-9)

ppp = h*c/wavelength                #power per photon




#print(delta_t_default)


def dn(n, p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const):
    return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p


def n_steady_state(p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t, p, I)*delta_t

        if abs((n_t1 - n_t)/ n_t1) < 0.0000001:
            return n_t1

        n_t = n_t1


I = np.linspace(0.001, 0.01, 5)


def G_P_I():
    for i in I:
        p_ar = np.logspace(24, 30, 100)
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


def change_of_bit(bit, sequence):

    if sequence[bit - 1] == sequence[bit]:
        return 0
    elif sequence[bit - 1] < sequence[bit]:
        return 1
    elif sequence[bit - 1] > sequence[bit]:
        return -1


def sequence_risetime_slow(ar, t, op, step_time, t_0, f):
    for i in ar:
        op += [i]
        t += step_time

    return op, t, t_0


def generate_random_full_sequence_with_risetime(length, f, step_time, rise_time, fall_time, p_on = P_on, p_off = P_off):
    assert type(length) == int

    total_time = (length * 1/f)
    sequence = generate_random_sequence(length, p_on, p_off)

    bit = 0
    t = 0
    t_0 = 0
    bit_change = 0
    op = []
    bit_change_index = [0]

    max_to_min = np.linspace(p_on, p_off, int(fall_time/step_time))
    min_to_max = np.linspace(p_off, p_on, int(rise_time/step_time))

    while t < total_time:
        op += [sequence[bit]]
        t += step_time

        if t - t_0 >= 1/f:
            bit_change_index += [len(op) - 1]
            bit += 1
            t_0 = t
            if bit > len(sequence):
                raise ValueError

            bit_change = change_of_bit(bit, sequence)
            if bit_change == 0:
                pass
            elif bit_change == 1:
                op, t, t_0 = sequence_risetime_slow(min_to_max, t, op, step_time, t_0, f)
            elif bit_change == -1:
                op, t, t_0 = sequence_risetime_slow(max_to_min, t, op, step_time, t_0, f)
            else:
                raise Exception




    #print(op)
    # plt.plot(op)
    # plt.show()

    return op, bit_change_index





def p_out_for_sequence(given_sequence, seq):


    f = 10 * 10 ** 6

    #rise_time = 1/(20*f)

    time = [0]

    t_step = min(1/(100*f), 1*10**(-10))
    bit = 0

    rise_time = 1/(10*f)

    print(rise_time)
    fall_time = rise_time
    assert rise_time < 1/f

    symbol_no = 100
    assert type(symbol_no) == int

    sequence, bit_index = generate_random_full_sequence_with_risetime(symbol_no, f, t_step, rise_time, fall_time, P_on, P_off)
    print(len(sequence))

    #sequence = generate_random_full_sequence_without_risetime(100, f, t_step, P_on, P_off)

    n_t0 = n_steady_state(sequence[0])
    #print(n_t0)

    n_o = [n_t0]
    gain_ar = [gain_coefficient * (n_t0 - n_0_const)]
    P_o_ar = [gain_ar[-1] * sequence[0]]

    loss = 0 #gain loss coefficient

    while time[-1] < (1/f * len(sequence)):

        n_t1 = n_t0 + dn(n_t0, sequence[bit]) * t_step

        #print(time[-1], n_t0)

        gain_ar += [np.exp((gain_coefficient * (n_t1 - n_0_const) -loss)*Length)]

        n_o += [n_t1]
        n_t0 = n_t1

        P_o_ar += [gain_ar[-1]*sequence[bit]]

        time += [time[-1] + t_step]

        bit += 1

        if bit >= len(sequence):
            break
            raise ValueError


    #print(gain_ar)
    #print(time)
    #print(n_o)
    # plt.plot(time[1:], n_o[1:])
    # plt.title('Carrier Density Against Time')
    # plt.show()
    # plt.plot(time[1:], gain_ar[1:])
    # plt.title('Gain against time')
    # plt.show()
    # #print(P_o_ar.shape())
    # plt.plot(time[1:], P_o_ar[1:])
    # plt.title('Power Out against time')
    # plt.show()
    return P_o_ar[1:], bit_index


#P_out, f, t_step, bit_index = p_out_for_sequence(0, [])


def bit_index_removal(ar):
    op = []
    for i in range(len(ar)):
        if i%3 == 0:
            op += [bit_index[i]]

    return op




def eye_diagram_plot(P_out, bit_index):
    bit_index = bit_index_removal(bit_index)
    #print(len(P_out), int(3/(t_step*f)), 1/f, t_step)
    for i in range(len(bit_index)-1):
        plt.plot(P_out[bit_index[i]:bit_index[i+1]])
    plt.show()


P_out, bit_index = p_out_for_sequence(0, [])
eye_diagram_plot(P_out, bit_index)


for i in range(2):
    P_out, bit_index = p_out_for_sequence(1, P_out)
    eye_diagram_plot(P_out, bit_index)




#eye_diagram_plot(P_out, f, t_step, bit_index)

