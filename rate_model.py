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


def dn(n, p = P, noise = 0, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const):
    if noise == 0:
        return -(n / tau_s) + I / (e * V) - g * (n - n_0) * p
    else:
        return -(n/tau_s) + I/(e*V) - g*(n - n_0)*p + (-(n/tau_s) + I/(e*V) - g*(n - n_0)*p)/20* (np.random.random() - 0.5)


def n_steady_state(p = P, I = I_initial, tau_s = tau_s_default, e = E, V = volume, g = gain_coefficient, n_0 = n_0_const, delta_t = delta_t_default): #Find the steady state of n while everything else is constant
    t = 0
    n_t1 = 0
    n_t = n_0

    while True:

        n_t1 = n_t + dn(n_t, p, 0, I)*delta_t

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


def power_rand():
    return ((P_on - P_off)/25 * (np.random.random() - 0.5))


def sequence_risetime_slow(ar, t, op, step_time, t_0, f):
    for i in ar:
        op += [i + power_rand()]
        t += step_time

    return op, t, t_0

def add_input_noise(power):
    for i in range(len(power)):
        power[i] += power_rand()
    return power


def generate_random_full_sequence_with_risetime(length, f, step_time, rise_time, fall_time, p_on = P_on, p_off = P_off, add_noise = 0):
    assert type(length) == int

    total_time = (length * 1/f)
    sequence = generate_random_sequence(length, p_on, p_off)

    bit = 0
    t = 0
    t_0 = 0

    op = []
    bit_change_index = [0]

    max_to_min = np.linspace(p_on, p_off, int(fall_time/step_time))
    #print(len(max_to_min))
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

    if add_noise == 1:
        op = add_input_noise(op)

    #print(op)
    # plt.plot(op)
    # plt.show()

    return op, bit_change_index


# pw, bci = generate_random_full_sequence_with_risetime(500, 1000 * 10 ** (6), 1/(1000* 10 ** 6 * 100), 1/(1000* 10 ** 6 * 10), 1/(1000* 10 ** 6 * 10), P_on, P_off , 1)
# plt.plot(pw)

def p_out_for_sequence(given_sequence, seq, loss, noise_ip, noise_sys):

    f = 10000 * 10 ** 6

    time = [0]

    t_step = min(1/(100*f), 1*10**(-10))
    bit = 0

    rise_time = 1/(10*f)

    fall_time = rise_time
    assert rise_time < 1/f

    symbol_no = 1000
    assert type(symbol_no) == int

    sequence, bit_index = generate_random_full_sequence_with_risetime(symbol_no, f, t_step, rise_time, fall_time, P_on, P_off, noise_ip)

    #   sequence = generate_random_full_sequence_without_risetime(100, f, t_step, P_on, P_off)

    if given_sequence == 1:
        sequence = seq

    n_t0 = n_steady_state(sequence[0])
    #   print(n_t0)

    n_o = [n_t0]
    gain_ar = [gain_coefficient * (n_t0 - n_0_const)]
    P_o_ar = [gain_ar[-1] * sequence[0]]

    while time[-1] < (1/f * len(sequence)):

        n_t1 = n_t0 + dn(n_t0, sequence[bit], noise_sys) * t_step

        gain_ar += [np.exp((gain_coefficient * (n_t1 - n_0_const) - loss)*Length)]

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
    # # #print(P_o_ar.shape())
    # plt.plot(time[1:], [x * ppp * volume for x in P_o_ar[1:]])
    # plt.title('Power Out against time')
    # plt.show()
    return P_o_ar[1:], bit_index


#P_out, bit_index = p_out_for_sequence(0, [], 0, 0)


def power_chop(power, bit_index):

    print(len(bit_index))
    for m in range(int(len(bit_index)) - 2):

        power1 = (power[bit_index[m]: bit_index[(m + 1)]])

        test = bit_index[(m + 2)]
        power2 = (power[bit_index[(m + 1)]: bit_index[(m + 2)]])

        p1_av = 0
        p2_av = 0

        for i in power1:
            p1_av += i
        p1_av /= len(power1)

        for n in power2:
            p2_av += n
        p2_av /= len(power2)

        print(p1_av, p2_av)

#power_chop(P_out, bit_index)


def bit_index_removal(ar):
    op = []
    for i in range(len(ar)):
        if i % 3 == 0:
            op += [ar[i]]

    return op


def eye_diagram_plot(P_out, bit_index, passno, loss, type):

    bit_index = bit_index_removal(bit_index)

    plt.figure()
    for i in range(1, len(bit_index)-1):
        plt.plot([x * ppp * volume for x in P_out[bit_index[i]:bit_index[i+1]]])
        plt.title("Eye Diagram after {} passes with a loss of {}".format(passno + 1, int(loss)))
        if type == 1:
            plt.ylim(0, 2 * 10 ** (-7))
        elif type == 0:
            if loss > 10000:
                plt.ylim(0, 1.4 * 10 ** (-7))
            else:
                plt.ylim(0, 2 * 10 ** (-7))
                pass


def loss_analysis(type):
    P_out, bit_index = p_out_for_sequence(0, [], 0, 0, 0)
    eye_diagram_plot(P_out, bit_index, 0, 0, type)
    #xmin, xmax, ymin, ymax = axis()
    for i in np.logspace(4, 5.5, 4):
        P_out, bit_index = p_out_for_sequence(0, [], i, 0, 0)
        eye_diagram_plot(P_out, bit_index, 0, i, type)


def pass_through_multiple(type, loss):
    total = 0
    print(1/(total + 1))
    P_out, bit_index = p_out_for_sequence(0, [], loss, 0, 1)
    eye_diagram_plot(P_out, bit_index, 0, loss, type)

    for i in range(total):
        print((i + 2)/(total + 1))

        P_out, bit_index = p_out_for_sequence(1, P_out, loss, 0, 1)
        eye_diagram_plot(P_out, bit_index, i + 1, loss, type)

loss = 0
pass_through_multiple(0, loss)         #0 pass multi
#loss_analysis(1)                        #1 loss analysis

plt.show()
