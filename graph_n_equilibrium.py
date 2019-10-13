import rate_model
import numpy as np
import matplotlib.pyplot as plt

n_initial_array = np.logspace(18, 22, 4)

for n_initial in n_initial_array:
    e = rate_model.E
    I = rate_model.I_initial

    t = 0

    n_out = []
    n_t = n_initial

    n_out += [n_t]

    t_out = []
    t_out += [t]

    delta_t = rate_model.delta_t_default

    while t < rate_model.t_stop:
        if t > rate_model.t_stop/2:
            I = rate_model.I_initial*2

        n_t1 = n_t + rate_model.dn(n_t, I)*delta_t
        print(t, n_t)

        t += delta_t

        n_out += [n_t1]
        t_out += [t]

        n_t = n_t1

    plt.semilogy(t_out, n_out)

plt.show()