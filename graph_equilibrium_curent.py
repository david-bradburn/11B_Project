import rate_model
import numpy as np
import matplotlib.pyplot as plt

I_array = np.linspace(0.001, 1, 5)

for I_initial in I_array:
    e = rate_model.E
    I = I_initial
    p_0 = rate_model.P

    t = 0

    n_out = []
    n_t = rate_model.n_0_const

    n_out += [n_t]

    t_out = []
    t_out += [t]

    delta_t = rate_model.t_stop/10

    while t < rate_model.t_stop:

        n_t1 = n_t + rate_model.dn(n_t, p_0 , I)*delta_t
        print(t, n_t)

        t += delta_t

        n_out += [n_t1]
        t_out += [t]

        n_t = n_t1

    plt.semilogy(t_out, n_out, label = I)

plt.legend()
plt.show()