"""Calculation of quantum reaction-diffusion dynamics from derived rate equations.
This module contains three initial conditions and multiple rate equation belonging
to different reaction types."""

# Imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
L = 4000
DISTANCE = 1
N_BOSONS = 4000
REL_TOL = 1e-8
ABS_TOL = 1e-8
T_START = 0
T_END = 1e6
SCALE = 0.5

# Plot values
time_steps = np.logspace(-3,6, 200)
scaled_time = SCALE*time_steps

superposition_angle = [np.pi/4, np.pi/3, np.pi/5]
distances = [0,1,20]


# Initial Conditions
def n_init_bc() -> np.array:
    """This function returns an array of length L. Each position represents the
    the lattice position in momentum space of an 1-dimensional spin chain.
    The value of an element in this array represents the number of bosons at a
    certain position in the lattice.
    The number of bosons N_BOSONS is sorted after the Bose condensate, that is, all
    bosons are at momentum q=0.

    Returns:
        np.array: returns array with lattice length. Each element of this array
                gives the number of bosons in each lattice site at t=0.
    """
    n_init = np.arange(1, L+1)
    for idx, itm in enumerate(n_init):
        q = -np.pi + 2*np.pi*itm/L
        if  q == 0:
            n_init[idx] = N_BOSONS
        else:
            n_init[idx] = 0
    return n_init


def n_init_flat() -> np.array:
    """This function places one boson at every momentum of a 1-D
    spin chain of length L.

    Returns:
        np.array: returns array with lattice length. Each element of this array
                gives the number of bosons in each lattice site at t=0.
    """
    n_init = np.ones(L)
    return n_init

def n_init_gauss() -> np.array:
    """This function gaussian distributes the number of bosons around q=0 for every
    momentum of a 1-D spin chain of length L.

    Returns:
        np.array: returns array with lattice length. Each element of this array
                gives the number of bosons in each lattice site at t=0.
    """
    q = -np.pi + 2*np.pi/L*np.arange(1, L+1)
    n_init = 2*np.pi**(1/2)*np.exp(-np.power(q,2))
    return n_init

# Equations of motion
def first_order_annihilation(t: np.array, n: np.array,
                            length: float, distance: float, theta: float) -> np.array:
    """Defines right-hand-side of rate equation for first-order annihilation dynamics of bosons.
    The paramemter n (boson density) depends on t.

    Args:
        t (np.array): time
        n (np.array): density
        length (float): lattice length
        distance (float): particle reaction distance
        theta (float): superposition angle

    Returns:
        np.array: returns right-hand-side of rate equation to be solved
    """
    k = -np.pi + 2*np.pi/length*np.arange(1, length+1)
    term_1 = np.sum(n)*(2 - np.sin(2*theta)*np.cos(2*distance*k))
    term_2 = 2*np.matmul(n, np.cos(distance*k))*(np.cos(distance*k)
                                                - np.sin(2*theta)*np.cos(distance*k))
    term_3 = 2*np.matmul(n, np.sin(distance*k))*(np.sin(distance*k)
                                                + np.sin(2*theta)*np.sin(distance*k))
    term_4 = -np.matmul(n, np.cos(2*distance*k))*np.sin(2*theta)
    result = -1/length*n*(term_1 + term_2 + term_3 + term_4)
    _ = (t,)
    return result

def second_order_annihilation(t: np.array, n: np.array, length: float):

    k = -np.pi + 2*np.pi/length*np.arange(1, length+1)
    term_1 = np.matmul(n, np.power(np.cos(k),2))
    term_3 = np.sum(n)
    term_2 = np.matmul(n, np.cos(k))
    result = -4/length*n*(term_1 + 2*term_2*np.cos(k) + term_3*(np.power(np.cos(k),2) -4*np.cos(k)+ 4) - 4*term_2)
    _ = (t,)
    return result


# Choosing initial condition
initial = n_init_gauss()


# Solving the rate equation and calculating the effective exponent at each time step
for index, item in enumerate(superposition_angle):
    sol = solve_ivp(fun=first_order_annihilation, t_span=(T_START, T_END), y0=initial,
            args=(L,1,item),atol=ABS_TOL, rtol=REL_TOL, t_eval=time_steps)
    average_density = np.sum(sol.y, axis=0)/len(sol.y)

    sol_scaled = solve_ivp(fun=first_order_annihilation, t_span=np.array((T_START, T_END))*SCALE,
            y0=initial, args=(L,1,item), t_eval=scaled_time, atol=ABS_TOL, rtol=REL_TOL)
    average_density_scaled = np.sum(sol_scaled.y, axis=0)/len(sol_scaled.y)
    exponent = -np.log(average_density_scaled/average_density)/np.log(SCALE)

    plt.plot(sol.t, average_density, label=str(item))


plt.grid()
plt.xlabel('time t')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
