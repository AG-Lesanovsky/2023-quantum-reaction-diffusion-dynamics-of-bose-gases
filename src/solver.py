'''
This module contains important tools to solve differential equations of quantum reaction diffusion systems is the reaction-limited
regime for bosonic particles.

This module requires the packages: numpy.

'''

import numpy as np
from scipy.integrate import solve_ivp


class Lattice:
    '''creates a lattice'''

    def __init__(self, no_sites: float):

        self.sites = no_sites




class InitialStates:
    '''Contains all used initial conditions'''

    def __init__(self, no_occupied_sites: float, lattice: Lattice):
        '''initializes  initial state'''

        self.sites = lattice.sites
        self.occupation = no_occupied_sites


    def bec(self):
        '''initial state is a bose condensate, i.e. only one mode (quasi-momentum = 0) is occupied by all bosons in the system'''
        
        lattice = np.arange(1, self.sites + 1)

        for idx, itm in enumerate(lattice):
            q = -np.pi + 2*np.pi*itm/self.sites
            if  q == 0:
                lattice[idx] = self.occupation
            else:
                lattice[idx] = 0
        return lattice
    
    
    def flat_filling(self):
        '''initial lattice is evenly filled with the same amount of particles on every lattice site (i.e. quasi-momenta - reciprocal lattice)'''
        
        no_bosons = self.occupation/self.sites
        lattice = np.arange(1, self.sites + 1)

        for idx, itm in enumerate(lattice):
            lattice[idx] = no_bosons
        return lattice


    def __help_gaussian(self, q: np.array):
        '''help function that gaussain distributes an array of quasi-momenta '''

        gauss_function = 2*(np.pi)**(1/2)*np.exp(-np.power(q,2))

        return gauss_function
    

    def gaussian(self):
        '''initial state is such that occupation is gaussian ditributed among all quasi-momenta (around q=0)'''

        q = -np.pi + 2*np.pi/self.sites*np.arange(1, self.sites + 1)

        states = self.__help_gaussian(q)

        return states
    


class RateEquations:
    '''contains rate equations for the used reaction types'''

    def __init__(self, time: np.array, density: np.array, lattice: Lattice):
        '''initializes rate equation'''

        self.t = time
        self.n = density

        self.sites = lattice.sites

        self.q =  -np.pi + 2*np.pi/self.sites*np.arange(1, self.sites+1)


    def  coagulation(self):
        '''rate equation of coagulation reaction'''

        return  -(6*np.power(self.n,3)  + 2*np.power(self.n,2))
    

    def first_order_annihilation(self, theta: float, d: float):
        '''rate equation of first order annihilation reaction, additional information is the distance of reaction range d 
        and the angle defining the channel strength of reactants.'''

        term_1 = np.sum(self.n)*(2 - np.sin(2*theta)*np.cos(2*d*self.q))
        term_2 = 2*np.matmul(self.n, np.cos(d*self.q))*(np.cos(d*self.q) - np.sin(2*theta)*np.cos(d*self.q))
        term_3 = 2*np.matmul(self.n, np.sin(d*self.q))*(np.sin(d*self.q) + np.sin(2*theta)*np.sin(d*self.q))
        term_4 = -np.matmul(self.n, np.cos(2*d*self.q))*np.sin(2*theta)

        result = -1/self.sites*self.n*(term_1 + term_2 + term_3 + term_4)

        return result
    
    def second_oder_annihilation(self):

        term_1 = np.matmul(self.n, np.power(np.cos(self.q),2))
        term_3 = np.sum(self.n)
        term_2 = np.matmul(self.n, np.cos(self.q))

        result = -4/self.sites*self.n*(term_1 + 2*term_2*np.cos(self.q) + term_3*(np.power(np.cos(self.q),2) -4*np.cos(self.q)+ 4) - 4*term_2)
    
        return result
    


class Solver:
    '''solves differential rate equations for given initial state'''

    def __init__(self, time_steps: np.array, time_length: float, tolerance: float, initial: InitialStates, equation):
        '''
        initializes initial value problem solver
        
        :param initial: Should be a method of InitialStates
        :param equation: Should be a method of RateEquations
        '''

        self.start = 0
        self.end = time_length

        self.steps = time_steps
        self.tol = tolerance

        self.initial = initial
        self.equation = equation

        self.arg = ()

    def solve(self):

        solution = solve_ivp(fun=self.equation, t_span=(self.start, self.end), y0=self.initial)


