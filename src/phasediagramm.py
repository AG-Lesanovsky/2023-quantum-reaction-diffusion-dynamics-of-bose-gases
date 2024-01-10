'''
This script creates a phase diagram for the contact process of non-interacting bosonic particles.

This script requires the packages numpy and matplotlib.
'''

#Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Setting font etc. for the final image
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 17

#Function
def u2(beta,delta):
    '''This function calculates the factor belonging to the quadratic term u2/2*n^2.
    It depends on the single particle decay rate delta and the branching rate beta.'''
    return delta-2*beta

def u3(beta,gamma=1):
    '''This function calculates the factor belonging to the term u3/3*n^3.
    It depends on the coagulation rate gamma and the branching rate beta. Without loss of generality
    we set gamma=1.'''
    return 2*gamma-8*beta

def u4(beta,gamma=1):
    '''This function calculates the factor belonging to the term u4/4*n^4.
    It depends on the coagulation rate gamma and the branching rate beta.Without loss of generality
    we set gamma=1.'''
    return 6*(gamma-beta)

def stationary_state(beta,delta):
    '''Calculates the stationary state in the active phase depending on the abovely introcuced factors. Also gamma=1.'''
    state = (-u3(beta)+np.sqrt(np.power(u3(beta),2)-4*u2(beta,delta)*u4(beta)))/(2*u4(beta))
    return state

def crit(beta,delta):
    '''Calculates the critical relation between beta and delta. (For gamma=1)'''
    return -3*np.sqrt(1/2*u2(beta, delta)*u4(beta))


def find_state(beta,delta):
    '''Finds the correct stationary state depending on the rates. (For gamma=1)'''
    if (u2(beta,delta) <= 0):
        return stationary_state(beta,delta)
    elif (u2(beta,delta) > 0 and u3(beta) >= 0):
        return 0
    elif (u2(beta,delta) > 0 and u3(beta) < 0):
        if (u3(beta) >= crit(beta,delta)):
            return 0
        else:
            return stationary_state(beta,delta)
        

def transition_first_order(delta):
    '''Calculates the branching rates on the transition line from the absorbing to the active phase. (For gamma=1, beta/gamma, delta/gamma)'''
    beta = 1/20*(9*np.sqrt(9*np.power(delta,2)+28*delta+4)-27*delta-22)
    return beta

def transition_second_order(beta):
    '''Calculates the decay rates on the transition line from the absorbing to the active phase. (For gamma=1, beta/gamma, delta/gamma)'''
    delta = 2*beta
    return delta


#Choosing values to plot phase diagramm
beta = np.linspace(0,0.9,1000)
delta = np.linspace(0,1,1000)

density = np.array([find_state(i,j) for i in beta for j in delta])
D = density.reshape(1000,1000)

beta_first = np.linspace(0.51,1,1000)
beta_second = np.linspace(0,0.244,1000)

#Plot settings
plt.imshow(D, origin='lower',  extent=[0,1,0,0.9],vmin=0, vmax=round(find_state(0.65,0)))
cbar = plt.colorbar(shrink=0.75)
cbar.set_label(r'$n_{SS}$')
plt.scatter(1/2,1/4, color='orange')
plt.plot(beta_first,transition_first_order(beta_first), color='yellow',linestyle='dashed')
plt.plot(transition_second_order(beta_second),beta_second,color='red')
plt.ylabel(r'$\beta/\gamma$')
plt.xlabel(r'$\delta/\gamma$')
plt.xlim(0,1)
plt.ylim(0,0.7)
plt.text(0.55,0.23,s='1st order transition', color='yellow', rotation=18)
plt.text(0.1,0.0,s='2nd order transition', color='red', rotation=27.5)
plt.text(0.5,0.09,s='absorbing phase', color='white')
plt.text(0.58,0.03,s=r'$n_{ss} = 0$', color='white')
plt.text(0.08,0.6,s='active phase', color='white')
plt.text(0.13,0.54,s=r'$n_{ss} \neq 0$', color='white')
plt.text(0.4,0.45,s='bicritical', color='orange')
plt.text(0.4,0.4,s='point', color='orange')
plt.arrow(0.45, 0.38, 0.035, -0.09, color='orange', head_width=0.015, width = 0.0005)
plt.axvline(0.39, color='black',linestyle='dashed')
plt.axvline(0.61, color='black',linestyle='dashed')
plt.show()