"""Module providing the stationary phase diagramm of quantum reaction-diffusion
    systems with incoherent branching and coagulation reactions.
"""

# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Font Settings
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 17

# Constants
MAX_BETA = 0.9
MAX_DELTA = 3.2
CMAP= 'viridis'
FIGURE_COLOUR = 'white'

# Functions
def u2(beta: float, delta:float) -> float:
    """Calculates the factor of the quadratic term u2/2*n^2 of the potential W(n).

    Args:
        beta (float): branching rate
        delta (float): single-body decay rate

    Returns:
        float : rate of the quadratic term in potential W(n)
    """
    return delta-2*beta

def u3(beta: float, gamma=1) -> float:
    """Calculates the factor of the cubic term u3/3*n^3 of the potential W(n).

    Args:
        beta (float): branching rate
        gamma (int, optional): coagulation rate. Defaults to 1.

    Returns:
        float: rate of the cubic term in the potential W(n)
    """
    return 2*gamma-8*beta

def u4(beta: float, gamma=1) -> float:
    """Calculates the factor of the quartic term u4/4*n^4 of the potential W(n).

    Args:
        beta (float): branching rate
        gamma (int, optional): coagulation rate. Defaults to 1.

    Returns:
        float: rate of the quartic term in the potential W(n)
    """
    return 6*(gamma-beta)

def stationary_state(beta: float,delta: float) -> float:
    """Calculates the stationary state for non-zero density by minimizing the potential W(n).

    Args:
        beta (float): branching rate
        delta (float): decay rate

    Returns:
        float: density in the stationary state in the active phase
    """
    state = (-u3(beta)+np.sqrt(np.power(u3(beta),2)-4*u2(beta,delta)*u4(beta)))/(2*u4(beta))
    return state

def crit(beta: float,delta: float) -> float:
    """Calculates the critical relation between the branching and decay rate
    (for a coagulation rate of 1) that gives the line between the active and the absorbing phase.

    Args:
        beta (float): branching rate
        delta (float): decay rate

    Returns:
        float: rate relation that defines critical transition line
    """
    return -2*np.sqrt(u2(beta, delta)*u4(beta))


def find_state(beta: float,delta: float) -> float:
    """Finds the stationary state depending on the branching and the decay rate. Decides whether
    the system is in the absorbing or the active phase and then calculates the stationary density.

    Args:
        beta (float): branching rate
        delta (float): decay rate

    Returns:
        float: stationary state depending on given rates
    """
    if u2(beta,delta) <= 0:
        return stationary_state(beta,delta)
    elif (u2(beta,delta) > 0 and u3(beta) >= 0):
        return 0
    elif (u2(beta,delta) > 0 and u3(beta) < 0):
        if u3(beta) >= crit(beta,delta):
            return 0
        else:
            return stationary_state(beta,delta)


def transition_first_order(beta: float) -> float:
    """For a given relation beta/gamma (coagulation is set to 1) finds the relation delta/gamma
    (gamma=1) such that the stationary density lies on the first-order transition line.

    Args:
        beta (float): branching rate
    Returns:
        float: decay rate
    """
    delta = 2*beta + 1/24*np.divide(np.power(2-8*beta,2),(1-beta))
    return delta

def transition_second_order(beta: float) -> float:
    """For a given relation beta/gamma (coagulation is set to 1) finds the relation delta/gamma
    (gamma=1) such that the stationary density lies on the second-order transition line.

    Args:
        beta (float): branching rate

    Returns:
        float: decay rate
    """
    delta = 2*beta
    return delta


def polynom(x: np.array, a:float, b:float, c:float) -> np.array:
    """Simple polynom function to vividly show stable states depending on branching and decay rates.

    Args:
        x (np.array): density as a variable
        a (float): parameter of the quadratic term
        b (float): parameter of the cubic term
        c (float): parameter of the quatric term

    Returns:
        np.array: action potential of the density
    """
    return np.power(x,2)*a+ b*np.power(x,3)+c*np.power(x,4)


# Calculating the stationary density for given branching and decay rates
branching_rates = np.linspace(0,MAX_BETA,1000)
decay_rates = np.linspace(0,MAX_DELTA,1000)

stationary_densities = np.array([find_state(i,j) for i in branching_rates for j in decay_rates])
matrix_sd = stationary_densities.reshape(1000,1000)

# Settings for the phase diagramm
beta_first = np.linspace(0.255,MAX_DELTA/2,1000) #rate ranges we want to plot
beta_second = np.linspace(0,0.25,1000)
beta_bistable = np.linspace(0.25,MAX_DELTA/2,1000)

d_cuts = [0.6, 0.5, 0.4] # cut plots for certain delta values

# Starting figure
fig = plt.figure()

gs =  fig.add_gridspec(3,2, width_ratios=[1.5,0.5])
gs.update(hspace=0.4, wspace=0.4)

pd = fig.add_subplot(gs[:,0])  # phase diagram
cut_a = fig.add_subplot(gs[0,1]) # cut at delta value above critical point
cut_crit = fig.add_subplot(gs[1,1]) # cut at delta value at critical point
cut_b = fig.add_subplot(gs[2,1]) # cut at delta value below critical point

x_data_1 = np.linspace(0,10,1000) # numeric values for potential sketch
x_data_2 = np.linspace(0,1.5,1000) # numeric values for potential sketch
x_data_3 = np.linspace(0,0.8,1000) # numeric values for potential sketch


#connection lines between subplots
pd.annotate("",xy=(0, d_cuts[0]), xycoords=pd.transData,xytext=(MAX_BETA, d_cuts[0]),
            textcoords=pd.transData, arrowprops=dict(arrowstyle="-",linestyle= (0, (1, 5)),
            linewidth=1.5,color='green'))
pd.annotate("",xy=(MAX_BETA, d_cuts[0]), xycoords=pd.transData,xytext=(-0.15, -0.05),
            textcoords=cut_a.transData,arrowprops=dict(arrowstyle="<-",color='green'))
pd.annotate("",xy=(0, d_cuts[1]), xycoords=pd.transData,xytext=(MAX_BETA, d_cuts[1]),
            textcoords=pd.transData,arrowprops=dict(arrowstyle="-",linestyle= (0, (1, 5)),
            linewidth=1.5,color='orange'))
pd.annotate("",xy=(MAX_BETA, d_cuts[1]), xycoords=pd.transData,xytext=(-0.1, 0),
            textcoords=cut_crit.transData,arrowprops=dict(arrowstyle="<-",color='orange'))
pd.annotate("",xy=(0, d_cuts[2]), xycoords=pd.transData,xytext=(MAX_BETA, d_cuts[2]),
            textcoords=pd.transData,arrowprops=dict(arrowstyle="-",linestyle= (0, (1, 5)),
            linewidth=1.5,color='red'))
pd.annotate("",xy=(MAX_BETA, d_cuts[2]), xycoords=pd.transData,xytext=(-0.07, 0.3),
            textcoords=cut_b.transData,arrowprops=dict(arrowstyle="<-",color='red'))



# Heatmap
pd.imshow(matrix_sd.T, origin='lower', extent=[0,MAX_BETA,0,MAX_DELTA],
        vmin=0, vmax=round(find_state(MAX_BETA,0)), cmap=CMAP, aspect='auto')
pd.plot(beta_first,transition_first_order(beta_first),color='green', zorder=2)
pd.plot(beta_second,transition_second_order(beta_second),color='red', zorder=1)
pd.plot(beta_bistable,transition_second_order(beta_bistable),color='red',linestyle='dashed')
pd.scatter(1/4,1/2, color='orange', zorder=3)
pd.set_xlabel(r'$\beta/\gamma$')
pd.set_ylabel(r'$\delta/\gamma$')
pd.set_yticks([0,0.5,1,1.5,2,2.5,3], labels=[0,0.5,1.0,1.5,2.0,2.5,3.0])
pd.set_xticks([0,0.25,0.5,0.75], labels=[0,0.25,0.5,0.75])

# Heatmap - labels
pd.text(0.14,0.8,s='bicritical', color='orange', fontsize=13)
pd.text(0.14,0.65,s='point', color='orange', fontsize=13)
pd.text(0.15,MAX_DELTA-0.2,s='c) absorbing phase', color='white', fontsize=13)
pd.text(0.22,MAX_DELTA-0.38,s=r'$(\mathrm{n}_{\mathrm{ss}} = 0)$', color='white', fontsize=13)
pd.text(0.6,0.25,s='a) active phase', color='white', fontsize=13)
pd.text(0.65,0.07,s=r'$(\mathrm{n}_{\mathrm{ss}} \neq 0)$', color='white', fontsize=13)
pd.text(0.72,2.9,s='b) bistable', color='white', fontsize=11)
pd.text(0.76,2.7,s='phase', color='white', fontsize=11)
pd.fill_between(np.linspace(0.255,MAX_DELTA/2,1000),
                transition_first_order(np.linspace(0.255,MAX_DELTA/2,1000)),
                transition_second_order(np.linspace(0.255,MAX_DELTA/2,1000)),
                facecolor='grey', alpha=.5)
pd.tick_params(top=True,right=True, which='both')

# Insert no.1 - sketch of potential W(n) in absorbing phase
pot_active = inset_axes(pd, width=0.5, height=0.5, bbox_to_anchor=(0.36,0.75,0.1,0.1),
                    bbox_transform=pd.transAxes)
pot_active.plot(x_data_1, polynom(x_data_1,1,1,1), color=FIGURE_COLOUR)
pot_active.spines[['right', 'top']].set_visible(False)
pot_active.spines['bottom'].set_position('zero')
pot_active.spines['left'].set_position('zero')
pot_active.spines['bottom'].set_color(FIGURE_COLOUR)
pot_active.spines['left'].set_color(FIGURE_COLOUR)
pot_active.plot(1, 0, marker=">", ms=3, transform=pot_active.get_yaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_active.plot(0, 1, marker="^", ms=3, transform=pot_active.get_xaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_active.tick_params(left=False, bottom=False)
pot_active.xaxis.set_ticklabels([])
pot_active.yaxis.set_ticklabels([])
pot_active.set_ylabel(r'$W(n)$', fontsize=10, rotation=0, y=0.6, labelpad=5, color=FIGURE_COLOUR)
pot_active.set_xlabel(r'$n$',fontsize=10, labelpad=-5, color=FIGURE_COLOUR, loc='right')
pot_active.patch.set_facecolor('white')
pot_active.patch.set_alpha(0)

# Insert no.2 - sketch of potential W(n) in bistable phase
pot_bistable = inset_axes(pd, width=0.5, height=0.5, bbox_to_anchor=(0.92,0.72,0.1,0.1),
                    bbox_transform=pd.transAxes)
pot_bistable.plot(x_data_2, polynom(x_data_2,5,-8.5,3.7), color=FIGURE_COLOUR)
pot_bistable.spines[['right', 'top']].set_visible(False)
pot_bistable.spines['bottom'].set_position('zero')
pot_bistable.spines['left'].set_position('zero')
pot_bistable.spines['bottom'].set_color(FIGURE_COLOUR)
pot_bistable.spines['left'].set_color(FIGURE_COLOUR)
pot_bistable.plot(1, 0, marker=">", ms=3, transform=pot_bistable.get_yaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_bistable.plot(0, 1, marker="^", ms=3, transform=pot_bistable.get_xaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_bistable.tick_params(left=False, bottom=False)
pot_bistable.xaxis.set_ticklabels([])
pot_bistable.yaxis.set_ticklabels([])
pot_bistable.set_ylabel(r'$W(n)$', fontsize=10, rotation=0, y=0.6, labelpad=5, color=FIGURE_COLOUR)
pot_bistable.set_xlabel(r'$n$',fontsize=10, labelpad=-5, color=FIGURE_COLOUR, loc='right')
pot_bistable.patch.set_facecolor('white')
pot_bistable.patch.set_alpha(0)

# Insert no.3 - sketch of potential W(n) in active phase
pot_absorbing = inset_axes(pd, width=0.5, height=0.5,bbox_to_anchor=(0.85,0.3,0.1,0.1),
                    bbox_transform=pd.transAxes)
pot_absorbing.plot(x_data_3, polynom(x_data_3,-1.4,-0.6,5), color=FIGURE_COLOUR)
pot_absorbing.spines[['right', 'top']].set_visible(False)
pot_absorbing.spines['bottom'].set_position('zero')
pot_absorbing.spines['left'].set_position('zero')
pot_absorbing.spines['bottom'].set_color(FIGURE_COLOUR)
pot_absorbing.spines['left'].set_color(FIGURE_COLOUR)
pot_absorbing.plot(1, 0, marker=">", ms=3, transform=pot_absorbing.get_yaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_absorbing.plot(0, 1, marker="^", ms=3, transform=pot_absorbing.get_xaxis_transform(),
            clip_on=False, color=FIGURE_COLOUR)
pot_absorbing.tick_params(left=False, bottom=False)
pot_absorbing.xaxis.set_ticklabels([])
pot_absorbing.yaxis.set_ticklabels([])
pot_absorbing.set_ylabel(r'$W(n)$', fontsize=10, rotation=0, y=0.6, labelpad=5, color=FIGURE_COLOUR)
pot_absorbing.set_xlabel(r'$n$',fontsize=10, labelpad=-5, color=FIGURE_COLOUR, loc='right')
pot_absorbing.patch.set_facecolor('white')
pot_absorbing.patch.set_alpha(0)


#density plots for fixed delta/gamma
cut_a.scatter(branching_rates,
            matrix_sd[:,np.where(np.isclose(decay_rates,d_cuts[0],atol=1e-03))[0][0]],
            color='green', s=1.5)
cut_a.set_ylabel(r'$\mathrm{n}_{\mathrm{SS}}$',ha='left', y=1.2, rotation=0)
cut_a.text(0.1,0.8,s=r'$\delta/\gamma = $'  + str(d_cuts[0]),
        fontsize=13,horizontalalignment='left',
        verticalalignment='center', transform=cut_a.transAxes)
cut_a.set_ylim(-0.01,0.4)
cut_a.set_xlim(0,0.35)
cut_a.set_xticks([0,0.15,0.3], labels=[0,0.15,0.3])
cut_a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_a.set_yticks([0,0.2,0.4], labels=[0,0.2,0.4])
cut_a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_a.tick_params(top=True,right=True, which='both', direction="in")

cut_crit.scatter(branching_rates,
            matrix_sd[:,np.where(np.isclose(decay_rates,d_cuts[1],atol=1e-03))[0][0]],
            color='orange', s=1.5)
cut_crit.text(0.1,0.8,s=r'$\delta/\gamma = $'  + str(d_cuts[1]),
        fontsize=13,horizontalalignment='left',
        verticalalignment='center', transform=cut_crit.transAxes)
cut_crit.set_xticks([0,0.15,0.3], labels=[0,0.15,0.3])
cut_crit.xaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_crit.set_yticks([0,0.2,0.4], labels=[0,0.2,0.4])
cut_crit.yaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_crit.tick_params(top=True,right=True, which='both', direction="in")
cut_crit.set_ylim(-0.01,0.4)
cut_crit.set_xlim(0,0.35)

cut_b.scatter(branching_rates,
            matrix_sd[:,np.where(np.isclose(decay_rates,d_cuts[2],atol=1e-03))[0][0]],
            color='red', s=1.5)
cut_b.text(0.1,0.8,s=r'$\delta/\gamma = $'  + str(d_cuts[2]),
        fontsize=13,horizontalalignment='left',
        verticalalignment='center', transform=cut_b.transAxes)
cut_b.set_xlabel(r'$\beta/\gamma$')
cut_b.set_xticks([0,0.15,0.3], labels=[0,0.15,0.3])
cut_b.xaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_b.set_yticks([0,0.2,0.4], labels=[0,0.2,0.4])
cut_b.yaxis.set_minor_locator(ticker.AutoMinorLocator())
cut_b.tick_params(top=True,right=True, which='both', direction="in")
cut_b.set_ylim(-0.01,0.4)
cut_b.set_xlim(0,0.35)


#colormap
divider = make_axes_locatable(pd)
cax = divider.append_axes('top', '5%', pad=0.1)
pcm = pd.imshow(matrix_sd.T, origin='lower', extent=[0,MAX_BETA,0,MAX_DELTA],
                vmin=0, vmax=round(find_state(MAX_BETA,0)), cmap=CMAP, aspect='auto')
cbar = plt.colorbar(pcm, cax=cax, orientation='horizontal', ticklocation='top',
                    ticks=np.linspace(0,round(find_state(MAX_BETA,0)),4))
cbar.set_label(r'$\mathrm{n}_{\mathrm{SS}}$')

fig.subplots_adjust(bottom=0.15)
plt.show()

# End-of-file (EOF)
