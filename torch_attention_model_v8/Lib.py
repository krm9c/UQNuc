import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.font_manager as font_manager
from Lib import *

# Data importing functions
def return_dict(i):
    import pickle
    with (open(i, "rb")) as openfile:
        while True:
            try:
                x = pickle.load(openfile)
            except EOFError:
                break
    return x



# First define a function that finds the nearest neighbors
def x_near(x, x_v):
    x_d = np.abs(x_v - x)
    x_index = np.argsort(x_d)
    return x_index

# Construct the interpolating polynomials up to second order
def polint_quadratic(x, x_v):
    n = x.shape[0]
    n_v = x_v.shape[0]
    poly = np.zeros((n,3))
    x_index = np.zeros((n, n_v), dtype=int)
    for i in range (n):
        x_index[i] = x_near(x[i], x_v)
        x_i = x_v[x_index[i, 0:3]]   
        poly[i,0] = (x[i]-x_i[1])*(x[i]-x_i[2])/(x_i[0]-x_i[1])/(x_i[0]-x_i[2])
        poly[i,1] = (x[i]-x_i[0])*(x[i]-x_i[2])/(x_i[1]-x_i[0])/(x_i[1]-x_i[2])
        poly[i,2] = (x[i]-x_i[0])*(x[i]-x_i[1])/(x_i[2]-x_i[0])/(x_i[2]-x_i[1])
    return poly, x_index

# Actual interpolation function
def interp_quadratic(x, x_v, y_v, poly, x_index):
    y = np.zeros(x.shape[0])
    for i in range (x.shape[0]):
        y[i] = np.sum( poly[i, 0:3] * y_v[x_index[i, 0:3]])
    return y


def interpolate_Alessandro(omega_fine, omega_, R_):
    omega_fine = omega_fine.reshape([-1])
    R_temp = np.zeros([R_.shape[0],omega_fine.shape[0]])
    poly, x_index = polint_quadratic(omega_fine.reshape([-1]), omega_.reshape([-1]))
    print("I am starting the main interpolation loop")
    for i in range(R_.shape[0]):
        R_temp[i,:] = interp_quadratic(omega_fine.reshape([-1]),\
         omega_.reshape([-1]), R_[i,:].reshape([-1]), poly, x_index)
    return R_temp, poly, x_index

def interpolate_integrate_self(tau_, omega_, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    # Get the fine grid
    dw = wmax / nw
    omega_fine  = np.zeros(nw)
    for i in range (nw):
        omega_fine[i] = (i + 0.5) * dw
    # Get the integration coefficients.
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern_R[0,j] = (omega_fine[j+1,0]-omega_fine[j,0]) 
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])*Kern_R[0,j]   
    E_      = np.zeros([n_points,n_tau])
    R_temp, poly, x_index  =  interpolate_Alessandro(omega_fine, omega_, R_)
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_temp) ) )
    R_temp[R_temp<1e-08] = 1e-08
    return E_, R_temp, omega_fine, Kern, Kern_R, poly, x_index

def integrate(tau_, omega_, omega_fine, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    dw = wmax / nw
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern_R[0,j] = (omega_fine[j+1,0]-omega_fine[j,0])  
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])*dw   
    E_      = np.zeros([n_points,n_tau])
    # R_[R_<1e-08] = 1e-08
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_) ) )
    return E_, R_, Kern, Kern_R

# The Metric calculation
def chi2_vec(y, yhat, factor):
    return np.mean(((y-yhat)**2/factor**2), axis = 1)


def get_r2_numpy_manual_vec(y_true, y_pred):
    ybar = np.mean(y_true, axis = 1).reshape([-1,1])
    SS_res = np.sum(np.square(y_pred - y_true), axis = 1) 
    SS_tot = np.sum(np.square(y_true - ybar), axis = 1) 
    r2 = (1-(SS_res/(SS_tot+0.00000001)))
    return r2


def Entropy(x, xhat, int_coeff = 1):
    import numpy as np
    return np.sum(xhat-x-xhat*np.log( np.divide(xhat,x) ) , axis = 1)


def best_five(x, n): 
    import numpy as np 
    return np.argsort(x)[:n]


def worst_five(x, n): 
    import numpy as np
    ranked = np.argsort(x)
    return ranked[::-1][:n]


# Let us create some plots with these
def Corr_plot(x, y, filename, x_inch, y_inch, xlabel, ylabel, labels, xlim = None, ylim = None, save_fig = False, log_scale = False):
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    
    def cm2inch(value):
            return value/2.54
        
    small = 10
    med = 10
    large = 12
    
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(x_inch),cm2inch(y_inch)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.linewidth' : 0.3,
            'xtick.color' : COLOR,
            'ytick.color' : COLOR}

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    large = 24; med = 8; small = 7
    fig,a =  plt.subplots(1,1, sharex = False, dpi = 1000, gridspec_kw = {'wspace':0.7, 'hspace':0.7})
    # Some Plot oriented settings 
    a.spines["top"].set_visible(True)    
    a.spines["bottom"].set_visible(True)    
    a.spines["right"].set_visible(True)    
    a.spines["left"].set_visible(True)  
    a.grid(linestyle=':', linewidth=0.5)
    a.get_xaxis().tick_bottom()    
    a.get_yaxis().tick_left()  
    leg = np.round(np.corrcoef(x,y)[0,1], 3)**2
    a.scatter(x, y, alpha=0.8, c=color[0], edgecolors='none', s=30, label = labels +'(Correlation ='+str(leg)+')' )
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    # a.get_xaxis().get_major_formatter().scaled['100'] = '%y'
    
    
    if log_scale is True:
        a.set_yscale('log')
    
    if xlim is not None:
        a.set_xlim(xlim)
    
    if ylim is not None:
        a.set_ylim(ylim)
    # a.legend()
    a.legend(bbox_to_anchor=(0.01, -1, 0.3, 0.1), loc = 'upper left',ncol=1, markerscale=1)
    
    if save_fig is True:
        plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight') 
    plt.show()



def Box_plot(data, labels, lims, filename, log_scale = False, spacing = 0.4):
    import pandas as pd
    
    import matplotlib.pyplot as plt
    def cm2inch(value):
            return value/2.54
    small = 10
    med = 11
    large = 12
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(6),cm2inch(4)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
                'text.color' : COLOR,
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 0.3,
                'xtick.color' : COLOR,
                'ytick.color' : COLOR}

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    fig,a =  plt.subplots(1,data.shape[1], sharex = False, dpi = 1000,\
                          gridspec_kw = {'wspace':spacing, 'hspace':spacing})
    c = 'slategray'
    for i in range(data.shape[1]):
        a[i].boxplot(data[:,i], patch_artist=True,
            boxprops=dict(facecolor = 'white', color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='steelblue'),
            )

        a[i].set_xlabel(labels[i])
        a[i].set_ylim(lims)
        if log_scale is True:
            a[i].set_yscale('log')
        
    plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight') 
    

def vals(chi_2_values):
    print("no. of responses with chi2 E > 5:", len(chi_2_values[chi_2_values>1.5]))
    # chi_2_values = chi_2_values[chi_2_values<1]
    print("Min", np.min(chi_2_values) )
    print("Median", np.median(chi_2_values) )
    print("Max", np.max(chi_2_values) )
    print("Mean", chi_2_values.mean())
    print("std", chi_2_values.std()/np.sqrt(len(chi_2_values)))



# The Metric calculation
def chi2_vec(y, yhat, factor):
    return np.mean(((y-yhat)**2/factor**2), axis = 1)


def get_r2_numpy_manual_vec(y_true, y_pred):
    ybar = np.mean(y_true, axis = 1).reshape([-1,1])
    SS_res = np.sum(np.square(y_pred - y_true), axis = 1) 
    SS_tot = np.sum(np.square(y_true - ybar), axis = 1) 
    r2 = (1-(SS_res/(SS_tot+0.00000001)))
    return r2


def Entropy(x, xhat, int_coeff):
    import numpy as np
    return -1*np.sum(xhat-x-xhat*np.log( np.divide(xhat,x) ) , axis = 1)


def best_five(x, n): 
    import numpy as np 
    return np.argsort(x)[:n]


def worst_five(x, n): 
    import numpy as np
    ranked = np.argsort(x)
    return ranked[::-1][:n]


def plot_responses(data_E, data_R, org_E, list_index, limits, filename_list, \
                   omega, tau, labels, save_fig = False):

    # Let us create some plots with these
    import matplotlib.pyplot as plt
    def cm2inch(value):
            return value/2.54
        
    small = 12
    med = 16
    large = 16
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(16),cm2inch(12)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
                'text.color' : COLOR,
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 0.3,
                'xtick.color' : COLOR,
                'ytick.color' : COLOR}

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'

    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    for k,element in enumerate(list_index):
        fig,a =  plt.subplots(2,1, sharex = False, dpi = 1000, gridspec_kw = {'wspace':0.3, 'hspace':0.3})
        # Some Plot oriented settings 
        a[0].spines["top"].set_visible(False)    
        a[0].spines["bottom"].set_visible(False)    
        a[0].spines["right"].set_visible(False)    
        a[0].spines["left"].set_visible(True)  
        a[0].grid(linestyle=':', linewidth=0.5)
        a[0].get_xaxis().tick_bottom()    
        a[0].get_yaxis().tick_left()  

        # Some Plot oriented settings 
        a[1].spines["top"].set_visible(False)    
        a[1].spines["bottom"].set_visible(False)    
        a[1].spines["right"].set_visible(False)    
        a[1].spines["left"].set_visible(True)  
        a[1].grid(linestyle=':', linewidth=0.5)
        a[1].get_xaxis().tick_bottom()    
        a[1].get_yaxis().tick_left()  


        for i, datum in enumerate(data_R):
            a[0].plot(omega,  datum[element,:], color = color[i], linewidth = 1.5, linestyle = '-.',\
            label = labels[i] )

        for i, datum in enumerate(data_E):
            chi_value = np.round( chi2_vec( org_E[element,:].reshape([-1,1]),\
            datum[element,:].reshape([-1,1]), 0.0001).mean(),3)
            
            a[1].errorbar(tau, datum[element,:], yerr = 0.0001, fmt = 'x', color = color[i], ms = 1,\
            linewidth = 1, label = '  '+labels[i]+'\n$(\chi^{2}_{E}, $'+str(chi_value)+')')


        a[0].set_xlim(limits[k])
        a[0].set_xlabel('$\omega~[MeV]$')
        a[0].set_ylabel('$R(\omega)~[MeV^{-1}]$')

        a[1].set_yscale('log')
        a[1].set_xlabel('$\\tau~[MeV^{-1}]$')
        a[1].set_ylabel('$E(\\tau)$')

        a[1].legend(bbox_to_anchor=(-0.1, -0.4, 0.3, 0.1), loc = 'upper left',ncol=3, markerscale=2 )
        
        if save_fig is True:
            plt.savefig(filename_list[k]+'.pdf', dpi = 800,  bbox_inches='tight') 
        
        plt.show()