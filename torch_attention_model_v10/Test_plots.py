import torch
from Lib import *
import argparse
import json
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)
    


# Load the overall data
x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')
P = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/Test_MEM_data.p')

print(x.keys())

omega_fine = x['omega_fine']
omega = x['omega']
tau = x['tau']
R = np.concatenate([x['One_Peak_R_interp'],\
    x['Two_Peak_R_interp'] ], axis=0)
_, _, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
# print(E.shape, R.shape)


## The one peak
# The test data (one peak)
R_test_1 = np.concatenate([  P['One_Peak_R']], axis=0)
R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
E_test_2 = np.concatenate([  P['Two_Peak_E']], axis=0)
R_test = np.concatenate([R_test_1, R_test_2], axis=0) 
E_test = np.concatenate([E_test_1, E_test_2], axis=0) 

E_test_1 = torch.from_numpy(E_test_1)
R_test_1 = torch.from_numpy(R_test_1)
E_test_2 = torch.from_numpy(E_test_2)
R_test_2 = torch.from_numpy(R_test_2)
E_test   = torch.from_numpy(E_test)
R_test   = torch.from_numpy(R_test)



testset_one = MyDataset(E_test_1, R_test_1)
testloader_one = DataLoader(testset_one, batch_size=128, shuffle=False)

testset_two = MyDataset(E_test_2, R_test_2)
testloader_two = DataLoader(testset_two, batch_size=128, shuffle=False)

testset= MyDataset(E_test, R_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False)






#### We got our basis and it is good
Kern = Kern.astype('float64')
Kern_R[:, (Kern_R.shape[1]-1)] = 1
torch.pi = torch.acos(torch.zeros(1)).item() 



### Load the required model
rbfnet = Network(Kern, Kern_R)
rbfnet.load_state_dict(torch.load('results/models/modella_both_00', map_location=torch.device('cpu')) )
rbfnet.to(device)



def evaluate_data(model, testloader, ERR):
    import matplotlib.pyplot as plt
    import numpy 
    model.eval()
    Ehat    = np.zeros([0,151])
    Rhat    = np.zeros([0,2000])
    E_test  = np.zeros([0,151])
    R_test  = np.zeros([0,2000])
    R_err  = np.zeros([0,2000])
    E_err  = np.zeros([0,151])

    for x_batch, y_batch in testloader:
        x_batch = x_batch.float().to(device) 
        y_batch = y_batch.float().to(device) 
        
        ehat, rhat, _, rvar, evar = model.forward(x_batch, ERR)   
        
        Rhat   = np.concatenate([Rhat, rhat.detach().cpu().numpy()], axis = 0)
        Ehat   = np.concatenate([Ehat, ehat.detach().cpu().numpy()], axis = 0)
        E_test = np.concatenate([E_test, x_batch.detach().cpu().numpy()], axis = 0)
        R_test = np.concatenate([R_test, y_batch.detach().cpu().numpy()], axis = 0)
        E_err = np.concatenate([E_err, abs( (ehat-evar).cpu().detach().numpy()) ], axis = 0)
        R_err = np.concatenate([R_err, abs( (rhat-rvar).cpu().detach().numpy()) ], axis = 0)
        
    return Rhat, Ehat, E_test, R_test, E_err, R_err




# # Gather the reconstructions from my model
# Err = 1e-4
# Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet, testloader_one, Err)
# print(Rhat.shape, Ehat.shape, E_test.shape, R_test.shape, E_err.shape, R_err.shape)



# Gather the reconstructions from MEM
# R_MEM_1  = P['One_Peak_R_MEM']
# E_MEM_1  = P['One_Peak_E_MEM']
# E_MEM_2  = P['Two_Peak_E_MEM']
# R_MEM_2  = P['Two_Peak_R_MEM']
# R_MEM = np.concatenate([R_MEM_1, R_MEM_2], axis=0) 
# E_MEM = np.concatenate([E_MEM_1, E_MEM_2], axis=0) 


# print("########CHi2######")
# vals(chi2_vec(E_test, Ehat, Err))
# print("#############R2###########")
# vals(1-get_r2_numpy_manual_vec(Rhat, R_test))
# print("$R^2$")
# vals(1-get_r2_numpy_manual_vec(Ehat, E_test))
# print("#############Entropy###########")
# vals(-1*Entropy(R_test, Rhat, int_coeff = Kern_R))



# # Now, I have to generate all the plots
# Err = 1e-4
# Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet, testloader_two, Err)
# print(Rhat.shape, Ehat.shape, E_test.shape, R_test.shape, E_err.shape, R_err.shape)
# print("########CHi2######")
# vals(chi2_vec(E_test, Ehat, Err))
# print("#############R2###########")
# vals(1-get_r2_numpy_manual_vec(Rhat, R_test))
# print("$R^2$")
# vals(1-get_r2_numpy_manual_vec(Ehat, E_test))
# print("#############Entropy###########")
# vals(-1*Entropy(R_test, Rhat, int_coeff = Kern_R))


tag = 'orig'
# # Now, I have to generate all the plots
Err = 1e-1
Rhat, Ehat, _, _, E_err, R_err = evaluate_data(rbfnet,\
testloader, Err)

np.savetxt(tag+'Ehat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'Rhat_err_'+str(Err)+'.csv', Ehat, delimiter=',')

np.savetxt(tag+'E_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'R_err_'+str(Err)+'.csv', Ehat, delimiter=',')


# # Now, I have to generate all the plots
Err = 1e-2
Rhat, Ehat, _, _, E_err, R_err = evaluate_data(rbfnet,\
testloader, Err)
np.savetxt(tag+'Ehat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'Rhat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'E_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'R_err_'+str(Err)+'.csv', Ehat, delimiter=',')


# # Now, I have to generate all the plots
Err = 1e-3
Rhat, Ehat, _, _, E_err, R_err = evaluate_data(rbfnet,\
testloader, Err)
np.savetxt(tag+'Ehat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'Rhat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'E_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'R_err_'+str(Err)+'.csv', Ehat, delimiter=',')



# # Now, I have to generate all the plots
Err = 1e-4
Rhat, Ehat, _, _, E_err, R_err = evaluate_data(rbfnet,\
testloader, Err)

np.savetxt(tag+'Ehat_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'Rhat_err_'+str(Err)+'.csv', Ehat, delimiter=',')

np.savetxt(tag+'E_err_'+str(Err)+'.csv', Ehat, delimiter=',')
np.savetxt(tag+'R_err_'+str(Err)+'.csv', Ehat, delimiter=',')


# print(Rhat.shape, Ehat.shape, E_test.shape, R_test.shape, E_err.shape, R_err.shape)
# print("########CHi2######")
# vals(chi2_vec(E_test, Ehat, Err))
# print("#############R2###########")
# vals(1-get_r2_numpy_manual_vec(Rhat, R_test))
# print("$R^2$")
# vals(1-get_r2_numpy_manual_vec(Ehat, E_test))
# print("#############Entropy###########")
# vals(-1*Entropy(R_test, Rhat, int_coeff = Kern_R))





# chi2 = chi2_vec(Ehat, E_test, factor= Err)
# R2   = 1-(get_r2_numpy_manual_vec(Rhat, R_test))
# Ent = -1*Entropy(R_test, Rhat, int_coeff = Kern_R)

# list_best_index = [best_five(Ent, 5)[0], best_five(Ent, 5)[1], best_five(Ent, 5)[2]]
# print(Ent[list_best_index])
# print(chi2[list_best_index])
# print(list_best_index)

# Rhat = np.expand_dims(Rhat, axis = 0)
# R  = np.expand_dims(R_test, axis = 0)
# Ehat = np.expand_dims(Ehat, axis = 0)
# E  = np.expand_dims(E_test, axis = 0)


# print(Rhat.shape, R.shape, R_MEM.shape)
# print(Ehat.shape, E.shape, E_MEM.shape)


# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.font_manager as font_manager
# large = 10; med = 10; small = 10
# marker_size = 1.01
# lw = 1
# inten = 0.4
# index = np.arange(0,151, 1)
# list_best_index = [1540, 1600, 1720]
# labels = ['UQ-NN', 'Original', 'MaxEnt']
# #########################################################
# ####################

# def cm2inch(value):
#     return value/2.54
# plt.style.use('seaborn-white')
# COLOR = 'darkslategray'
# params = {'axes.titlesize': small,
#           'legend.fontsize': small,
#           'figure.figsize': (cm2inch(24),cm2inch(9)),
#           'axes.labelsize': med,
#           'axes.titlesize': small,
#           'xtick.labelsize': small,
#           'lines.markersize': marker_size,
#           'ytick.labelsize': med,
#           'figure.titlesize': small, 
#           'font.family': "sans-serif",
#           'font.sans-serif': "Myriad Hebrew",
#             'text.color' : COLOR,
#             'axes.labelcolor' : COLOR,
#             'axes.linewidth' : 0.5,
#             'xtick.color' : COLOR,
#             'ytick.color' : COLOR}

# plt.rcParams.update(params)
# plt.rc('text', usetex = False)
# color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', '#9467bd']
# plt.rcParams['mathtext.fontset'] = 'cm'

# # create plots with numpy array
# fig,a =  plt.subplots(2,3, dpi = 1200,\
# gridspec_kw = {'wspace':0.25, 'hspace':0.4})


# element = list_best_index[0]
# ##################CME
# # Some Plot oriented settings 
# a[0][0].grid(linestyle=':', linewidth=1)
# # Some Plot oriented settings 
# a[1][0].grid(linestyle=':', linewidth=1)

# # The final Plots with CME
# mean_R = np.mean(Rhat[:,element,:], axis = 0)
# std_R  = np.std( Rhat[:,element,:], axis = 0)
# org_R  = R[0, element,:]
# mem_R = R_MEM[element,:]
# omega=omega_fine.reshape([-1])
# tau=tau.reshape([-1])

# print(mean_R.shape, std_R.shape, org_R.shape, mem_R.shape, omega.shape, tau.shape)


# a[0][0].plot(omega, org_R, color = color[0], linewidth = lw, linestyle = '-')
# a[0][0].fill_between(omega, (mean_R+std_R), (mean_R-std_R), alpha=inten, color = color[1])
# a[0][0].plot(omega, mean_R, color = color[1], linewidth = lw, linestyle = '--')
# a[0][0].plot(omega, mem_R, color = color[2], linewidth = lw, linestyle = '-.')

# a[0][0].set_xlabel('$\omega~[\mathrm{MeV}]$')
# a[0][0].set_ylabel('$R(\omega)~[\mathrm{MeV}^{-1}]$')


# mean_E = np.mean(Ehat[:,element,:], axis = 0)
# std_E  = np.std(Ehat[:,element, :], axis = 0)
# org_E  = E[0, element,:]
# mem_E  = E_MEM[element,:]

# a[1][0].errorbar(tau[index], org_E[index], yerr = 0.0001, fmt = 'x', color = color[0],\
#                  ms = 0.00001, linewidth = 0.4, label = labels[1])
# a[1][0].scatter(tau, org_E, color = color[0])
# a[1][0].fill_between(tau, (mean_E+std_E), (mean_E-std_E), alpha=inten, color = color[1])
# a[1][0].plot(tau, mean_E, color = color[1], linewidth = lw, linestyle = '--', label = labels[0])
# a[1][0].plot(tau, mem_E, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])

# a[1][0].set_yscale('log')
# a[1][0].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$\n  Best')
# a[1][0].set_ylabel('$E(\\tau)$')
# a[0][0].set_xlim([0,250])
# a[1][0].set_yticks([ 0.01, 0.1, 1])


# ###################################
# element = list_best_index[1]
# ##################CME
# # Some Plot oriented settings 
# a[0][1].grid(linestyle=':', linewidth=1) 
# # Some Plot oriented settings 
# a[1][1].grid(linestyle=':', linewidth=1) 

# # The final Plots with CME
# mean_R = np.mean(Rhat[:,element,:], axis = 0)
# std_R  = np.std( Rhat[:,element,:], axis = 0)
# org_R  = R[0, element,:]
# mem_R  = R_MEM[element,:]


# a[0][1].plot(omega, org_R, color = color[0], linewidth = lw, linestyle = '-', label = labels[1])
# a[0][1].fill_between(omega, (mean_R+std_R), (mean_R-std_R), alpha=inten, color = color[1])
# a[0][1].plot(omega, mean_R, color = color[1], linewidth = lw, linestyle = '--')
# a[0][1].plot(omega, mem_R, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])
# a[0][1].set_xlabel('$\omega~[\mathrm{MeV}]$')

# # The final Plots with CME
# mean_E = np.mean(Ehat[:,element,:], axis = 0)
# std_E  = np.std(Ehat[:,element, :], axis = 0)
# org_E  = E[0, element,:]
# mem_E  = E_MEM[element,:]


# a[1][1].errorbar(tau[index], org_E[index], yerr = 0.0001, fmt = 'x', color = color[0],\
#                  ms = 0.00001, linewidth = 0.4, label = labels[1])
# a[1][1].scatter(tau, org_E, color = color[0])
# a[1][1].fill_between(tau, (mean_E+std_E), (mean_E-std_E), alpha=inten, color = color[1])
# a[1][1].plot(tau, mean_E, color = color[1], linewidth = lw, linestyle = '--', label = labels[0])
# a[1][1].plot(tau, mem_E, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])
 
# a[1][1].set_yscale('log')
# a[1][1].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$\n  Median')
# a[0][1].set_xlim([0,750])
# a[1][1].set_yticks([0.0001, 0.001, 0.1, 1])



# ###################################
# element = list_best_index[2]
# ##################CME
# # Some Plot oriented settings 
# a[0][2].grid(linestyle=':', linewidth=1)
# a[1][2].grid(linestyle=':', linewidth=1)

# mean_R = np.mean(Rhat[:,element,:], axis = 0)
# std_R  = np.std( Rhat[:,element,:], axis = 0)
# std_R = std_R
# org_R  = R[0, element,:]
# mem_R  = R_MEM[element,:]


# a[0][2].plot(omega, org_R, color = color[0], linewidth = lw, linestyle = '-', label = labels[1])
# a[0][2].fill_between(omega, (mean_R+std_R), (mean_R-std_R), alpha=inten, color = color[1])
# a[0][2].plot(omega, mean_R, color = color[1], linewidth = lw, linestyle = '--', label = labels[0])
# a[0][2].plot(omega, mem_R, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])
# a[0][2].set_xlabel('$\omega~[\mathrm{MeV}]$')
# # a[0][1].set_ylabel('$R(\omega)~[MeV^{-1}]$')

# a[0][2].legend(loc = 'upper right',ncol=1 )

# # The final Plots with CME
# mean_E = np.mean(Ehat[:,element,:], axis = 0)
# std_E  = np.std(Ehat[:,element, :], axis = 0)
# org_E  = E[0, element,:]
# mem_E  = E_MEM[element,:]



# a[1][2].errorbar(tau[index], org_E[index], yerr = 0.01, fmt = 'x', color = color[0],\
#                  ms = 0.00001, linewidth = 0.4, label = labels[1])
# a[1][2].scatter(tau, org_E, color = color[0])
# a[1][2].fill_between(tau, (mean_E+std_E), (mean_E-std_E), alpha=inten, color = color[1])
# a[1][2].plot(tau, mean_E, color = color[1], linewidth = lw, linestyle = '--', label = labels[1])
# a[1][2].plot(tau, mem_E, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])

 
# a[1][2].set_yscale('log')
# a[1][2].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$\n  Worst')
# # a[1][1].set_ylabel('$E(\\tau)$')
# a[0][2].set_xlim([0,400])
# # a[0][2].set_ylim([0,0.05])
# # a[1][2].set_yticks([0.000001, 0.01])


# from matplotlib import ticker
# a[0][2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
# a[0][1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
# a[0][0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))


# a[0][2].spines["top"].set_visible(False)    
# a[0][2].spines["bottom"].set_visible(False)    
# a[0][2].spines["right"].set_visible(False)    
# a[0][2].spines["left"].set_visible(True)  
# a[0][2].grid(linestyle=':', linewidth=0.5)

# a[0][1].spines["top"].set_visible(False)    
# a[0][1].spines["bottom"].set_visible(False)    
# a[0][1].spines["right"].set_visible(False)    
# a[0][1].spines["left"].set_visible(True)  
# a[0][1].grid(linestyle=':', linewidth=0.5)
    
# a[0][0].spines["top"].set_visible(False)    
# a[0][0].spines["bottom"].set_visible(False)    
# a[0][0].spines["right"].set_visible(False)    
# a[0][0].spines["left"].set_visible(True)  
# a[0][0].grid(linestyle=':', linewidth=0.5)
    
# a[1][0].spines["top"].set_visible(False)    
# a[1][0].spines["bottom"].set_visible(False)    
# a[1][0].spines["right"].set_visible(False)    
# a[1][0].spines["left"].set_visible(True)  
# a[1][0].grid(linestyle=':', linewidth=0.5)
    
# a[1][1].spines["top"].set_visible(False)    
# a[1][1].spines["bottom"].set_visible(False)    
# a[1][1].spines["right"].set_visible(False)    
# a[1][1].spines["left"].set_visible(True)  
# a[1][1].grid(linestyle=':', linewidth=0.5)
    
    
# a[1][2].spines["top"].set_visible(False)    
# a[1][2].spines["bottom"].set_visible(False)    
# a[1][2].spines["right"].set_visible(False)    
# a[1][2].spines["left"].set_visible(True)  
# a[1][2].grid(linestyle=':', linewidth=0.5)

# plt.savefig('Two_Peak.png', bbox_inches='tight', pad_inches = 0, dpi=1200) 
# plt.show()


# # print("One Peak"
# # _,_=rbfnet.evaluate_METRICS(testloader_one, fac_var=-1,\
# #     save_dir='.', epoch=10000000, filee='one_peak')
# # rbfnet.evaluate_plots(testloader_one, omega_fine, tau, epoch=10000000,\
# #         save_dir='.', filee='one_peak', fac=-1)

# # print("Two Peak")
# # _,_=rbfnet.evaluate_METRICS(testloader_two, fac_var=-1,\
# #     save_dir='.', epoch=10000000, filee='two_peak')
# # rbfnet.evaluate_plots(testloader_two, omega_fine, tau, epoch=100000000,\
# #         save_dir='.', filee='two_peak', fac=-1)