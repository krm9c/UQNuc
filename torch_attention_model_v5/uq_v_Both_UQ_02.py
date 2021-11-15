import torch
from Lib import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import math 

torch.pi = torch.acos(torch.zeros(1)).item() 

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
    R_[R_<1e-08] = 1e-08
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


# class Network_inverter(nn.Module):
#     def __init__(self, input_shape=151, k=20):
#         super(Network_inverter, self).__init__()

#         # A linear Layer
#         self.l1=nn.Linear(in_features=151, out_features=500)
#         self.l2=nn.Linear(in_features=500, out_features=500)

#         self.n_peaks = k
#         self.mean_internal_layers = nn.ModuleList()
#         for i in range(self.n_peaks):
#             self.mean_internal_layers .append(
#             nn.Linear(in_features=500, out_features=1))

#         self.var_internal_layers = nn.ModuleList()
#         for i in range(self.n_peaks):
#             self.var_internal_layers .append(
#             nn.Linear(in_features=500, out_features=1))

#         # self.alpha_internal_layers = nn.ModuleList()
#         # for i in range(self.n_peaks):
#         #     self.alpha_internal_layers .append(
#         #     nn.Linear(in_features=500, out_features=1))

#     ##########################################
#     def forward(self, x):
#         x = x.float()
#         out=torch.sigmoid(self.l1(x))
#         out=torch.sigmoid(self.l2(out))
#         ##########################################
#         # for i in range(len(self.rbf_layers)):
#         mean_out_list=[]
#         var_out_list= []
#         alpha_out_list= []
#         for i in range(self.n_peaks):
#             mean_out_list.append(  torch.sigmoid(self.mean_internal_layers[i](out) ) )      
#             var_out_list. append(  torch.sigmoid((self.var_internal_layers[i](out) )) )
#             # alpha_out_list.append(  torch.nn.ReLU()(self.alpha_internal_layers[i](out)) )

#         return mean_out_list, var_out_list, alpha_out_list



class Network_selector(nn.Module):
    def __init__(self, input_shape=151, k=2):
        super(Network_selector, self).__init__()
        self.l1  =nn.Linear(in_features=151, out_features=151)
        self.l2  =nn.Linear(in_features=151, out_features=151)
        self.l3  =nn.Linear(in_features=151, out_features=151)

    ##########################################
    def forward(self, x):
        x = x.float()
        return self.l2(torch.nn.Sigmoid()(self.l1(x))),\
               self.l3(torch.nn.Sigmoid()(self.l1(x)))


class Network(nn.Module):
    def __init__(self, kern, kern_R, input_shape=151, k=2):
        super(Network, self).__init__()
        self.selector = Network_selector(k=k)
        self.kern=torch.from_numpy(kern)
        u, _, vh  = torch.svd(self.kern)
        # vh = torch.unsqueeze(vh, 0)
        self.U=vh

        # print("shapes U", self.U.shape, vh.shape)
        # self.U= torch.cat([vh for i in range(128)], dim=0)   
        self.kern_R=torch.from_numpy(kern_R)
        # self.inverter=torch.from_numpy(self.U)

    ##########################################
    def forward(self, x, omega, tau, epoch):

        x = x.float()
        # mean_l, var_l, a_l= self.inverter(x)
        # print("length", len(mean_l), len(var_l), mean_l[0].shape )
        select, var =self.selector(x)
        # print("select", select.shape, self.U.shape)

        #   2 / scale * scipy.stats.norm.pdf(t) * scipy.stats.norm.cdf(a*t)
        #   ! gauss(x) = 1/sqrt(2*pi)*exp(-x**2/2)
        #   ! Phi(x) = 1/2*(1+erf(x/sqrt(2)))
        #   f_t=[]
        #   for i, ele in enumerate(mean_l):
        #     # print(mean_l[i].shape, var_l[i].shape)
        #     # print("mean", mean_l[i], "var", var_l[i])
        #     # (omega-m/sig)
        #     # var_l[i]
        #     t=torch.div( (torch.from_numpy((omega.reshape([-1]))/2000)-mean_l[i]), (var_l[i]+1e-10) )
        #     # print(t.shape)
        #     norm_pdf= torch.mul(torch.div(1,(var_l[i]+1e-10)*(torch.sqrt(2*torch.tensor(math.pi)) ) ),\
        #         torch.exp(torch.mul( -0.5, t.pow(2) ) ) )
        #     # print("pdf", norm_pdf)
        #     scaled_cdf=0.5*(1+torch.erf(torch.mul(5,t)/math.sqrt(2)))
        #     # scaled_cdf =1
        #     # print(scaled_cdf)
        #     skewed_norm= torch.mul(norm_pdf,scaled_cdf)b
        #     # print(skewed_norm)
        #     # print(skewed_norm.shape)
        #     # oo=input("Enter some junk to continue")
        #     # f_t.append(norm_pdf.unsqueeze(0))
        #     f_t.append(skewed_norm.unsqueeze(0))

        #####################################################################
        # print("the output", select.shape, f_t[0].shape, select.t().unsqueeze(2).shape)
        # print("selecto", select.shape)
        select=select.double()
        var=var.double()
        self.U=self.U.double()
        # print(select.shape, self.U.shape)
        Rhat = torch.exp(torch.matmul(select, self.U.transpose(0,1))) 
        Rhat_var = torch.exp(torch.matmul(var, self.U.transpose(0,1))) 

        # print(Rhat.shape)
        #####################################################################
        # select=torch.repeat_interleave(select.t().unsqueeze(2), 2000, dim=2)
        # print("selecto maximo", select.shape)
        # F=torch.cat(f_t, dim=0)
        # print("R hactor", F)
        # print("selecto, Favtora", select)
        # print("before mult", select.shape, F.shape)
        # Rhat = (torch.sum(torch.mul(select, F), dim=0))
        # print(Rhat.shape)
        
        ##########################################
        # We have convert this code to pytorch
        # print(out.shape, self.kern.shape)
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)

        Ehat_var = torch.matmul(self.kern, Rhat_var.transpose(0, 1)).transpose(0, 1)
        correction_term_var = Ehat_var[:, 0].view(-1, 1).repeat(1, 2000)
        
        ##########################################
        # print(correction_term.shape, Ehat.shape, out.shape)
        Rhat = torch.div(Rhat, correction_term)
        multout = torch.matmul(self.kern, Rhat.transpose(0, 1))
        # print(multout.shape)
        Ehat = multout.transpose(0, 1)

        ##########################################
        # print(correction_term.shape, Ehat.shape, out.shape)
        Rhat_var = torch.div(Rhat_var, correction_term_var)
        multout_var = torch.matmul(self.kern, Rhat_var.transpose(0, 1))
        # print(multout.shape)
        Ehat_var = multout_var.transpose(0, 1)

        # Ehat=integrate(Rhat, omega, tau)
        return Ehat, Rhat, Ehat_var, Rhat_var

    ##########################################
    def loss_func(self, Ehat, Rhat, Evar, Rvar, E, R, fac):
        # The R loss
        non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
        loss_R = -torch.mean(non_integrated_entropy)
        # The E's loss
        loss_E   = torch.mean( torch.mul((Ehat- E).pow(2), (1/(0.0001*0.0001))) ) 
        loss_var = torch.mean( (Ehat- E).pow(2) + torch.log( torch.sqrt(2*torch.pi*Evar) ) - Evar.pow(2) )
        return (fac[0]*loss_R+fac[1]*loss_E+0.1*fac[1]*loss_var), loss_E, loss_R

    ##########################################
    def fit(self, x, y, omega, tau, epochs, batch_size, lr):
        self.train()
        obs = x.shape[0]
        LR = []
        LE = []
        x = torch.from_numpy(x)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.99)

        P = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/Test_MEM_data.p')
        R_test = np.concatenate([ P['One_Peak_R'], P['Two_Peak_R']], axis=0)
        E_test = np.concatenate([ P['One_Peak_E'], P['Two_Peak_E']], axis=0)
        print(E_test.shape, R_test.shape)
        E_test = torch.from_numpy(E_test)
        R_test = torch.from_numpy(R_test)
        testset = MyDataset(E_test, R_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        print("Everything is defined now")

        ##########################################
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss=0
            current_loss_E = 0
            current_loss_R = 0
            batches = 0
            progress = 0

            ##########################################
            if epoch < int(round(epochs*0.1)):
                factor_R=  1e6
                factor_E = 1e-6
            elif epoch < int(round(epochs*0.2)):
                factor_R=  1e7
                factor_E = 1e-4
                scheduler.step()
            elif epoch < int(round(epochs*0.3)):
                factor_R=  1e8
                factor_E = 1e-2
                scheduler.step()
            elif epoch < int(round(epochs*0.4)):
                factor_R= 1e10
                factor_E= 1e-1
                scheduler.step()
            else:
                factor_R=1e10
                factor_E=1
                scheduler.step()


            ##########################################
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                ##########################################
                tt = torch.rand(x_batch.shape[0], 1).float()
                x_batch = x_batch.float()+(1e-02)*torch.randn(x_batch.size())
                # x_batch = torch.cat((x_batch, tt), 1)
                x_batch = x_batch.to(device)

                y_batch = y_batch.float().to(device)
                xhat, yhat, xvar, yvar = self.forward(x_batch, omega, tau, epoch)
                loss, E_L, R_L = self.loss_func(xhat, yhat, xvar, yvar, x_batch, y_batch, [factor_R, factor_E])
                # print(loss, E_L, R_L)
                current_loss    += (1/batches) * (loss.cpu().item() - current_loss)
                current_loss_E  += (1/batches) * (E_L.cpu().item() - current_loss_E)
                current_loss_R  += (1/batches) * (R_L.cpu().item() - current_loss_R)
                loss.backward()
                optimiser.step()
                progress += x_batch.size(0)

                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f' %(epoch,\
                    progress, obs, current_loss, current_loss_R, current_loss_E))

                # print('\rEpoch: %d, Progress: %d/%d,\
                # Loss: %f' %(epoch, progress, obs, current_loss))
                sys.stdout.flush()
                # profiler.step()


            if epoch % 1 ==0:
                ##########################################
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots( 5,2, figsize=(12, 16) )
                for x_batch, y_batch in testloader:
                    y_batch = y_batch.float()
                    x_batch = x_batch.float()+(1e-02)*torch.randn(x_batch.size())
                
                    # tt = torch.zeros(x_batch.size(0), 1)+(alpha/2)
                    # # print(x_batch.shape, tau.shape)
                    # x_batch_up = torch.cat((x_batch, (tt)), 1)
                    # print(x_batch.shape, tau.shape)
                    x_hat, y_hat, xvar, yvar = self.forward(x_batch, omega, tau, epoch)
                    # print(y_hat.shape,y_batch.shape)

                    
                    for i in range(5):
                        ## PLOT THINGS ABOUT THE R
                        curve=y_batch[i,:].detach().numpy()
                        ax[i][0].plot((omega_fine).reshape([-1]), curve, '--', label='R', color='blue')    
                        Rhat = y_hat.detach().numpy()[i, :]
                        yerr = yvar.detach().numpy()[i, :]
                        fill_up = Rhat+yerr
                        fill_down = Rhat-yerr
                        fill_down[fill_down<0]= 0
                        ax[i][0].fill_between(omega_fine.reshape([-1]), fill_up, fill_down, alpha=1, color='orange')
                        ax[i][0].errorbar(omega_fine.reshape([-1]), Rhat, yerr=yerr, fmt='x', linewidth=0.1, ms = 0.2, label='Rhat', color='orange')
                        ax[i][0].set_xlim([0,500])
                        # ax[i][0].set_ylim([1e-10,1e-02])
                        ax[i][0].legend(loc='upper right')


                        ## PLOT THINGS ABOUT THE E
                        curve=x_batch[i,:].detach().numpy()
                        ax[i][1].plot( (tau).reshape([-1]), curve, label='E', color='blue')    
                        Ehat = x_hat.detach().numpy()[i, :]
                        yerr = xvar.detach().numpy()[i, :]
                        # fill_up = Ehat+yerr
                        # fill_down = Ehat-yerr
                        #fill_down[fill_down<0]= 0
                        #ax[i][1].fill_between(tau.reshape([-1]), fill_up, fill_down, alpha=1, color='orange')
                        ax[i][1].errorbar(tau.reshape([-1]), Ehat, yerr=yerr, fmt='x', linewidth=0.1, ms = 0.2, label='Ehat', color='orange')
                        ax[i][1].set_yscale('log')
                        ax[i][1].legend(loc='upper right')
                    break        
                fig.tight_layout()
                plt.savefig("sample_02/Rhat_1_"+str(epoch)+".png", dpi=300)
                plt.close()



device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# x = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/inverse_data_interpolated_numpy.p')
x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')


print(x.keys())
tau = x['tau']
omega_fine=x['omega_fine']
omega=x['omega']
# x = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/inverse_data_interpolated_numpy.p')
# x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')

R = np.concatenate([x['One_Peak_R_interp'], x['Two_Peak_R_interp']], axis=0)
print(R.shape)
# R = x['One_Peak_R_interp']
# /gpfs/jlse-fs0/users/kraghavan/Inverse
# Loaded=np.load('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/Inverse_new_Data.npz')
# # E=Loaded['E'][0:1000,:]
# R=Loaded['R'][0:10000,:]
# omega_fine = omega_fine/2000
# omega = omega/2000
E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
print(E.shape, R.shape)
# print("I defined the network")
Kern = Kern.astype('float64')
Kern_R[:, (Kern_R.shape[1]-1)] = 1
rbfnet = Network(Kern, Kern_R, k=20)
rbfnet.to(device)
# omega_fine = omega_fine/2000
# print("I moved the model to device")
_  =  rbfnet.fit(E, R, omega_fine, tau, 25, 128, 0.001)