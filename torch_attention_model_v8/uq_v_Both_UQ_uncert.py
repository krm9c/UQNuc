import torch
from Lib import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import math 

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



class Network_selector(nn.Module):
    def __init__(self, input_shape=151, k=2):
        super(Network_selector, self).__init__()
        self.l1  =nn.Linear(in_features=151, out_features=151)
        self.l2  =nn.Linear(in_features=151, out_features=151)
        self.l3  =nn.Linear(in_features=151, out_features=151)

    ##########################################
    def forward(self, x):
        x = x.float()
        x = torch.nn.Tanh()(self.l1(x))
        identity = x
        x = torch.nn.Tanh()(self.l2(x)) + identity
        skip =x
        return self.l3(x)+skip


class Network(nn.Module):
    def __init__(self, kern, kern_R, input_shape=151):
        super(Network, self).__init__()
        self.selector = Network_selector().to(device)
        self.kern=torch.from_numpy(kern).to(device)
        u, _, vh  = torch.svd(self.kern)
        self.U=vh
        self.kern_R=torch.from_numpy(kern_R).to(device)

    ##########################################
    def forward(self, x, sigma):
        x = x.float()
        self.U=self.U.double().to(device)
        select=self.selector(x).double().to(device)
        Rhat = torch.exp(torch.matmul(select, self.U.transpose(0,1))) 
        # Normalize E
        ####################################################################################
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)
        Rhat = torch.div(Rhat, correction_term)
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)

        # print("The value of sigma is", sigma)
        x_batch_N = (x+sigma*torch.rand(x.size()).to(device))
        select_N=self.selector(x).double().to(device)
        Rhat_N = torch.exp(torch.matmul(select_N, self.U.transpose(0,1))) 
        
        # Normalize E
        ####################################################################################
        Ehat_N = torch.matmul(self.kern, Rhat_N.transpose(0, 1)).transpose(0, 1)
        correction_term = Ehat_N[:, 0].view(-1, 1).repeat(1, 2000)
        Rhat_N = torch.div(Rhat_N, correction_term)
        multout_N = torch.matmul(self.kern, Rhat.transpose(0, 1))
        Ehat_N = multout_N.transpose(0, 1)

        return Ehat, Rhat, Ehat_N, Rhat_N, x_batch_N

    ####################################################################################
    def loss_func(self, Ehat, ENhat, Rhat,  E, EN, R, fac):
        # The R loss
        non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
        loss_R = -torch.mean(non_integrated_entropy)
        loss_E = torch.mean( torch.mul((Ehat- E).pow(2), (1/(0.0001*0.0001))))\
               + torch.mean( torch.mul((EN- ENhat).pow(2), (1/(fac[2]*fac[2]))) ) 
        return (fac[0]*loss_R+fac[1]), loss_E, loss_R

    def evaluate_METRICS(self, testloader, fac_var):
        self.eval()
        ### Final Numbers 
        Ent_list =[]
        chi2=[]
        for x_batch, y_batch in testloader:
            y_batch = y_batch.float().to(device)
            x_batch = x_batch.float().to(device)
            x_hat, y_hat, _, yvar, xvar = self.forward(x_batch, fac_var)
            # Calculate Entropy
            my_list = Entropy(y_batch.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), 1 )
            [Ent_list.append(item) for item in my_list]
            my_list = chi2_vec(x_batch.cpu().detach().numpy(), x_hat.cpu().detach().numpy(), 1e-04)
            [chi2.append(item) for item in my_list]
        print("#######################ENTROPIES#################################")
        # print("Entropy shapes", len(Ent_list_one), len(Ent_list_two) )
        Ent_list=np.array(Ent_list)
        Ent_list=np.array(Ent_list)
        print("Min", np.min(Ent_list) )
        print("Median", np.median(Ent_list) )
        print("Max", np.max(Ent_list) )
        print("Mean", Ent_list.mean())
        print("std", Ent_list.std())
        print("#######################Chi 2################################# \n")
        chi2=np.array(chi2)
        print("Min", np.min(chi2) )
        print("Median", np.median(chi2) )
        print("Max", np.max(chi2) )
        print("Mean", chi2.mean())
        print("std", chi2.std())
        return Ent_list, chi2


    def evaluate_plots(self, testloader, omega_fine, tau, epoch, save_dir, filee):
        import matplotlib.pyplot as plt
        import numpy 
        self.eval()
        for j in range(5):
            fig, ax = plt.subplots( 5,2, figsize=(16,15) )
            for x_batch, y_batch in testloader:
                y_batch = y_batch.float().to(device)
                x_batch = x_batch.float().to(device)
                for i in range(5):
                    x_hat, y_hat, _, yvar, xvar = self.forward(x_batch, 1e-4*pow(10,i))
                    ## PLOT THINGS ABOUT THE R
                    curve=y_batch[j,:].cpu().detach().numpy()
                    ax[i][0].plot((omega_fine).reshape([-1]), curve, '--', label='R('+str(1e-4*pow(10,i))+')', color='blue')    
                    Rhat = y_hat.cpu().detach().numpy()[j, :]
                    yerr = (Rhat-yvar.cpu().detach().numpy()[j, :])
                    fill_up = Rhat+yerr
                    fill_down = Rhat-yerr
                    fill_down[fill_down<0]= 0
                    ax[i][0].fill_between(omega_fine.reshape([-1]), fill_up, fill_down, alpha=1, color='orange')
                    ax[i][0].set_xlim([0,400])
                    ax[i][0].legend(loc='upper right')
                    ## PLOT THINGS ABOUT THE E
                    curve=x_batch[j,:].cpu().detach().numpy()
                    ax[i][1].plot( (tau).reshape([-1]), curve, label='E', color='blue')    
                    Ehat = x_hat.cpu().detach().numpy()[j, :]
                    yerr = abs(Ehat-xvar.cpu().detach().numpy()[j, :])
                    ax[i][1].errorbar(tau.reshape([-1]), Ehat, yerr=yerr, fmt='x', linewidth=2, ms = 1, label='Ehat', color='orange')
                    ax[i][1].set_yscale('log')
                    ax[i][1].legend(loc='upper right')
                    ax[i][0].set_xlabel('$ \\omega (MeV)$')
                    ax[i][1].set_xlabel('$ \\tau (MeV^{-1})$')
                    ax[i][0].set_ylabel('$ R(\\omega)(MeV^{-1})$')
                    ax[i][1].set_ylabel('$ E(\\tau)$')
                fig.tight_layout()
                plt.savefig(save_dir+filee+str(epoch)+'_'+str(j)+".png", dpi=300)
                plt.close()
                break
        return


    ####################################################################################
    def fit(self, trainloader, testloader_one, testloader_two,  omega, tau, epochs, batch_size, lr, save_dir, model_name, flag):
        self.train()
        obs = len(trainloader)*batch_size
        LR = []
        LE = []
        optimiser = torch.optim.RMSprop(self.parameters(),\
            lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.99)
        ####################################################################################
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss=0
            current_loss_E = 0
            current_loss_R = 0
            batches = 0
            progress = 0
            ####################################################################################
            ####################################################################################
            if epoch < int(round(epochs*0.2)):
                factor_R=  1e6
                factor_E = 1e-4
                fac_var=1e-04
            elif epoch == int(round(epochs*0.25)):
                factor_R=  1e6
                factor_E = 1e-3
                fac_var=1e-03
                scheduler.step()
            elif epoch == int(round(epochs*0.3)):
                factor_R=  1e5
                factor_E = 1e-2
                fac_var=1e-02
                scheduler.step()
            elif epoch == int(round(epochs*0.4)):
                factor_R= 1e5
                factor_E= 1e-01
                fac_var=1e-01
                scheduler.step()
            else:
                factor_R=1e5
                factor_E=1
                fac_var=1
            
            ####################################################################################
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                ####################################################################################
                x_batch = x_batch.float()
                x_batch   = x_batch.to(device)

                y_batch = y_batch.float().to(device)
                
                xhat, yhat, xNhat, _, x_batch_N = self.forward(x_batch, fac_var)
                loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
                                                x_batch_N, y_batch, \
                                                [factor_R, factor_E, fac_var] )

                current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
                current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
                current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R)
                loss.backward()
                optimiser.step()
                progress += x_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f' %(epoch,\
                progress, obs, current_loss, current_loss_R, current_loss_E) )
                # print('\rEpoch: %d, Progress: %d/%d,\
                # Loss: %f' %(epoch, progress, obs, current_loss))
                sys.stdout.flush()
                # profiler.step()
                ####################################################################################
            # Printing and saving code 
            if epoch % 1 ==0:

                if flag==0:
                    print("########################################################")
                    print("\n One Peak")
                    torch.save(self.state_dict(), model_name)
                    _,_=self.evaluate_METRICS(testloader_one, fac_var)
                    self.evaluate_plots(testloader_one, omega, tau, epoch,\
                         save_dir, filee='one_peak')
                elif flag==1:
                    print("########################################################")
                    print("\n Two Peak")
                    torch.save(self.state_dict(), model_name)
                    _,_=self.evaluate_METRICS(testloader_two, fac_var)
                    self.evaluate_plots(testloader_two, omega, tau,\
                         epoch, save_dir, filee='two_peak')
                    print("########################################################")
                else:
                    print("########################################################")
                    print("\n One Peak")
                    torch.save(self.state_dict(), model_name)
                    _,_=self.evaluate_METRICS(testloader_one, fac_var)
                    self.evaluate_plots(testloader_one, omega, tau, epoch,\
                         save_dir, filee='one_peak')
                    print("########################################################")
                    print("Two Peak")
                    _,_=self.evaluate_METRICS(testloader_two, fac_var)
                    self.evaluate_plots(testloader_two, omega, tau, epoch,\
                         save_dir, filee='two_peak')    
                    print("########################################################")
        return self




n_runs=1
for runs in range(n_runs):
    ## Theta
    # x = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/inverse_data_interpolated_numpy.p')
    ## JLSE
    x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')

    ## ThetaGPU
    # P = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/Test_MEM_data.p')
    ## JLSE
    P = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/Test_MEM_data.p')

    # The test data (one peak)
    R_test_1 = np.concatenate([  P['One_Peak_R']], axis=0)
    E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
    print(E_test_1.shape, R_test_1.shape)
    E_test_1 = torch.from_numpy(E_test_1)
    R_test_1 = torch.from_numpy(R_test_1)
    testset_one = MyDataset(E_test_1, R_test_1)
    testloader_one = DataLoader(testset_one, batch_size=128, shuffle=False)

    # # The test data (two peak)
    R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
    E_test_2 = np.concatenate([ P['Two_Peak_E']], axis=0)
    print(E_test_2.shape, R_test_2.shape)
    E_test_2 = torch.from_numpy(E_test_2)
    R_test_2 = torch.from_numpy(R_test_2)
    testset_two = MyDataset(E_test_2, R_test_2)
    testloader_two = DataLoader(testset_two, batch_size=128, shuffle=False)
    ## The device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Everything is defined now")



    ## The actual model
    load = 1
    flag = 2
    batche =128
    epochs = 500
    learning_rate=0.001
    if flag == 0:
        print("one peaks")
        save_dir = 'results/samples_one_peak/'
        model_ref = 'results/models/modella_uncert_one'+str(runs)

        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([x['One_Peak_R_interp']], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        print(E.shape, R.shape)
        # print("I defined the network")
        Kern = Kern.astype('float64')
        Kern_R[:, (Kern_R.shape[1]-1)] = 1
        print(torch.cuda.is_available())
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)
        rbfnet = Network(Kern, Kern_R)
        if load==0:
            rbfnet.load_state_dict(torch.load(model_ref))
        rbfnet.to(device)
        rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_one, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag)
    elif flag==1:
        print("two peaks")
        save_dir = 'results/samples_two_peak/'
        model_ref = 'results/models/modella_uncert_two'+str(runs)
        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([x['Two_Peak_R_interp']], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        print(E.shape, R.shape)
        # print("I defined the network")
        Kern = Kern.astype('float64')
        Kern_R[:, (Kern_R.shape[1]-1)] = 1
        print(torch.cuda.is_available())
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)
        rbfnet = Network(Kern, Kern_R)
        if load==0:
            rbfnet.load_state_dict(torch.load(model_ref))
        rbfnet.to(device)
        rbfnet =  rbfnet.fit(trainloader, testloader_two, testloader_two, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag)
    else:
        print("both peaks")
        save_dir = 'results/sample_uncert/'
        model_ref = 'results/models/modella_uncert'+str(runs)
        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([x['One_Peak_R_interp'], x['Two_Peak_R_interp']], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        print(E.shape, R.shape)
        # print("I defined the network")
        Kern = Kern.astype('float64')
        Kern_R[:, (Kern_R.shape[1]-1)] = 1
        print(torch.cuda.is_available())
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)
        rbfnet = Network(Kern, Kern_R)
        if load==0:
            rbfnet.load_state_dict(torch.load(model_ref))
        rbfnet.to(device)
        rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_two, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag)

    torch.save(rbfnet.state_dict(), model_ref+str(runs))
