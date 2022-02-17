import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# RBF Layer
class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 0.01)
        nn.init.constant_(self.sigmas, 2)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        # print(size)
        x = input.unsqueeze(1).expand(size)
        # print(x.shape)
        c = self.centres.unsqueeze(0).expand(size)
        # print(c.shape)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.basis_func(distances)

# different radial basis kernels.


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(
        alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3)
           * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases


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


class Network(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func, kern_R, kern):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.kern_R = torch.from_numpy(kern_R)
        self.kern = torch.from_numpy(kern).float()
        # for i in range(len(layer_widths) - 1):
        #print(layer_widths[i], layer_centres[i])
        i = 0
        self.rbf_layers.append(
            RBF(layer_widths[i], layer_widths[i+1], basis_func))
        self.linear_layers.append(
            nn.Linear(layer_widths[i+1], layer_centres[i]))
        self.linear_layers.append(
            nn.Linear(layer_centres[i], layer_centres[i]))

    ##########################################
    def forward(self, x):
        out = x.float()
        ##########################################
        # for i in range(len(self.rbf_layers)):
        i = 0
        out = self.rbf_layers[i](out)
        out = torch.relu(self.linear_layers[i](out))
        i = 1
        out = torch.exp(self.linear_layers[i](out))
        ##########################################
        # We have convert this code to pytorch
        # print(out.shape, self.kern.shape)
        Ehat = torch.matmul(self.kern, out.transpose(0, 1)).transpose(0, 1)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)
        ##########################################
        # print(correction_term.shape, Ehat.shape, out.shape)
        Rhat = torch.div(out, correction_term)
        # print(out.shape, self.kern.shape)
        ##########################################
        multout = torch.matmul(self.kern, Rhat.transpose(0, 1))
        # print(multout.shape)
        Ehat = multout.transpose(0, 1)
        return Rhat, Ehat

    ##########################################
    def loss_func(self, Rhat, R, Ehat, E, factor_E, factor_R, factor_R_uq, factor_E_uq):
        # Rewrite this in pytorch
        # Entropy driven Loss function

        # The Entropy
        non_integrated_entropy = (Rhat-R-torch.multiply(Rhat, torch.log(torch.div(Rhat, R))))
        loss_R = -1*torch.mean(torch.multiply(self.kern_R,non_integrated_entropy))

        # The uq loss
        tt = E[:,151]
        diff = Rhat - R
        mask = (diff.ge(0).float() - tt.repeat(2000).view(-1, 2000)).detach()
        loss_uq_R = (mask * diff).mean()
        
        diff = Ehat - E[:,0:151]
        mask = (diff.ge(0).float() - tt.repeat(151).view(-1, 151)).detach()
        loss_uq_E = (mask * diff).mean()


        # The E's loss
        loss_E = torch.mean(torch.square(Ehat - E[:, :151])*(1/(0.0001*0.0001)))

        # Total Loss
        LOSS = torch.mean(factor_E*loss_E + factor_R*loss_R  + factor_R_uq*loss_uq_R)

        return LOSS, loss_E, loss_R, loss_uq_R, loss_uq_E

    ##########################################
    def fit(self, x, y, x_test, y_test, epochs,
            batch_size, lr, tau, omega_fine, Config):
        self.train()
        obs = x.shape[0]
        LR = []
        LE = []
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = MyDataset(x_test, y_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        optimiser = torch.optim.RMSprop(self.parameters(),\
        lr=lr, weight_decay=0.001)
        ##########################################
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            current_loss_R = 0
            current_loss_E = 0
            current_loss_uq_R =0
            current_loss_uq_E =0
            progress = 0

            ##########################################
            # Make Hyper Parameter decisions
            if (epoch % 200 == 0):
                optimiser.param_groups[0]['lr'] = optimiser.param_groups[0]['lr']*0.95
            elif epoch % 500 == 0:
                optimiser.param_groups[0]['lr'] = optimiser.param_groups[0]['lr']*0.99

            ##########################################
            if epoch < int(round(Config['n_iterations']*0.125)):
                factor_E = 1e-10
                factor_R = 1e6
                factor_R_uq = 1e6
                factor_E_uq = 0
            elif epoch < int(round(Config['n_iterations']*0.25)):
                factor_E = 1e-8
                factor_R = 1e6
                factor_R_uq = 1e6
                factor_E_uq = 0
            elif epoch < int(round(Config['n_iterations']*0.5)):
                factor_E = 1e-6
                factor_R = 1e7
                factor_R_uq = 1e7
                factor_E_uq = 0
            elif epoch < int(round(Config['n_iterations']*0.66)):
                factor_E = 1e-4
                factor_R = 1e8
                factor_R_uq = 1e8
                factor_E_uq = 0
            elif epoch < int(round(Config['n_iterations']*0.75)):
                factor_E = 1e-2
                factor_R = 1e8
                factor_R_uq = 1e8
                factor_E_uq = 0
            else:
                factor_E = 1
                factor_R = 1e8
                factor_R_uq = 1e8
                factor_E_uq = 0


            ##########################################
            for x_batch, y_batch in testloader:
                batches += 1
                optimiser.zero_grad()

                ##########################################
                tt = torch.rand(x_batch.shape[0], 1)
                # print(x_batch.shape, tau.shape)
                x_batch = torch.cat((x_batch, tt), 1)
                # print(x_batch.shape, tau.shape)

                ##########################################
                y_batch = y_batch.float()+torch.normal(0,1e-30, y_batch.shape)
                y_hat, x_hat = self.forward(x_batch)
                loss, los_E, los_R, los_uq_R, los_uq_E = \
                    self.loss_func(y_hat, y_batch,
                                   x_hat, x_batch, factor_E, factor_R, factor_R_uq, factor_E_uq)


                # print(loss, los_E, los_R)
                current_loss    += (1/batches) * (loss.item() - current_loss)
                current_loss_R  += (1/batches) * (los_R.item() - current_loss_R)
                current_loss_E  += (1/batches) * (los_E.item() - current_loss_E)
                
                current_loss_uq_R += (1/batches) * \
                    (los_uq_R.item() - current_loss_uq_R)
                current_loss_uq_E += (1/batches) * \
                    (los_uq_E.item() - current_loss_uq_E)

                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d,\
                Loss: %f %f %f %f %f' %(epoch, progress, obs, current_loss,\
                current_loss_R, current_loss_E, current_loss_uq_R, current_loss_uq_E))
                sys.stdout.flush()

                
            print("\n######################################")
            LR.append(current_loss_R)
            LE.append(current_loss_E)
            E_hat_up = np.zeros([0, 151])
            R_hat_up = np.zeros([0, 2000])
            E_t = np.zeros([0, 151])
            R_t = np.zeros([0, 2000])
            E_hat_down = np.zeros([0, 151])
            R_hat_down = np.zeros([0, 2000])

            E_hat_m = np.zeros([0, 151])
            R_hat_m = np.zeros([0, 2000])

            ##########################################
            if epoch % Config['print_it'] == 0:
                alpha = 0.1
                for x_batch, y_batch in trainloader:
                    y_batch = y_batch.float()
                    
                    E_t = np.concatenate(
                        [E_t, x_batch.detach().numpy()], axis=0)
                    R_t = np.concatenate(
                        [R_t, y_batch.detach().numpy()], axis=0)
                    tt = torch.zeros(x_batch.size(0), 1)+(alpha/2)
                    # print(x_batch.shape, tau.shape)
                    x_batch_up = torch.cat((x_batch, (tt)), 1)
                    # print(x_batch.shape, tau.shape)
                    y_hat_up, x_hat_up = self.forward(x_batch_up)
                    
                    # print(y_hat.shape,y_batch.shape)
                    R_hat_up = np.concatenate(
                        [R_hat_up, y_hat_up.detach().numpy()], axis=0)
                    E_hat_up = np.concatenate(
                        [E_hat_up, x_hat_up.detach().numpy()[:, :151]], axis=0)

                    tt = torch.zeros(x_batch.size(0), 1)+(1-alpha/2)
                    # print(x_batch.shape, tau.shape)
                    x_batch_down = torch.cat((x_batch, (tt)), 1)
                    # print(x_batch.shape, tau.shape)
                    y_hat_down, x_hat_down = self.forward(x_batch_down)
                    # print(y_hat.shape,y_batch.shape)
                    # print(x_hat.shape, x_batch.shape)
                    R_hat_down = np.concatenate(
                        [R_hat_up, y_hat_down.detach().numpy()],\
                        axis=0)
                    E_hat_down = np.concatenate(\
                        [E_hat_up, x_hat_down.detach().numpy()[:, :151]],\
                        axis=0)
                    
                    # tt = torch.zeros(x_batch.size(0), 1)+0
                    # # print(x_batch.shape, tau.shape)
                    # x_batch_m = torch.cat((x_batch, (tt)), 1)
                    # # print(x_batch.shape, tau.shape)
                    # y_hat_m, x_hat_m = self.forward(x_batch_m)
                    # # print(y_hat.shape,y_batch.shape)
                    # # print(x_hat.shape, x_batch.shape)
                    # R_hat_m = np.concatenate([R_hat_m, y_hat_m.detach().numpy()], axis=0)
                    # E_hat_m = np.concatenate([E_hat_m, x_hat_m.detach().numpy()[:, :151]], axis=0)





                def chi2_vec(y, yhat, factor):
                    return np.mean(((y-yhat)**2/factor**2), axis=1)
                chi2 = chi2_vec(x_batch.detach().numpy(),
                                x_hat_up.detach().numpy(), 0.0001)
                # Let us create some plots with these
                import matplotlib.pyplot as plt

                def cm2inch(value):
                    return value/2.54
                small = 7
                med = 10
                large = 12
                plt.style.use('seaborn-white')
                COLOR = 'darkslategray'
                params = {
                    'axes.titlesize': small,
                    'legend.fontsize': small,
                    'figure.figsize': (cm2inch(15), cm2inch(8)),
                    'axes.labelsize': med,
                    'axes.titlesize': small,
                    'xtick.labelsize': small,
                    'ytick.labelsize': med,
                    'figure.titlesize': small,
                    'font.family': "sans-serif",
                    'font.sans-serif': "Myriad Hebrew",
                    'text.color': COLOR,
                    'axes.labelcolor': COLOR,
                    'axes.linewidth': 0.3,
                    'xtick.color': COLOR,
                    'ytick.color': COLOR
                }

                plt.rcParams.update(params)
                plt.rc('text', usetex=False)
                color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                plt.rcParams['mathtext.fontset'] = 'cm'
                from matplotlib.lines import Line2D
                import matplotlib.font_manager as font_manager
                large = 24
                med = 8
                small = 7
                labels = ['Old', 'New', 'Original']
                titles = ['worst', 'median', 'best']

                for j, element in enumerate([40, 30]):
                    # create plots with numpy array
                    fig, a = plt.subplots(2, 1, sharex=False, dpi=600,
                        gridspec_kw={'wspace': 0.7, 'hspace': 0.7})

                    # The final Plots with CME
                    temp_omega = omega_fine[:].reshape([-1])

                    # temp = np.concatenate([y_hat_up.detach().numpy()[element,   :],\
                    #                        y_hat_down.detach().numpy()[element, :]],\
                    #                     axis = 0)
                    # R_up = np.max(temp, axis = 0)
                    # R_down = np.min(temp, axis = 0)
                    # a[0].errorbar(temp_omega, mean_R,
                    #               yerr=yerr,
                    #               fmt='x',
                    #               color=color[0], ms=0.2,
                    #               linewidth=0.1, label=labels[0])

                    R_up   = y_hat_up.detach().numpy()[element,   :]
                    R_down = y_hat_down.detach().numpy()[element, :]
                    org_R = y_batch.detach().numpy()[element, :]
                    yerr = (R_up - R_down)
 
                    # fill_up = y+yerr
                    # fill_down = y-yerr
                    # fill_down[fill_down<0]= 0
                    # median_R = y_hat_m.detach().numpy()[element, :]
                    # y = median_R

                    a[0].fill_between(temp_omega, R_up, R_down,
                                      color=color[0], alpha=0.5)
                    a[0].plot(temp_omega, org_R, color=color[1],
                              linewidth=0.5, linestyle='--', label=labels[2])
                    mean_R = (R_up+R_down)/2
                    a[0].errorbar(temp_omega, mean_R,
                                  yerr=yerr,
                                  fmt='x',
                                  color=color[0], ms=0.2,
                                  linewidth=0.1, label=labels[0])

                    # a[0].errorbar(temp_omega, median_R,
                    #               yerr=yerr,
                    #               fmt='x',
                    #               color=color[2], ms=0.2,
                    #               linewidth=0.1, label=labels[0])
                    # The final Plots with CME
                    # temp = np.concatenate([x_hat_up.detach().numpy()[element,   :],\
                    #                        x_hat_down.detach().numpy()[element, :]], axis = 0)
                    # E_up = np.max(temp, axis = 0)
                    # E_down = np.min(temp, axis = 0)




                    temp_tau = tau[:].reshape([-1])
                    E_up = x_hat_up.detach().numpy()[element, :]
                    E_down = x_hat_up.detach().numpy()[element, :]
                    org_E = x_batch.detach().numpy()[element, :]
                    yerr = (E_up - E_down)

                    # median_E = x_hat_m.detach().numpy()[element, :]
                    # y = median_E
                    fill_up = E_up+yerr
                    fill_down = E_down-yerr
                    fill_down[fill_down<0]= 0


                    a[1].fill_between(temp_tau, fill_up, fill_down,
                                      color=color[0], alpha=0.5)
                    a[1].errorbar(temp_tau, org_E, yerr=0, fmt='o',
                                  color=color[1], ms=0.2,
                                  linewidth=0.5, label=labels[2])
                    mean_E = (E_up+E_down)/2
                    a[1].errorbar(temp_tau, mean_E,
                                  yerr=yerr, fmt='x',
                                  color=color[0], ms=0.2,
                                  linewidth=0.1, label=labels[0])


                    # a[1].errorbar(temp_tau, median_E,
                    #               yerr=yerr, fmt='x',
                    #               color=color[2], ms=0.2,
                    #               linewidth=0.1, label=labels[0])

                    # Some Plot oriented settings
                    a[0].spines["top"].set_visible(False)
                    a[0].spines["bottom"].set_visible(False)
                    a[0].spines["right"].set_visible(False)
                    a[0].spines["left"].set_visible(True)
                    a[0].grid(linestyle=':', linewidth=0.5)
                    a[0].get_xaxis().tick_bottom()
                    a[0].get_yaxis().tick_left()
                    a[0].set_xlim([0, 500])
                    # a[0].set_yscale('log')
                    a[0].set_xlabel('$\omega~[MeV]$')
                    a[0].set_ylabel('$R(\omega)~[MeV^{-1}]$')
                    a[0].legend(bbox_to_anchor=(0.0008, -0.5, 0.3, 0.1),
                                loc='upper left', ncol=3)

                    a[1].set_yscale('log')
                    a[1].set_title(chi2[element])
                    a[1].set_xlabel('$\\tau~[MeV^{-1}]$')
                    a[1].set_ylabel('$E(\\tau)$')
                    a[1].legend(bbox_to_anchor=(0.0008, -0.5, 0.3, 0.1),
                                loc='upper left', ncol=3)
                    # Some Plot oriented settings
                    a[1].spines["top"].set_visible(False)
                    a[1].spines["bottom"].set_visible(False)
                    a[1].spines["right"].set_visible(False)
                    a[1].spines["left"].set_visible(True)
                    a[1].grid(linestyle=':', linewidth=0.5)
                    a[1].get_xaxis().tick_bottom()
                    a[1].get_yaxis().tick_left()
                    a[1].set_yscale('log')

                    plt.savefig(
                        Config['output_model']+"reconstruction" +
                        str(element)+"epoch"+str(epoch)+".png",
                        dpi=600)
                    plt.close()

        return E_hat_up, R_hat_up, E_hat_down, R_hat_down, E_t, R_t, LR, LE


#                     ##########################################
#                     plt.clf()
# #                     plt.plot(omega_fine, y_hat_up.detach().numpy()[0,:])
# #                     plt.plot(omega_fine, y_hat_down.detach().numpy()[0,:])
#                     plt.fill_between(omega_fine.reshape([-1]), y_hat_up.detach().numpy()[0,:], y_hat_down.detach().numpy()[0,:], color = 'grey', alpha = 0.5)
#                     plt.plot(omega_fine, y_batch.detach().numpy()[0,:])
#                     plt.show()


# #                     ##########################################
#                     plt.clf()
#                     plt.plot(tau, x_hat.detach().numpy()[0,:])
#                     plt.plot(tau, x_batch.detach().numpy()[0,0:151])
#                     plt.show()
#                     break
