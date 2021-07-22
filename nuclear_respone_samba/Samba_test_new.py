# Original torch libraries
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from typing import List, Tuple
import time

# Sambaflow routines
import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.utils as utils
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.utils.dataset.mnist import dataset_transform
from sambaflow.samba.utils.trainer.samba import train as samba_train
from sambaflow.samba.utils.trainer.torch import train as torch_train



## Placeholder for samba tensors
def get_inputs() -> Tuple[samba.SambaTensor, samba.SambaTensor]:
  	#placeholde
    image = samba.randn(100, 151, name='x_data', batch_dim=0)
    label = samba.randn(100, 2000, name='y_data', batch_dim=0)
    return image, label


############################################################
# Integrator
# First define a function that finds the nearest neighbors
def x_near(x, x_v):
    x_d = np.abs(x_v - x)
    x_index = np.argsort(x_d)
    return x_index


############################################################
# Integrator
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



############################################################
# Integrator
# Actual interpolation function
def interp_quadratic(x, x_v, y_v, poly, x_index):
    y = np.zeros(x.shape[0])
    for i in range (x.shape[0]):
        y[i] = np.sum( poly[i, 0:3] * y_v[x_index[i, 0:3]])
    return y

############################################################
def interpolate_Alessandro(omega_fine, omega_, R_):
    omega_fine = omega_fine.reshape([-1])
    R_temp = np.zeros([R_.shape[0],omega_fine.shape[0]])
    poly, x_index = polint_quadratic(omega_fine.reshape([-1]), omega_.reshape([-1]))
    print("I am starting the main interpolation loop")
    for i in range(R_.shape[0]):
        R_temp[i,:] = interp_quadratic(omega_fine.reshape([-1]),\
         omega_.reshape([-1]), R_[i,:].reshape([-1]), poly, x_index)
    return R_temp, poly, x_index




############################################################
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

#####################################################################
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


###################################################################
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


################################################################
# RBF Layer
# This is the nested class that I want as an object in the original class.
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

    def __init__(self, in_features: int, out_features: int):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        # self.reset_parameters()
        # def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 0.01)
        nn.init.constant_(self.sigmas, 2)

    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = (inputs.size(0), self.out_features, self.in_features)
        x = inputs.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        
        ## Source of Error #1
        ## Original line of code (throws an error on pow function)
        # Would be a much more efficient inline operation if pow function works
        distances=(x-c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        
        ## My workaround 
        # t = torch.add(x, -1*c)
        # t = torch.mul(t, t)
        # t = t.sum(-1)
        # t = torch.sqrt(t)
        # distances = t*self.sigmas.unsqueeze(0)
        return torch.exp(-1*torch.mul(distances, distances))
        
    # return self.gaussian(distances)

    def gaussian(self,alpha: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(-1*torch.mul(alpha, alpha))
        return phi


##################################################
# Torch dataset and dataloader
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


## The main network
class Network(nn.Module):
    ##################################################    
    def __init__(self, layer_widths, layer_centres, kern_R=0, kern=0):
        super(Network, self).__init__()

        i=0
        #################################
        # This the nested class object. 
        # The instantiation should work but when we call the instantiation in forward, it will throw errors.
        self.rbf1=RBF(layer_widths[i], layer_widths[i+1])    
        self.l1=nn.Linear(layer_widths[i+1], layer_centres[i])
        self.l2=nn.Linear(layer_centres[i], layer_centres[i]) 
        self.sig=torch.sigmoid
        self.e=torch.exp
        self.criterion=nn.MSELoss()

        # Functions to determine integration
        # self.kern_R = torch.from_numpy(kern_R)
        # self.kern = torch.from_numpy(kern).float()
                
    ##########################################
    def forward(self,inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor]:    
        out = inputs 
        ##########################################
        ## Error Source no. 2
        # uncommenting the following line would throw an error
        out = self.rbf1(out)
        # Without the above the line things would work
        out = self.sig(self.l1(out))
        Rhat = self.e(self.l2(out))

        ##########################################
        # We have convert the follwing pytorch code to sambai
        # Error Source no. 3.
        ##########################################
        # print(out.shape, self.kern.shape)
        # Ehat = torch.matmul(self.kern, out.transpose(0, 1)).transpose(0, 1)
        # correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)
        ##########################################
        # print(correction_term.shape, Ehat.shape, out.shape)
        # Rhat = torch.div(out, correction_term)
        # Rhat = out
        # print(out.shape, self.kern.shape)
        ##########################################
        # multout = torch.matmul(self.kern, Rhat.transpose(0, 1))
        # print(multout.shape)
        # Ehat = multout.transpose(0, 1) 

        # The Entropy
        # non_integrated_entropy = (Rhat-R-torch.multiply(Rhat, torch.log(torch.div(Rhat, R))))
        # loss_R = -1*torch.mean(torch.multiply(self.kern_R,non_integrated_entropy))

        # The uq loss
        # tt = inputs[:,151]
        # diff = Rhat - targets
        # mask = (diff.ge(0).float() - tt.repeat(2000).view(-1, 2000)).detach()
        # loss_uq_R = (mask * diff).mean()

        # diff = Ehat - inputs[:,0:151]
        # mask = (diff.ge(0).float() - tt.repeat(151).view(-1, 151)).detach()
        # loss_uq_E = (mask * diff).mean()


        # The E's loss
        # loss_E = torch.mean(torch.square(Ehat - E[:, :151])*(1/(0.0001*0.0001)))

        # Total Loss
        # LOSS = torch.mean(factor_E*loss_E + factor_R*loss_R  + factor_R_uq*loss_uq_R)
        

        # The loss current, we want the previous one though
        LOSS = self.criterion(Rhat,targets)
        return LOSS, Rhat
    



##################################################
def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('-c', '--num-classes', type=int, default=10)
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('-k', '--num-features', type=int, default=784)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in CH regression.')



##################################################
def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--log-path', type=str, default='checkpoints')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--dump-interval', type=int, default=1)
    parser.add_argument('--gen-reference', action='store_true', help="Generate PyTorch reference data")
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('-n', '--num-iterations', type=int, default=100, help='Number of iterations to run the pef for')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--measure-train-performance', action='store_true')


##################################################
def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")

##################################################
# Main train loop 
def train(model, optimizer: samba.optim.SGD, x: torch.Tensor, y:  torch.Tensor, epochs:int, batch_size:int, lr:float) -> None:
    trainset = MyDataset(x, y)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epoch = 0
    while epoch < epochs:
        epoch += 1
        for x_batch, y_batch in trainloader:
           y_batch = y_batch.float()
           
           ##########################################
           # The following code is used when we start with the UQ part
           ##########################################
           # tt = torch.rand(x_batch.shape[0], 1)
           # print(x_batch.shape, tau.shape)
           # x_batch = torch.cat((x_batch, tt), 1)
           # print(x_batch.shape, tau.shape)           
           

           # Samba oriented code
           s_x = samba.from_torch(x_batch, name ='x_data', batch_dim = 0).float() #rcw
           s_y = samba.from_torch(y_batch, name ='y_data', batch_dim = 0).float() #rcw
           loss, outputs = samba.session.run(input_tensors=[s_x, s_y],output_tensors= model.output_tensors)
           sys.stdout.write('\rEpoch: %d, Loss:%f' %(epoch, samba.to_torch(loss).mean()))




#########################################################################################
'''
*** The following is for the full dataset

def main(argv: List[str]):
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
# The following code extracts data when the full dataset is used.

  x = return_dict('../inverse_data_interpolated_numpy.p')
  print(x.keys())
  R = x['One_Peak_R_interp']
  E = x['One_Peak_E']
  

    if params['peak'] == 'one_peak':
        print("I started with one peak data")
        R = x['One_Peak_R_interp']
        E = x['One_Peak_E']
    elif params['peak'] == 'two_peak':
        print("I went with the two peak data")
        R = x['Two_Peak_R_interp']
        E = x['Two_Peak_E']
    elif params['peak'] == 'one_two_peak':
        print("I went with the two peak data")
        R = x['One_Peak_R_interp']
        E = x['One_Peak_E']
    elif params['peak'] == 'two_one_peak':
        print("I went with the two peak data")
        R = x['Two_Peak_R_interp']
        E = x['Two_Peak_E']
    elif params['peak'] == 'both
        R = np.concatenate([x['One_Peak_R_interp'],
                            x['Two_Peak_R_interp']
                            ], axis=0 )
        E = np.concatenate([x['One_Peak_E'],
                            x['Two_Peak_E']
                            ], axis=0 )
  
    # Extract the training data
    # R = R[0:R.shape[0]-1000, :]
    # E = E[0:E.shape[0]-1000, :]
    # E_test = E[E.shape[0]-1000:, :]
    # R_test = R[R.shape[0]-1000:, :]
    # Select indexes according to the number of datapoints
  
  
  index = np.random.randint(0, R.shape[0], 1000000)
  # The shape of the training data
  print("Training data", R.shape, E.shape)
  R = R[index]
  E = E[index]
  
  # The following code would be uncommented if the integration operations work
  # tau = x['tau'] 
  # omega_ = x['omega']
  # omega_fine = x['omega_fine']
  # E, R, Kern, Kern_R = integrate(tau, omega_, omega_fine, R)
  ###################################################
  # Kern_R[:, (Kern_R.shape[1]-1)] = 1
  ################################################### 
  
  
  E = E.astype('float64')
  R = R.astype('float64')
    
    # print(Kern_R.shape)
    Kern = Kern.astype('float64')
    Kern_R[:, (Kern_R.shape[1]-1)] = 1
    Kern_R = np.repeat(Kern_R.astype('float64'), Config['batch_size'], axis=0)
    print(Kern_R.shape, E.shape, R.shape)
    x = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/Test_MEM_data.p')
    print(x.keys())

    if params['peak'] == 'two_peak':
        # print("I started with one peak data")
        R_test = x['Two_Peak_R']
        E_test = x['Two_Peak_E']
    elif params['peak'] == 'one_peak':
        R_test = x['One_Peak_R']
        E_test = x['One_Peak_E']
    elif params['peak'] == 'one_two_peak':
        R_test = x['Two_Peak_R']
        E_test = x['Two_Peak_E']
    elif params['peak'] == 'two_one_peak':
        R_test = x['One_Peak_R']
        E_test = x['One_Peak_E']
    elif params['peak'] == 'both':
        R_test = np.concatenate([x['One_Peak_R'], x['Two_Peak_R']], axis=0)
        E_test = np.concatenate([x['One_Peak_E'], x['Two_Peak_E']], axis=0)

    tau = x['tau']
    omega_fine = x['omega_fine']
    omega = x['omega_coarse']
    # The final test data results
    Ehat = np.zeros([0, 151])
    Rhat = np.zeros([0, 2000])
    E_t = np.zeros([0, 151])
    R_t = np.zeros([0, 2000])

*** The above is for the full dataset
  '''
##########################################################################
 
def main(argv: List[str]):
  args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
  # I created a smaller dataset with just 1000 points for the purpose of analysis
  tx = np.loadtxt('E.csv', delimiter = ',')
  ty = np.loadtxt('R.csv', delimiter = ',')
  tx = torch.from_numpy(tx)
  ty = torch.from_numpy(ty)
  print(tx.shape, ty.shape)
  # To add more layers, change the layer_widths and layer_centres lists
  layer_widths  = [151, 151]
  layer_centres = [2000, 2000]
  samples = 100
  model = Network(layer_widths, layer_centres)
  samba.from_torch_(model)
  model.bfloat16()
  optim = sambaflow.samba.optim.SGD(model.parameters(),
                                        lr=0.0001,
                                        momentum=0.1,
                                        weight_decay=0.01)

  inputs = get_inputs()
  common_app_driver(args, model, inputs, optim, name='inverse', app_dir=utils.get_file_dir(__file__)) 
  if args.command == "test":
      utils.trace_graph(model, inputs, optim, pef='out/inverse/inverse.pef')
      # utils.trace_graph(model, inputs, optim, config_dict=vars(args))
      outputs = model.output_tensors
      #run_test( model, inputs, outputs)
  elif args.command == "run":
    print (args)
    #rcw utils.trace_graph(model, inputs, optim, config_dict=vars(args))
    utils.trace_graph(model, inputs, optim, pef='out/inverse/inverse.pef')
    train(model, optim, tx, ty, 100, 100, 0.0001)    
  
  
  

if __name__ == '__main__':
    print (sambaflow.__version__)
    

    start_time = time.time()
    print(time.time())
    main(sys.argv[1:])
    print("/n The time taken for training is--", time.time()-start_time)
