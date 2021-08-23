# Originan torch libraries
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



###################################################################
## Placeholder for samba tensors
def get_inputs() -> Tuple[samba.SambaTensor, samba.SambaTensor]:
  	#placeholde
    x = samba.randn(100, 151, 151, name='x_data', batch_dim=0)
    R  = samba.randn(100, 2000, name='y_data', batch_dim=0)  
    E = samba.randn(100, 151, name='E', batch_dim=0)
    tt = samba.randn(100, name='tt', batch_dim=0)
    return x, R, E, tt


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


###################################################################
## The main network
class Network(nn.Module):
    ##################################################    
    def __init__(self, layer_widths, layer_centres, kern_R=0, kern=0):
        super(Network, self).__init__()

        # The definition of the rbf layer
        self.in_features = 151
        self.out_features = 151
        self.centres = nn.Parameter(torch.Tensor(100, self.out_features, self.in_features), requires_grad=True)
        self.sigmas = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
        # self.kern=torch.from_numpy(kern).float()
        self.kern=samba.from_torch(torch.from_numpy(kern).float()) #rcw
        
        self.kern_R=torch.from_numpy(kern_R).float() 
        
        nn.init.normal_(self.centres, 0, 0.01)
        nn.init.constant_(self.sigmas, 2)

        #################################
        # This the nested class object. I have moved all of this to forward 
        # The instantiation should work but when we call the instantiation in forward, it will throw errors.
        self.l1=nn.Linear(layer_widths[1], layer_centres[0])
        self.l2=nn.Linear(layer_centres[0], layer_centres[0]) 
        self.sig=torch.nn.Sigmoid()
        self.e=torch.exp
        self.criterion=nn.MSELoss() 


     
        # Functions to determine integration
        # self.kern_R = torch.from_numpy(kern_R)
        # self.kern = torch.from_numpy(kern).float()
           
    ##########################################
    def forward(self, input_0: torch.Tensor, targets: torch.Tensor, E:torch.Tensor, tt: torch.Tensor) -> Tuple[torch.Tensor]:    

        ##########################################################
        print("I am in the forward loop")         
        print(input_0.shape, targets.shape, E.shape, tt.shape)
        print("targets",targets)
        print("inut", input_0)
        print("E",E)
        print("tt", tt)
        print("input", inputs)
        ################################################ 
        input_0 = torch.abs(input_0)
        targets=torch.abs(targets)
        E= torch.abs(E)
        tt= torch.abs(tt)
        ##################################################

  
        ######################################################
        # The Gaussian Layer
        c =self.centres 
        #rcwc = samba.from_torch(c) #rcw
        x=input_0
        c.batch_dim=x.batch_dim #rcw
        # print("c batch_dim",c.batch_dim)
        ## My workaround 
        t = torch.add(x, -1*c)
        t = torch.mul(t, 1*t)
        t = torch.sum(t, dim=2)
        t = torch.sqrt(t)
        # print(t, t.shape)
        # print("t.batch_dim", t.batch_dim)
        t.batch_dim=x.batch_dim
        distances = t*self.sigmas.unsqueeze(0)
        # print(distances)
        # print("distances.batch_dim", distances.batch_dim)
        out= self.e(-0.001*torch.mul(distances, 1*distances))
        # print("out.batch_dim", out.batch_dim)
        # print(distances.shape)
        # print(out)
        # ircwE_er=samba.to_torch(E_er)
        # loss_E = torch.mean(torch.mul(E_er, 1*E_er)*(1/(0.0001*0.0001)))
        
        # The Neural Network 
        out = self.sig(self.l1(out))
        # print(out)
        # Go to torch to do the integration as it is easy
        #rcw Rhat = samba.to_torch(self.e(self.l2(out)))
        Rhat = self.e(self.l2(out))
        
        # print(Rhat)
        # print("Rhat.batch_dim=",Rhat.batch_dim)
        #rcwRhat = samba.to_torch(Rhat) #rcw
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)
        # print(Ehat)
        

        #rcw Ehat = samba.to_torch(Ehat) #rcw
        #rcwcorrectin_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)#rcw
        tmp = Ehat[:, 0].view(-1, 1)
        correction_term = tmp.expand(100,2000)
        Rhat = torch.div(Rhat, correction_term)

        # print("corrected Rhat", Rhat)
        # print(Rhat.shape, self.kern.shape)
        # x=input("Hola d tados")
        multout = torch.matmul(self.kern, Rhat.transpose(0, 1))
        Ehat = multout.transpose(0, 1)
        
        
        
        # print("corrected EHAT", Ehat)
        # Go back to Sambanova
        #rcw Ehat= samba.from_torch(Ehat)
        #rcw Rhat=samba.from_torch(Rhat)
        # print(Rhat)
        # print("Targets", targets)
        # Calculate the losses
        # The Entropy 
        non_integrated_entropy = (targets-Rhat-torch.mul(targets, samba.log(samba.div(targets, Rhat))))
        #rcwnon_integrated_entropy.batch_dim=x.batch_dim
        #rcwprint("non_integrated_entropy.batch_dim=",non_integrated_entropy.batch_dim)
        #rcwnon_integrated_entropy=samba.to_torch(non_integrated_entropy) #rcw
        loss_R = -1*torch.mean(non_integrated_entropy)


        # loss_R=0.0
        # loss_R = self.criterion(Rhat, targets)
        # rcwE_er=samba.to_torch(E_er)
        # loss_R = torch.mean(torch.mul(R_er,  1*R_er))
    
        # The E's loss
        E_er=(Ehat-E)
        #rcwE_er=samba.to_torch(E_er)
        loss_E = torch.mean(torch.mul(E_er, 1*E_er)*(1/(0.0001*0.0001)))
        # The uq loss
        #rcwdiff = samba.to_torch(Rhat - targets)
        diff = (Rhat - targets)
        tt=samba.to_torch(tt)
        diff=samba.to_torch(diff)
        mask = (diff.ge(0).float() - tt.repeat(2000).view(-1, 2000)).detach()
        #rcwloss_uq_R = samba.from_torch( (mask * diff).mean()) 
        loss_uq_R = (mask * diff).mean() 


        # Add to get the total loss.
        # Total Loss
        LOSS = torch.mean(1e-6*loss_E+loss_R)
        #print("The loss is", LOSS)
        # print("The loss for E is", loss_E)
        # print("The loss for R is", loss_R)
        # print("The uq loss is", loss_uq_R)
        # LOSS = self.criterion(Rhat,targets)
        #  print("Rhat", Rhat)
        # print("Ehat", Ehat)
        # x = input("THis is a sampele R and an E") 

        

        return LOSS, Rhat, Ehat
    

##################################################
def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('-c', '--num-classes', type=int, default=151)
    parser.add_argument('-e', '--num-epochs', type=int, default=100)
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
    # print(tx)
    # print(ty)
    trainset = MyDataset(x, y)
    # print(x.shape, y.shape, batch_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epoch = 0
    while epoch < epochs:
        epoch += 1
        for x_batch, y_batch in trainloader:
           y_batch = y_batch.float()
           x_batch = x_batch.float()
           
           #print("x", x_batch, "y", x_batch)
           x = torch.repeat_interleave(x_batch.unsqueeze(1), 151, dim=1)
           # x = x_batch    
           ##########################################
           # The following code is used when we start with the UQ part
           ##########################################
           
           # print(x)
           tt = torch.rand(x_batch.shape[0], 1)


           # print(x_batch.shape, y_batch.shape)
           s_x = samba.from_torch(x, name ='x_data', batch_dim = 0).float() #rcwi
           E   = samba.from_torch(x_batch, name ='E', batch_dim = 0).float() #rcw
           s_y = samba.from_torch(y_batch, name ='y_data', batch_dim = 0).float() #rcw
           s_tt  = samba.from_torch(tt, name ='tt', batch_dim = 0).float() #rcw
           
           # print(s_x)
           #print("in torch", y_batch)
           # print("in sambanova", s_y)
           # print(s_x.shape, s_y.shape)
           loss, outputs = samba.session.run(input_tensors=[s_x, s_y, E, s_tt],output_tensors= model.output_tensors)
           sys.stdout.write('\rEpoch: %d, Loss:%f' %(epoch, samba.to_torch(loss).mean()))




#########################################################################################

# *** The following is for the full dataset

def main(argv: List[str]):
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
    # The following code extracts data when the full dataset is used.

    # x = return_dict('../../inverse_data_interpolated_numpy.p')
    # print(x.keys())
    # R = x['One_Peak_R_interp']
    # E = x['One_Peak_E']
  
    # index = np.random.randint(0, R.shape[0], 1000)
    # The shape of the training data
    # print("Training data", R.shape, E.shape)
    # R = R[index]
    # E = E[index]
 

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


    # Here I will put the code for generating the integration coefficients.
    E = np.loadtxt("E.csv", delimiter=',')
    R = np.loadtxt("R.csv", delimiter=',')
    
    # print(E, R) 
    omega_ = np.loadtxt("omega.csv", delimiter=',').reshape([-1,1])
    omega_fine = np.loadtxt("omega_fine.csv", delimiter=',').reshape([-1,1])
    tau = np.loadtxt("tau.csv", delimiter=',').reshape([-1,1])
    
    _, _, Kern, Kern_R = integrate(tau, omega_, omega_fine, R)
    Kern_R[:, (Kern_R.shape[1]-1)] = 1
    tx = torch.from_numpy(E)
    ty = torch.from_numpy(R)
    # print("E", tx)
    # print("target", ty)
    # print(tx.shape, ty.shape, Kern.shape, Kern_R.shape)
    # To add more layers, change the layer_widths and layer_centres lists
    layer_widths  = [151, 151]
    layer_centres = [2000, 2000]
    inputs = get_inputs()
    # print("inputs=", inputs)
    model = Network(layer_widths, layer_centres, Kern_R, Kern)
    #rcw just torch
    #rcwLOSS, Rhat, Ehat = model(inputs[0], inputs[1], inputs[2], inputs[3]) #rcw
    #rcwprint("Just torch")
    #rcwprint("LOSS=", LOSS)
    #rcwprint("Rhat=", Rhat)
    #rcwprint("Ehat=", Ehat)
    #rcwexit()
    samba.from_torch_(model)
    model.float()
    optim = sambaflow.samba.optim.SGD(model.parameters(),
                                        lr=0.0001,
                                        momentum=0.1,
                                        weight_decay=0.01)

    #rcwinputs = get_inputs()
    if args.command == "compile" or args.command == "measure-cpu" or args.command == "measure-performance":
       #rcwcommon_app_driver(args, model, inputs, optim, name='inverse', app_dir=utils.get_file_dir(__file__)) 
       # train(model, optim, tx, ty, 100, 100, 0.0001)    
       
       common_app_driver(args=args,
                       model=model,
                       inputs=inputs,
                       name='inverse',
                       optim=optim,
                       squeeze_bs_dim=False, get_output_grads=False, app_dir=utils.get_file_dir(__file__))
       
    elif args.command == "test":
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
