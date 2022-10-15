import torch
from Lib import *
import argparse
import json
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


#################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '--runs',
    default=None,
    help="number of runs")

parser.add_argument(
    '--load',
    default=None,
    help="load model or not")

parser.add_argument(
    '--flag',
    default='false',
    help="one peak, two peak, uncertainty, original")

parser.add_argument(
    '--method',
    default='UQ',
    help="one peak, two peak, uncertainty, original")


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join('../torch_attention_model_v10/params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict

    ##############################################
    if args.runs is not None:
        params['n_runs'] = int(args.runs)

    if args.load is not None:
        params['load'] = int(args.load)

    if args.flag is not None:
        params['flag'] = int(args.flag)

    if args.method is not None:
        params['method'] = str(args.method)

    if params["flag"] ==3:
        params['fac_var']=-1

    print("The config files are", params)
    n_runs=params['n_runs']
    load=params['load']
    flag=params['flag']
    batche=params['batche']
    method =params['method']
    epochs=params['epochs']
    learning_rate=params['learning_rate']
    n_data=params['n_points']


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
        
        if method == 'PRC':
            print(method)
            if flag==0:
                print("one peaks")
                save_dir = 'results/sample_one_peak_PRC/'
                model_ref = 'results/models/modella_uncert_one_PRC_'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data]], axis=0)
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
                rbfnet = NetworkPRC(Kern, Kern_R)
                if load==0:
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_one, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            
            elif flag==1:
                
                print("two peaks")
                save_dir = 'results/sample_two_peak_PRC/'
                model_ref = 'results/models/modella_uncert_two_PRC_'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                rbfnet = NetworkPRC(Kern, Kern_R)
                if load==0:
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_two, testloader_two, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            
            elif flag==3:
                print("both peaks, without noise")
                save_dir = 'results/sample_inverse_PRC/'
                model_ref = 'results/models/modella_both__PRC_'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data], x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                rbfnet = NetworkPRC(Kern, Kern_R)
                if load==0:
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_two, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            else:
                print("both peaks")
                save_dir = 'results/sample_uncert_PRC/'
                model_ref = 'results/models/modella_uncert_PRC_'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data], x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                rbfnet = NetworkPRC(Kern, Kern_R)
                if load==0:
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_two, omega_fine, tau, epochs, batche,\
                                        learning_rate, save_dir, model_ref, flag, params)
            torch.save(rbfnet.state_dict(), model_ref+str(runs))

        else:
            print(method)
            if flag==0:
                print("one peaks")
                save_dir = 'results/sample_one_peak/'
                model_ref = 'results/models/modella_uncert_one'+str(runs)

                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data]], axis=0)
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
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_one, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            
            elif flag==1:
                print("two peaks")
                save_dir = 'results/sample_two_peak/'
                model_ref = 'results/models/modella_uncert_two'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_two, testloader_two, omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            elif flag==3:
                print("both peaks, without noise")
                save_dir = 'results/sample_inverse/'
                model_ref = 'results/models/modella_both_'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data], x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))
                rbfnet.to(device)
                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_two, \
                          omega_fine, tau, epochs, batche, learning_rate, save_dir, model_ref, flag,params)
            else:
                print("both peaks")
                save_dir = 'results/sample_uncert/'
                model_ref = 'results/models/modella_uncert'+str(runs)
                tau = x['tau']
                omega_fine=x['omega_fine']
                omega=x['omega']
                R = np.concatenate([x['One_Peak_R_interp'][0:n_data], x['Two_Peak_R_interp'][0:n_data]], axis=0)
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
                    rbfnet.load_state_dict(torch.load(model_ref+str(runs)))

                rbfnet.to(device)

                rbfnet =  rbfnet.fit(trainloader, testloader_one, testloader_two,\
                omega_fine, tau, epochs, batche,\
                learning_rate, save_dir, model_ref, flag, params)

            torch.save(rbfnet.state_dict(), model_ref+str(runs))
