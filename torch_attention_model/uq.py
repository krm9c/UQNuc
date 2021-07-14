from model import *
from Lib import *
import time

import argparse
import json
import logging
import os
import shutil


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
    '--output_model',
    default=None,
    help="load model path")

parser.add_argument(
    '--input_model',
    default=None,
    help="kappa value")

parser.add_argument(
    '--input_flag',
    default='false',
    help="Directory containing params.json")

parser.add_argument(
    '--output_flag',
    default='true',
    help="Directory containing params.json")

parser.add_argument(
    '--save',
    default='true',
    help="Directory containing params.json")

parser.add_argument(
    '--json_file',
    default='One_peak.json',
    help="Directory containing params.json"
)

parser.add_argument(
    '--opt',
    default='Adam',
    help="The optimization")

parser.add_argument(
    '--learning_rate',
    default=None,
    help="learning rate")

parser.add_argument(
    '--total_runs',
    default=None,
    help="total number of runs value")

parser.add_argument(
    '--peak',
    default=None,
    help="one Peak or Two Peak")

parser.add_argument(
    '--n_points',
    default=None,
    help="number of data points")

parser.add_argument(
    '--batch_size',
    default=None,
    help="kappa value"
)

parser.add_argument(
    '--factor_reset',
    default=10000000,
    help="kappa value")

parser.add_argument(
    '--n_iterations',
    default=1000,
    help="kappa value")

parser.add_argument(
    '--factor_E',
    default=1,
    help="the E factor")

parser.add_argument(
    '--factor_R',
    default=1e7,
    help="the E factor")


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join('/home/kraghavan/Projects/Nuclear/UQNuc/orig_torch_code', args.json_file)
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict

    ##############################################
    if args.output_model is not None:
        params['output_model'] = args.output_model

    if args.input_model is not None:
        params['input_model'] = args.input_model

    if args.total_runs is not None:
        params['total_runs'] = int(args.total_runs)

    if args.learning_rate is not None:
        params['learning_rate'] = float(args.lr)

    if args.n_points is not None:
        params['n_datapoints'] = int(args.n_points)

    if args.peak is not None:
        params['peak'] = str(args.peak)

    if args.input_flag is not None:
        params['input_flag'] = int(args.input_flag)

    if args.output_flag is not None:
        params['output_flag'] = int(args.output_flag)

    if args.factor_reset is not None:
        params['factor_reset'] = int(args.factor_reset)

    if args.n_iterations is not None:
        params['n_iterations'] = int(args.n_iterations)

    if args.factor_E is not None:
        params['factor_E_rest_val'] = float(args.factor_E)

    params['factor_entropy_rest_val'] = float(args.factor_R)
    Config = params
    print("The config files are", Config)

    ##################################################
    # The one peak data
    x = return_dict(
        '/grand/NuQMC/UncertainityQ/theta_JLSE_Port/inverse_data_interpolated_numpy.p')
    print(x.keys())

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
    elif params['peak'] == 'both':
        R = np.concatenate([x['One_Peak_R_interp'],
                            x['Two_Peak_R_interp']
                            ], axis=0 )
        E = np.concatenate([x['One_Peak_E'],
                            x['Two_Peak_E']
                            ], axis=0 )

    # Extract the training data
    R = R[0:R.shape[0]-1000, :]
    E = E[0:E.shape[0]-1000, :]
    E_test = E[E.shape[0]-1000:, :]
    R_test = R[R.shape[0]-1000:, :]
    # Select indexes according to the number of datapoints
    index = np.random.randint(0, R.shape[0], Config['n_datapoints'])
    # The shape of the training data
    print("Training data", R.shape, E.shape)
    R = R[index]
    tau = x['tau']
    omega_ = x['omega']
    omega_fine = x['omega_fine']
    E, R, Kern, Kern_R = integrate(tau, omega_, omega_fine, R)
    ###################################################
    Kern_R[:, (Kern_R.shape[1]-1)] = 1
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


    # Instanciating and training an RBF network with the Gaussian basis function
    # This network receives a 2-dimensional input, transforms it into a 40-dimensional
    # hidden representation with an RBF layer and then transforms that into a
    # 1-dimensional output/prediction with a linear layer
    # To add more layers, change the layer_widths and layer_centres lists
    print(Kern_R.shape, E.shape, R.shape, E_test.shape, R_test.shape)
    layer_widths = [152, 200]
    layer_centres = [2000]
    basis_func = gaussian
    samples = 256
    rbfnet = Network(layer_widths, layer_centres, basis_func, Kern_R, Kern)
    rbfnet = rbfnet.to(device)

    print("The flag value before going to the input is", Config['input_flag'])
    if Config['input_flag'] ==1:
        print("I am going to load the model")
        rbfnet.load_state_dict(torch.load(Config['output_model']+'mod') )

    # E_hat_up, R_hat_up, E_hat_down, R_hat_down, E_t, R_t, LR, LE = \
    #     rbfnet.fit(E, R, E_test, R_test, Config['n_iterations'],
    #                Config['batch_size'], 0.001, tau, omega_fine, Config)


    LR, LE =  rbfnet.fit(E, R, E_test, R_test, Config['n_iterations'],
                   Config['batch_size'], 0.001, tau, omega_fine, Config)

    # print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    if Config['output_flag'] ==1 :
        print("I am going to save the model")
        torch.save(rbfnet.state_dict(), Config['output_model']+'mod')

    # def chi2_vec(y, yhat, factor):
    #     return np.mean(((y-yhat)**2/factor**2), axis=1)

    # chi_2_values = chi2_vec(E_t, E_hat_up, 0.0001)

    # print("no. of responses with chi2 E > 10:",
    #       len(chi_2_values[chi_2_values > 10]))
    # # chi_2_values = chi_2_values[chi_2_values<100]
    # print("Min", np.min(chi_2_values))
    # print("Median", np.median(chi_2_values))
    # print("Max", np.max(chi_2_values))
    # print("Mean", chi_2_values.mean())
    # print("std", chi_2_values.std())


    # # Save the final loss files.
    # LR = np.array(LR).reshape([-1, 1])
    # LE = np.array(LE).reshape([-1, 1])

    # fig, a = plt.subplots(1, 1, sharex=False,
    #                       dpi=600, gridspec_kw={'wspace': 0.7, 'hspace': 0.7})


    # a.plot(LR, label='R Loss', linewidth=1)
    # a.plot(LE, label='E Loss', linewidth=1)
    # a.set_xlabel("iterations")
    # a.set_ylabel('Loss value')
    # a.set_yscale('log')
    # a.legend()
    # plt.savefig(Config['output_model']+"losses.png", dpi=600)
    # np.savetxt(Config['output_model']+str("losses")+".csv",
    #            np.concatenate([LE, LR], axis=1), delimiter=',')
