import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import importlib
import os
from tqdm import tqdm
import re
from collections import OrderedDict
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib as mpl
plt.style.use(['dark_background'])
mpl.rcParams['axes.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['figure.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['legend.facecolor'] = (50 / 256, 50 / 256, 50 / 256)
mpl.rcParams['savefig.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
data_color = (0.2, 0.7, 1.0, 1.0)
samp_color = (1.0, 0.7, 0.1, 0.6)
aux_color = (0, 0.9, 0.0, 0.6)
colors = ['#ff005ca0', '#eb6a7d', '#cd98a1', '#9dbfc6', '#00ff80ff', '#00e2ed0b']

xrange = np.linspace(0, 1 / (25000 / 3 / 8) * 32 * 1000, 32)
B_xrange = np.array([-300.      , -281.81583 , -263.63168 , -245.44751 , -227.26334 ,
                     -209.07918 , -190.89502 , -172.71085 , -154.52669 , -136.34251 ,
                     -118.158356,  -99.97419 ,  -81.79002 ,  -63.60586 ,  -45.421696,
                      -27.237532,   -9.053368,    9.130797,   27.314962,   45.499126,
                       63.683292,   81.867455,  100.05162 ,  118.23579 ,  136.41995 ,
                      154.60411 ,  172.78828 ,  190.97244 ,  209.1566  ,  227.34077 ,
                      245.52493 ,  263.7091  ,  281.89328 ,  300.07742 ,  318.2616  ,
                      336.44577 ,  354.6299  ,  372.8141  ,  390.99826 ,  409.1824  ,
                      427.36658 ,  445.55075 ,  463.73492 ,  481.91907 ,  500.10324 ,
                      518.2874  ,  536.47156 ,  554.65576 ,  572.8399  ,  591.02405 ,
                      609.20825 ,  627.3924  ,  645.57654 ,  663.76074 ,  681.9449  ,
                      700.129   ,  718.31323 ,  736.4974  ,  754.6815  ,  772.8657  ,
                      791.04987 ,  809.234   ,  827.4182  ,  845.60236 ,  863.78656 ,
                      881.9707  ,  900.15485 ,  918.33905 ,  936.5232  ,  954.70734 ,
                      972.89154 ,  991.0757  , 1009.2598  , 1027.444   , 1045.6282  ,
                     1063.8124  , 1081.9965  , 1100.1807  , 1118.3649  , 1136.549   ,
                     1154.7332  , 1172.9174  , 1191.1014  , 1209.2856  , 1227.4698  ,
                     1245.6539  , 1263.8381  , 1282.0223  , 1300.2064  , 1318.3906  ,
                     1336.5748  , 1354.759   , 1372.9431  , 1391.1273  , 1409.3115  ,
                     1427.4956  , 1445.6798  , 1463.864   , 1482.0481  , 1500.2323  ,
                     1518.4165  , 1536.6006  , 1554.7848  , 1572.969   , 1591.1531  ,
                     1609.3373  , 1627.5215  , 1645.7056  , 1663.8898  , 1682.074   ,
                     1700.258   , 1718.4423  , 1736.6265  , 1754.8105  , 1772.9948  ,
                     1791.179   , 1809.363   , 1827.5472  , 1845.7314  , 1863.9155  ,
                     1882.0997  , 1900.2839  , 1918.468   , 1936.6522  , 1954.8364  ,
                     1973.0205  , 1991.2047  , 2009.3889  ])[::2]


class ConditionalFunc(torch.nn.Module):
    def __init__(self):
        super(ConditionalFunc, self).__init__()

    def forward(self, x): 
        # Average over discharge currents
        E_modified = (x[:, 133:183].mean(dim=1) - ((1100 - torch.tensor(mean[0], device=device, requires_grad=True)) / 
                                                   torch.tensor(ptp[0], device=device, requires_grad=True)))
        E_modified = E_modified.pow(2) * 0.001
        return E_modified


def conditional_sampler(x):
    conditional_func = ConditionalFunc().cuda()

    E_model = model(x)
    E_conditional = conditional_func(x)
    print("model: ")
    print(E_model[0:2])
    print("conditional: ")
    print(E_conditional[0:2])
    return E_model + E_conditional


def sample_langevin_cuda_tqdm(x, model, sample_steps=10, step_size=10, noise_scale=0.005, conditional_mask=None):
    if conditional_mask is None:
        conditional_mask = torch.ones((1, x.shape[1]), device=device)
    gradient_enabled_mask = torch.cat([torch.ones((1, x.shape[1] - 10)),
                                       torch.zeros((1, 10))], dim=1).to(x)
    for _ in tqdm(range(sample_steps)):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x + noise)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size * conditional_mask * gradient_enabled_mask
    return x


def plot_all_msi(data_samps, data_valid, data_valid_idx, ptp):
    data_sub_samps = data_samps[:]
    data_samps_mean = np.mean(data_sub_samps, axis=0)
    # data_samps_std = np.std(data_sub_samps, axis=0)

    fig, axes = plt.subplots(5, 2, figsize=(8, 7), dpi=200)

    for ax in axes.flatten():
        ax.set_autoscale_on(False)

    # [:, 32 * 0:32 * 1]
    # [:, 32 * 1:32 * 2]
    # [:, 32 * 2:32 * 3]
    # [:, 32 * 3:32 * 4]
    # [:, 32 * 4:32 * 5]
    # [:, 32 * 5:32 * 6]
    # [:, 32 * 6:32 * 7]
    # [:, 32 * 7:32 * 8]
    # [:, 32 * 8:32 * 8 + 64]
    # [:, 32 * 8 + 64:32 * 8 + 64 + 51]

    for i in range(data_samps.shape[0]):
        axes[0, 0].plot(xrange, data_samps[i, 32 * 0:32 * 1] * ptp['discharge_current'], color=colors[-1], label="Sampled")
        axes[0, 1].plot(xrange, data_samps[i, 32 * 1:32 * 2] * ptp['discharge_voltage'], color=colors[-1])
        axes[1, 0].plot(xrange, data_samps[i, 32 * 2:32 * 3] * ptp['interferometer'], color=colors[-1])
        axes[1, 1].plot(xrange, data_samps[i, 32 * 3:32 * 4] * ptp['diode_0'], color=colors[-1])
        axes[2, 0].plot(xrange, data_samps[i, 32 * 4:32 * 5] * ptp['diode_1'], color=colors[-1])
        axes[2, 1].plot(xrange, data_samps[i, 32 * 5:32 * 6] * ptp['diode_2'], color=colors[-1])
        axes[3, 0].plot(xrange, data_samps[i, 32 * 6:32 * 7] * ptp['diode_3'], color=colors[-1])
        axes[3, 1].plot(xrange, data_samps[i, 32 * 7:32 * 8] * ptp['diode_4'], color=colors[-1])
        axes[4, 0].plot(B_xrange, data_samps[i, 32 * 8:32 * 8 + 64] * ptp['magnet_profile'], color=colors[-1])
        axes[4, 1].scatter(np.arange(51), data_samps[i, 32 * 8 + 64:32 * 8 + 64 + 51] * ptp['pressures'], color=colors[-1])

    for ax in axes.flatten():
        ax.set_autoscale_on(True)

    axes[0, 0].plot(xrange, data_samps_mean[32 * 0:32 * 1] * ptp['discharge_current'], color=colors[4], label="Sampled mean")
    axes[0, 1].plot(xrange, data_samps_mean[32 * 1:32 * 2] * ptp['discharge_voltage'], color=colors[4])
    axes[1, 0].plot(xrange, data_samps_mean[32 * 2:32 * 3] * ptp['interferometer'], color=colors[4])
    axes[1, 1].plot(xrange, data_samps_mean[32 * 3:32 * 4] * ptp['diode_0'], color=colors[4])
    axes[2, 0].plot(xrange, data_samps_mean[32 * 4:32 * 5] * ptp['diode_1'], color=colors[4])
    axes[2, 1].plot(xrange, data_samps_mean[32 * 5:32 * 6] * ptp['diode_2'], color=colors[4])
    axes[3, 0].plot(xrange, data_samps_mean[32 * 6:32 * 7] * ptp['diode_3'], color=colors[4])
    axes[3, 1].plot(xrange, data_samps_mean[32 * 7:32 * 8] * ptp['diode_4'], color=colors[4])
    axes[4, 0].plot(B_xrange, data_samps_mean[32 * 8:32 * 8 + 64] * ptp['magnet_profile'], color=colors[4])
    axes[4, 1].scatter(np.arange(51), data_samps_mean[32 * 8 + 64:32 * 8 + 64 + 51] * ptp['pressures'], color=colors[4])

    axes[0, 0].plot(xrange, data_valid[data_valid_idx, 32 * 0:32 * 1] * ptp['discharge_current'], color=colors[0])
    axes[0, 1].plot(xrange, data_valid[data_valid_idx, 32 * 1:32 * 2] * ptp['discharge_voltage'], color=colors[0])
    axes[1, 0].plot(xrange, data_valid[data_valid_idx, 32 * 2:32 * 3] * ptp['interferometer'], color=colors[0])
    axes[1, 1].plot(xrange, data_valid[data_valid_idx, 32 * 3:32 * 4] * ptp['diode_0'], color=colors[0])
    axes[2, 0].plot(xrange, data_valid[data_valid_idx, 32 * 4:32 * 5] * ptp['diode_1'], color=colors[0])
    axes[2, 1].plot(xrange, data_valid[data_valid_idx, 32 * 5:32 * 6] * ptp['diode_2'], color=colors[0])
    axes[3, 0].plot(xrange, data_valid[data_valid_idx, 32 * 6:32 * 7] * ptp['diode_3'], color=colors[0])
    axes[3, 1].plot(xrange, data_valid[data_valid_idx, 32 * 7:32 * 8] * ptp['diode_4'], color=colors[0])
    axes[4, 0].plot(B_xrange, data_valid[data_valid_idx, 32 * 8:32 * 8 + 64] * ptp['magnet_profile'], color=colors[0])
    axes[4, 1].scatter(np.arange(51), data_valid[data_valid_idx, 32 * 8 + 64:32 * 8 + 64 + 51] * ptp['pressures'], color=colors[0])

    axes[0, 0].plot(xrange, data_valid[15000, 32 * 0:32 * 1] * ptp['discharge_current'], color=colors[0])
    axes[0, 1].plot(xrange, data_valid[15000, 32 * 1:32 * 2] * ptp['discharge_voltage'], color=colors[0])
    axes[1, 0].plot(xrange, data_valid[15000, 32 * 2:32 * 3] * ptp['interferometer'], color=colors[0])
    axes[1, 1].plot(xrange, data_valid[15000, 32 * 3:32 * 4] * ptp['diode_0'], color=colors[0])
    axes[2, 0].plot(xrange, data_valid[15000, 32 * 4:32 * 5] * ptp['diode_1'], color=colors[0])
    axes[2, 1].plot(xrange, data_valid[15000, 32 * 5:32 * 6] * ptp['diode_2'], color=colors[0])
    axes[3, 0].plot(xrange, data_valid[15000, 32 * 6:32 * 7] * ptp['diode_3'], color=colors[0])
    axes[3, 1].plot(xrange, data_valid[15000, 32 * 7:32 * 8] * ptp['diode_4'], color=colors[0])
    axes[4, 0].plot(B_xrange, data_valid[15000, 32 * 8:32 * 8 + 64] * ptp['magnet_profile'], color=colors[0])
    axes[4, 1].scatter(np.arange(51), data_valid[15000, 32 * 8 + 64:32 * 8 + 64 + 51] * ptp['pressures'], color=colors[0])

    axes[0, 0].set_title('Discharge I (Amps) vs ms')
    axes[0, 1].set_title('Discharge V vs ms')
    axes[1, 0].set_title('Interferometer (Volts) vs ms')
    axes[1, 1].set_title('Diode 0 (Volts) vs ms')
    axes[2, 0].set_title('Diode 1 (Volts) vs ms')
    axes[2, 1].set_title('Diode 2 (Volts) vs ms')
    axes[3, 0].set_title('Diode 3 (Volts) vs ms')
    axes[3, 1].set_title('Diode 4 (HeII) (Volts) vs ms')
    axes[4, 0].set_title('Magnet field profile (G) vs cm')
    axes[4, 1].set_title('log(RGA) (Torr)')
    plt.suptitle("Real discharges = red, conditionally sampled = blue")
    plt.tight_layout()


def plot_diagnostics_histogram(data_samps, data, ptp):
    n_bins = 200
    fig, axes = plt.subplots(5, 2, figsize=(8, 7), dpi=200)
    axes[0, 0].hist(np.mean(data[:, 32 * 0:32 * 1].detach().numpy()[:, 16:22], axis=1) * ptp['discharge_current'], bins=n_bins, density=True, color=data_color)
    axes[0, 1].hist(np.mean(data[:, 32 * 1:32 * 2].detach().numpy()[:, 16:22], axis=1) * ptp['discharge_voltage'], bins=n_bins, density=True, color=data_color)
    axes[1, 0].hist(np.mean(data[:, 32 * 2:32 * 3].detach().numpy()[:, 16:22], axis=1) * ptp['interferometer'], bins=n_bins, density=True, color=data_color)
    axes[1, 1].hist(np.mean(data[:, 32 * 3:32 * 4].detach().numpy()[:, 16:22], axis=1) * ptp['diode_0'], bins=n_bins, density=True, color=data_color)
    axes[2, 0].hist(np.mean(data[:, 32 * 4:32 * 5].detach().numpy()[:, 16:22], axis=1) * ptp['diode_1'], bins=n_bins, density=True, color=data_color)
    axes[2, 1].hist(np.mean(data[:, 32 * 5:32 * 6].detach().numpy()[:, 16:22], axis=1) * ptp['diode_2'], bins=n_bins, density=True, color=data_color)
    axes[3, 0].hist(np.mean(data[:, 32 * 6:32 * 7].detach().numpy()[:, 16:22], axis=1) * ptp['diode_3'], bins=n_bins, density=True, color=data_color)
    axes[3, 1].hist(np.mean(data[:, 32 * 7:32 * 8].detach().numpy()[:, 16:22], axis=1) * ptp['diode_4'], bins=n_bins, density=True, color=data_color)
    axes[4, 0].hist(np.std(data[:, 32 * 8:32 * 8 + 64].detach().numpy(), axis=1) * ptp['magnet_profile'], bins=n_bins, density=True, color=data_color)
    axes[4, 1].hist(data[:, 32 * 8 + 64 + 4].detach().numpy() * ptp['pressures'], bins=n_bins, density=True, color=data_color)

    axes[0, 0].hist(np.mean(data_samps[:, 32 * 0:32 * 1][:, 16:22], axis=1) * ptp['discharge_current'], bins=n_bins, density=True, color=aux_color)
    axes[0, 1].hist(np.mean(data_samps[:, 32 * 1:32 * 2][:, 16:22], axis=1) * ptp['discharge_voltage'], bins=n_bins, density=True, color=aux_color)
    axes[1, 0].hist(np.mean(data_samps[:, 32 * 2:32 * 3][:, 16:22], axis=1) * ptp['interferometer'], bins=n_bins, density=True, color=aux_color)
    axes[1, 1].hist(np.mean(data_samps[:, 32 * 3:32 * 4][:, 16:22], axis=1) * ptp['diode_0'], bins=n_bins, density=True, color=aux_color)
    axes[2, 0].hist(np.mean(data_samps[:, 32 * 4:32 * 5][:, 16:22], axis=1) * ptp['diode_1'], bins=n_bins, density=True, color=aux_color)
    axes[2, 1].hist(np.mean(data_samps[:, 32 * 5:32 * 6][:, 16:22], axis=1) * ptp['diode_2'], bins=n_bins, density=True, color=aux_color)
    axes[3, 0].hist(np.mean(data_samps[:, 32 * 6:32 * 7][:, 16:22], axis=1) * ptp['diode_3'], bins=n_bins, density=True, color=aux_color)
    axes[3, 1].hist(np.mean(data_samps[:, 32 * 7:32 * 8][:, 16:22], axis=1) * ptp['diode_4'], bins=n_bins, density=True, color=aux_color)
    axes[4, 0].hist(np.std(data_samps[:, 32 * 8:32 * 8 + 64], axis=1) * ptp['magnet_profile'], bins=n_bins, density=True, color=aux_color)
    axes[4, 1].hist(data_samps[:, 32 * 8 + 64 + 4] * ptp['pressures'], bins=n_bins, density=True, color=aux_color)

    axes[0, 0].set_title('Current')
    axes[0, 1].set_title('Voltage')
    axes[1, 0].set_title('Interferometer')
    axes[1, 1].set_title('Diode 0')
    axes[2, 0].set_title('Diode 1')
    axes[2, 1].set_title('Diode 2')
    axes[3, 0].set_title('Diode 2')
    axes[3, 1].set_title('Diode 3')
    axes[4, 0].set_title('Magnets')
    axes[4, 1].set_title('RGA (Helium)')
    plt.suptitle('MSI histogram (mean of 133 to 183)')
    plt.tight_layout()


def plot_energy_histogram(data, data_samps):
    n_bins = 201
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    plt.title('Energy histogram')
    axes.hist(model(data[np.random.randint(0, data.shape[0], 96)].to(device)).to('cpu').detach().numpy(),
              bins=n_bins, density=True, color=data_color)
    axes.hist(model(torch.tensor(data_samps).to(device)).to('cpu').detach().numpy(), bins=n_bins, density=True,
              color=aux_color)


def conditionally_sample(init_data, n_samp, steps, step_size, noise, samp_begin, samp_end, includes_flags=True):
    data_samps = init_data.clone()
    data_samps[:, samp_begin:samp_end] = torch.rand((n_samp, samp_end - samp_begin), requires_grad=True).to(device) * 2 - 1
    # samps = perturb_samples(samps)
    conditional_mask = torch.zeros((1, data_samps.shape[1]), device=device)
    conditional_mask[:, samp_begin:samp_end] = torch.ones((1, samp_end - samp_begin), device=device)

    for i in tqdm(range(3)):
        if includes_flags:
            data_samps[:, samp_begin:samp_end - 10] = perturb_samples(data_samps[:, samp_begin:samp_end - 10])
        else:
            data_samps[:, samp_begin:samp_end] = perturb_samples(data_samps[:, samp_begin:samp_end])
        data_samps = sample_langevin_cuda_tqdm(data_samps, model, step_size=step_size, sample_steps=steps,
                                               noise_scale=noise, conditional_mask=conditional_mask)
    data_samps = data_samps.to('cpu').detach().numpy()

    # torch.cuda.empty_cache()
    return data_samps


if __name__ == '__main__':
    device = torch.device('cuda:0')

    sys.path.remove('/home/phil/Desktop/EBMs/lapd-ebm')  # Remove this script directory path

    experiments_dir = "/home/phil/Desktop/EBMs/lapd-ebm/experiments_modular"
    # filepaths = []
    # for dirpath, dirs, files in os.walk(experiments_dir):
    #     for f in files:
    #         if ".pt" in f:
    #             filepaths.append(os.path.join(dirpath, f))
    # filepaths = sorted(filepaths)

    # filepaths = filepaths[133:]
    filepaths = [
                 # '2022-07-13_11h-31m-02s/checkpoints/model-0-3.pt',
                 # '2022-07-13_11h-31m-02s/checkpoints/model-0-352.pt',
                 '2022-07-13_11h-31m-02s/checkpoints/model-0-700.pt'
                 ]

    for f in filepaths:
        try:
            pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}h-\d{2}m-\d{2}s)\/checkpoints\/(model-\d+-\d+.pt)")
            match = re.search(pattern, f)
            identifier = match.group(1)
            rundir = experiments_dir + "/" + identifier
            model_num = match.group(2)

            with open(rundir + "/" + "modular_ebm_copy.py", 'r') as infile:
                for line in infile:
                    if "data_path = " in line:
                        data_filename = (line[17:-6])
                    if "\"step_size\": " in line:
                        step_size = float(line[21:-2])
                    if "\"sample_steps\": " in line:
                        sample_steps = int(line[24:-2])
                    if "\"noise_scale\": " in line:
                        noise_scale = float(line[23:-2])

        # ------------------------------------------ #
        # Everything below this goes in the for loop
        # ------------------------------------------ #

            # ------ Load model ------ #
            print("Loading model " + identifier + " " + model_num + ". ")

            # Add path for import the correct diagnostic and sampling modules
            sys.path.insert(0, os.path.join(experiments_dir, identifier))

            spec = importlib.util.spec_from_file_location("modular_ebm_copy", rundir + "/modular_ebm_copy.py")
            ebm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ebm)
            sample_langevin = ebm.sample_langevin
            sample_langevin_cuda = ebm.sample_langevin_cuda
            ReplayBuffer = ebm.ReplayBuffer
            try:
                perturb_samples = ebm.perturb_samples
            except:
                def perturb_samples(x):
                    return x

            model = ebm.ModularWithRNNBackbone().to(device)
            ckpt = torch.load("experiments_modular/" + f)

            model_dict = OrderedDict()
            pattern = re.compile('module.')
            state_dict = ckpt['model_state_dict']
            for k, v in state_dict.items():
                if re.search("module", k):
                    model_dict[re.sub(pattern, '', k)] = v
                else:
                    model_dict = state_dict
            model.load_state_dict(model_dict, strict=True)

            # ------ Load data ------ #

            # data_all = np.load(data_filename + ".npz")['signals']
            if 'train' in data_filename:
                data_filename = data_filename[:-6]
            data_valid = ebm.load_data(data_filename + "-valid.npz")

            data_path = data_filename + "-valid.npz"
            data = ebm.load_data(data_path)
            mean = np.zeros((10))
            ptp = np.load(data_path)['scale']
            datasize = data.shape[1]

            # replay_buffer = ReplayBuffer(ckpt['replay_buffer_list'].shape[0], np.random.randn(*data.shape))
            # replay_buffer.sample_list = ckpt['replay_buffer_list']
            print("Number of parameters: {}. ".format(np.sum([p.numel() for p in model.parameters() if p.requires_grad])))

            print("Unconditional sampling...")

            n_samp = 96
            steps = sample_steps * 2
            step_size = step_size
            noise = noise_scale

            # data_valid_idx = 15000
            data_valid_idx = 50123
            samp_begin = 32 * 0
            samp_end = datasize

            init_data = data_valid[data_valid_idx].clone().detach().repeat((n_samp, 1)).to(device)
            data_samps = conditionally_sample(init_data, n_samp, steps, step_size, noise, samp_begin, samp_end, includes_flags=True)

            print("Plotting unconditional samples")

            plot_all_msi(data_samps, data_valid, data_valid_idx, ptp)
            plt.savefig("experiments_modular/" + identifier + "/plots/traces-unconditional-" + model_num + ".png")

            # ------ Plot energy histrogram ------ #

            plot_energy_histogram(data, data_samps)
            plt.savefig("experiments_modular/" + identifier + "/plots/energies-unconditional-" + model_num + ".png")

            # ------ Plot Diagnostics histogram ------ #

            plot_diagnostics_histogram(data_samps, data, ptp)
            plt.savefig("experiments_modular/" + identifier + "/plots/hist-unconditional-" + model_num + ".png")

            print("Conditional sampling...")
            data_valid_idx = 50123
            samp_begin = 32 * 0
            samp_end = 32 * 8
            data_samps = conditionally_sample(init_data, n_samp, steps, step_size, noise, samp_begin, samp_end, includes_flags=False)
            print("Plotting conditional samples")
            plot_all_msi(data_samps, data_valid, data_valid_idx, ptp)
            plt.savefig("experiments_modular/" + identifier + "/plots/traces-conditional-" + model_num + ".png")
            plot_energy_histogram(data, data_samps)
            plt.savefig("experiments_modular/" + identifier + "/plots/energies-conditional-" + model_num + ".png")
            plot_diagnostics_histogram(data_samps, data, ptp)
            plt.savefig("experiments_modular/" + identifier + "/plots/hist-conditional-" + model_num + ".png")

            print("Single diagnostic sampling...")
            data_valid_idx = 50123
            samp_begin = 32 * 5
            samp_end = 32 * 6
            data_samps = conditionally_sample(init_data, n_samp, steps, step_size, noise, samp_begin, samp_end, includes_flags=False)
            range_begin = 32 * 5
            range_end = 32 * (5 + 1)
            data_sub_samps = data_samps[:, range_begin:range_end]
            data_samps_mean = np.mean(data_sub_samps, axis=0)
            data_samps_std = np.std(data_sub_samps, axis=0)
            idx = np.random.randint(0, data_samps.shape[0])
            fig, axes = plt.subplots(2, 1, figsize=(5, 5), dpi=200, sharex=True)

            for i in range(data_sub_samps.shape[0]):
                axes[0].plot(xrange, data_sub_samps[i] * ptp['diode_1'], color=colors[-1], label="Sampled" if i == 0 else None)
            axes[0].plot(xrange, data_valid[data_valid_idx, range_begin:range_end] * ptp['diode_1'], color=colors[0], label="Real")
            axes[0].plot(xrange, data_samps_mean * ptp['diode_1'], color=colors[4], label="Sampled mean")
            for i in range(data_sub_samps.shape[0]):
                axes[1].plot(xrange, (data_sub_samps[i] - data_samps_mean) * ptp['diode_1'], color=colors[-1])
            axes[1].plot(xrange, (data_valid[data_valid_idx, range_begin:range_end] - data_samps_mean) * ptp['diode_1'], color=colors[0])
            axes[1].plot(xrange, np.zeros(32), color=colors[4])
            axes[0].set_title('Diode 1 signal (Volts)')
            axes[1].set_title('Real minus sampled')
            axes[1].set_xlabel('Time(ms)')
            axes[0].legend()
            plt.tight_layout()
            plt.savefig("experiments_modular/" + identifier + "/plots/traces-single-" + model_num + ".png")

            plot_energy_histogram(data, data_samps)
            plt.savefig("experiments_modular/" + identifier + "/plots/energies-single-" + model_num + ".png")
            plt.close('all')

            # Remove module path
            sys.path.remove(os.path.join(experiments_dir, identifier))
        except OSError as e:
            print("Generating plots failed for " + f)
            print(e, e.args)
            # Just in case:
            try:
                sys.path.remove(os.path.join(experiments_dir, identifier))
            except:
                pass
