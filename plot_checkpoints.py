import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import importlib
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib as mpl
plt.style.use(['dark_background'])
mpl.rcParams['axes.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['figure.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['legend.facecolor'] = (50 / 256, 50 / 256, 50 / 256)
mpl.rcParams['savefig.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
data_color = (0.2, 0.7, 1.0, 1.0)
samp_color = (1.0, 0.7, 0.1, 0.6)
aux_color = (0, 0.9, 0.0, 0.6)

device = torch.device('cuda:0')


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
    for _ in tqdm(range(sample_steps)):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x + noise)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size * conditional_mask
    return x


experiments_dir = "/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi"
# filepaths = []
# for dirpath, dirs, files in os.walk(experiments_dir):
#     for f in files:
#         if ".pt" in f:
#             filepaths.append(os.path.join(dirpath, f))
# filepaths = sorted(filepaths)

# filepaths = filepaths[133:]
filepaths = [
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-0.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-10.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-20.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-30.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-40.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-50.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-60.pt',
            # '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-70.pt'
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-80.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-90.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-100.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-110.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-120.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-130.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-140.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-150.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-160.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-170.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-180.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-190.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-200.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-210.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-220.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-230.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-240.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-250.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-260.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-270.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-280.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-290.pt',
            '/home/phil/Desktop/EBMs/lapd-ebm/experiments_msi/2022-03-31_16h-40m-04s/checkpoints/model-300.pt'
            ]

for f in filepaths:
    try:
        identifier = f[49:71]
        rundir = experiments_dir + "/" + identifier
        model_num = f[90:-3]

        with open(rundir + "/" + "msi_ebm_copy.py", 'r') as infile:
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

        # Need to extract from file: steps, step_size, noise, data path

        # data_all = np.load(data_filename + ".npz")['signals']
        if 'train' in data_filename:
            data_filename = data_filename[:-6]
        data_valid = np.load(data_filename + "-valid.npz")['signals']

        # ------ Load model ------ #

        model_path = identifier
        model_version = f[72:-3]

        print("Loading model " + model_path + "/" + model_version + ". ")

        spec = importlib.util.spec_from_file_location("msi_ebm_copy", "experiments_msi/" + model_path + "/msi_ebm_copy.py")
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

        model = ebm.NeuralNet().to(device)
        ckpt = torch.load("experiments_msi/" + model_path + "/" + model_version + ".pt")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        # data = torch.tensor(np.load("data/isat_downsampled_8_div3.npz")['arr_0'].reshape(-1, 10)).float()

        data_path = data_filename + "-train.npz"
        data = ebm.load_data(data_path)
        mean = np.load(data_path)['mean']
        ptp = np.load(data_path)['ptp']

        replay_buffer = ReplayBuffer(ckpt['replay_buffer_list'].shape[0], np.random.randn(*data.shape))
        replay_buffer.sample_list = ckpt['replay_buffer_list']
        print("Number of parameters: {}. ".format(np.sum([p.numel() for p in model.parameters() if p.requires_grad])))

        print("Unconditional sampling...")

        n_samp = 256
        steps = sample_steps * 2
        step_size = step_size
        noise = noise_scale

        # data_valid_idx = 15000
        data_valid_idx = 22000
        samp_begin = 256 * 0
        samp_end = 1715

        def conditionally_sample(n_samp, steps, step_size, noise, idx, samp_begin, samp_end):
            data_samps = torch.tensor(data_valid[idx], device=device).repeat((n_samp, 1))
            data_samps[:, samp_begin:samp_end] = torch.rand((n_samp, samp_end - samp_begin), requires_grad=True).to(device) * 2 - 1
            # samps = perturb_samples(samps)
            conditional_mask = torch.zeros((1, 1715), device=device)
            conditional_mask[:, samp_begin:samp_end] = torch.ones((1, samp_end - samp_begin), device=device)

            for i in tqdm(range(3)):
                data_samps[:, samp_begin:samp_end] = perturb_samples(data_samps[:, samp_begin:samp_end])
                data_samps = sample_langevin_cuda_tqdm(data_samps, model, step_size=step_size, sample_steps=steps,
                                                       noise_scale=noise, conditional_mask=conditional_mask)
            data_samps = data_samps.to('cpu').detach().numpy()
            return data_samps

        data_samps = conditionally_sample(n_samp, steps, step_size, noise,
                                          data_valid_idx, samp_begin, samp_end)

        print("Plotting unconditional samples")

        xrange = np.linspace(0, 1 / (25000 / 3) * 255 * 1000, 256)
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
                             1973.0205  , 1991.2047  , 2009.3889  ])

        colors = ['#ff005ca0', '#eb6a7d', '#cd98a1', '#9dbfc6', '#00ff80ff', '#00e2ed0b']

        def plot_all_msi(data_samps):
            data_sub_samps = data_samps[:]
            data_samps_mean = np.mean(data_sub_samps, axis=0)
            # data_samps_std = np.std(data_sub_samps, axis=0)

            fig, axes = plt.subplots(4, 2, figsize=(8, 7), dpi=200)

            for ax in axes.flatten():
                ax.set_autoscale_on(False)

            for i in range(data_samps.shape[0]):
                axes[0, 0].plot(xrange, data_samps[i, 256 * 0:256 * 1] * ptp[0] + mean[0], color=colors[-1], label="Sampled")
                axes[1, 0].plot(xrange, data_samps[i, 256 * 1:256 * 2] * ptp[1] + mean[1], color=colors[-1])
                axes[2, 0].plot(xrange, data_samps[i, 256 * 2:256 * 3] * ptp[2] + mean[2], color=colors[-1])
                axes[0, 1].plot(xrange, data_samps[i, 256 * 3:256 * 4] * ptp[3] + mean[3], color=colors[-1])
                axes[1, 1].plot(xrange, data_samps[i, 256 * 4:256 * 5] * ptp[4] + mean[4], color=colors[-1])
                axes[2, 1].plot(xrange, data_samps[i, 256 * 5:256 * 6] * ptp[5] + mean[5], color=colors[-1])
                axes[3, 0].plot(B_xrange, data_samps[i, 256 * 6:256 * 6 + 128] * ptp[6] + mean[6], color=colors[-1])
                axes[3, 1].scatter(np.arange(51), data_samps[i, -51:] * ptp[7] + mean[7], color=colors[-1])

            for ax in axes.flatten():
                ax.set_autoscale_on(True)

            axes[0, 0].plot(xrange, data_samps_mean[256 * 0:256 * 1] * ptp[0] + mean[0], color=colors[4], label="Sampled mean")
            axes[1, 0].plot(xrange, data_samps_mean[256 * 1:256 * 2] * ptp[1] + mean[1], color=colors[4])
            axes[2, 0].plot(xrange, data_samps_mean[256 * 2:256 * 3] * ptp[2] + mean[2], color=colors[4])
            axes[0, 1].plot(xrange, data_samps_mean[256 * 3:256 * 4] * ptp[3] + mean[3], color=colors[4])
            axes[1, 1].plot(xrange, data_samps_mean[256 * 4:256 * 5] * ptp[4] + mean[4], color=colors[4])
            axes[2, 1].plot(xrange, data_samps_mean[256 * 5:256 * 6] * ptp[5] + mean[5], color=colors[4])
            axes[3, 0].plot(B_xrange, data_samps_mean[256 * 6:256 * 6 + 128] * ptp[6] + mean[6], color=colors[4])
            axes[3, 1].scatter(np.arange(51), data_samps_mean[-51:] * ptp[7] + mean[7], color=colors[4])

            axes[0, 0].plot(xrange, data_valid[data_valid_idx, 256 * 0:256 * 1] * ptp[0] + mean[0], color=colors[0])
            axes[1, 0].plot(xrange, data_valid[data_valid_idx, 256 * 1:256 * 2] * ptp[1] + mean[1], color=colors[0])
            axes[2, 0].plot(xrange, data_valid[data_valid_idx, 256 * 2:256 * 3] * ptp[2] + mean[2], color=colors[0])
            axes[0, 1].plot(xrange, data_valid[data_valid_idx, 256 * 3:256 * 4] * ptp[3] + mean[3], color=colors[0])
            axes[1, 1].plot(xrange, data_valid[data_valid_idx, 256 * 4:256 * 5] * ptp[4] + mean[4], color=colors[0])
            axes[2, 1].plot(xrange, data_valid[data_valid_idx, 256 * 5:256 * 6] * ptp[5] + mean[5], color=colors[0])
            axes[3, 0].plot(B_xrange, data_valid[data_valid_idx, 256 * 6:256 * 6 + 128] * ptp[6] + mean[6], color=colors[0])
            axes[3, 1].scatter(np.arange(51), data_valid[data_valid_idx, -51:] * ptp[7] + mean[7], color=colors[0])

            axes[0, 0].plot(xrange, data_valid[15000, 256 * 0:256 * 1] * ptp[0] + mean[0], color=colors[0])
            axes[1, 0].plot(xrange, data_valid[15000, 256 * 1:256 * 2] * ptp[1] + mean[1], color=colors[0])
            axes[2, 0].plot(xrange, data_valid[15000, 256 * 2:256 * 3] * ptp[2] + mean[2], color=colors[0])
            axes[0, 1].plot(xrange, data_valid[15000, 256 * 3:256 * 4] * ptp[3] + mean[3], color=colors[0])
            axes[1, 1].plot(xrange, data_valid[15000, 256 * 4:256 * 5] * ptp[4] + mean[4], color=colors[0])
            axes[2, 1].plot(xrange, data_valid[15000, 256 * 5:256 * 6] * ptp[5] + mean[5], color=colors[0])
            axes[3, 0].plot(B_xrange, data_valid[15000, 256 * 6:256 * 6 + 128] * ptp[6] + mean[6], color=colors[0])
            axes[3, 1].scatter(np.arange(51), data_valid[15000, -51:] * ptp[7] + mean[7], color=colors[0])

            axes[0, 0].set_title('Discharge I (Amps) vs ms')
            axes[1, 0].set_title('Discharge V vs ms')
            axes[2, 0].set_title('Interferometer (Volts) vs ms')
            axes[0, 1].set_title('Diode 0 (Volts) vs ms')
            axes[1, 1].set_title('Diode 1 (Volts) vs ms')
            axes[2, 1].set_title('Diode 2 (Volts) vs ms')
            axes[3, 0].set_title('Magnet field profile (G) vs cm')
            axes[3, 1].set_title('log(RGA) (Torr)')
            plt.suptitle("Real discharges = red, conditionally sampled = blue")
            plt.tight_layout()

        plot_all_msi(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/traces-unconditional-" + model_num + ".png")

        # ------ Plot energy histrogram ------ #

        def plot_energy_histogram(data_samps):
            n_bins = 201
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
            plt.title('Energy histogram')
            axes.hist(model(data[np.random.randint(0, data.shape[0], 2000)].to(device)).to('cpu').detach().numpy(),
                      bins=n_bins, density=True, color=data_color)
            axes.hist(model(torch.tensor(data_samps).to(device)).to('cpu').detach().numpy(), bins=n_bins, density=True,
                      color=aux_color)

        plot_energy_histogram(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/energies-unconditional-" + model_num + ".png")

        # ------ Plot Diagnostics histogram ------ #

        def plot_diagnostics_histogram(data_samps):
            n_bins = 200
            fig, axes = plt.subplots(4, 2, figsize=(8, 7), dpi=200)
            axes[0, 0].hist(np.mean(data[:, 256 * 0:256 * 1].detach().numpy()[:, 133:183], axis=1) * ptp[0] + mean[0], bins=n_bins, density=True, color=data_color)
            axes[0, 1].hist(np.mean(data[:, 256 * 1:256 * 2].detach().numpy()[:, 133:183], axis=1) * ptp[1] + mean[1], bins=n_bins, density=True, color=data_color)
            axes[1, 0].hist(np.mean(data[:, 256 * 2:256 * 3].detach().numpy()[:, 133:183], axis=1) * ptp[2] + mean[2], bins=n_bins, density=True, color=data_color)
            axes[1, 1].hist(np.mean(data[:, 256 * 3:256 * 4].detach().numpy()[:, 133:183], axis=1) * ptp[3] + mean[3], bins=n_bins, density=True, color=data_color)
            axes[2, 0].hist(np.mean(data[:, 256 * 4:256 * 5].detach().numpy()[:, 133:183], axis=1) * ptp[4] + mean[4], bins=n_bins, density=True, color=data_color)
            axes[2, 1].hist(np.mean(data[:, 256 * 5:256 * 6].detach().numpy()[:, 133:183], axis=1) * ptp[5] + mean[5], bins=n_bins, density=True, color=data_color)
            axes[3, 0].hist(np.std(data[:, 256 * 6:256 * 6 + 128].detach().numpy(), axis=1) * ptp[6] + mean[6], bins=n_bins, density=True, color=data_color)
            axes[3, 1].hist(data[:, 1715 - 51 + 4].detach().numpy() * ptp[7] + mean[7], bins=n_bins, density=True, color=data_color)

            axes[0, 0].hist(np.mean(data_samps[:, 256 * 0:256 * 1][:, 133:183], axis=1) * ptp[0] + mean[0], bins=n_bins, density=True, color=aux_color)
            axes[0, 1].hist(np.mean(data_samps[:, 256 * 1:256 * 2][:, 133:183], axis=1) * ptp[1] + mean[1], bins=n_bins, density=True, color=aux_color)
            axes[1, 0].hist(np.mean(data_samps[:, 256 * 2:256 * 3][:, 133:183], axis=1) * ptp[2] + mean[2], bins=n_bins, density=True, color=aux_color)
            axes[1, 1].hist(np.mean(data_samps[:, 256 * 3:256 * 4][:, 133:183], axis=1) * ptp[3] + mean[3], bins=n_bins, density=True, color=aux_color)
            axes[2, 0].hist(np.mean(data_samps[:, 256 * 4:256 * 5][:, 133:183], axis=1) * ptp[4] + mean[4], bins=n_bins, density=True, color=aux_color)
            axes[2, 1].hist(np.mean(data_samps[:, 256 * 5:256 * 6][:, 133:183], axis=1) * ptp[5] + mean[5], bins=n_bins, density=True, color=aux_color)
            axes[3, 0].hist(np.std(data_samps[:, 256 * 6:256 * 6 + 128], axis=1) * ptp[6] + mean[6], bins=n_bins, density=True, color=aux_color)
            axes[3, 1].hist(data_samps[:, 1715 - 51 + 4] * ptp[7] + mean[7], bins=n_bins, density=True, color=aux_color)

            axes[0, 0].set_title('Current')
            axes[0, 1].set_title('Voltage')
            axes[1, 0].set_title('Interferometer')
            axes[1, 1].set_title('Diode 0')
            axes[2, 0].set_title('Diode 1')
            axes[2, 1].set_title('Diode 2')
            axes[3, 0].set_title('Magnets')
            axes[3, 1].set_title('RGA (Helium)')
            plt.suptitle('MSI histogram (mean of 133 to 183)')
            plt.tight_layout()

        plot_diagnostics_histogram(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/hist-unconditional-" + model_num + ".png")

        print("Conditional sampling...")
        data_valid_idx = 22000
        samp_begin = 256 * 0
        samp_end = 256 * 6
        data_samps = conditionally_sample(n_samp, steps, step_size, noise,
                                          data_valid_idx, samp_begin, samp_end)
        print("Plotting conditional samples")
        plot_all_msi(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/traces-conditional-" + model_num + ".png")
        plot_energy_histogram(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/energies-conditional-" + model_num + ".png")
        plot_diagnostics_histogram(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/hist-conditional-" + model_num + ".png")

        print("Single diagnostic sampling...")
        data_valid_idx = 22000
        samp_begin = 256 * 5
        samp_end = 256 * 6
        data_samps = conditionally_sample(n_samp, steps, step_size, noise,
                                          data_valid_idx, samp_begin, samp_end)
        range_begin = 256 * 5
        range_end = 256 * (5 + 1)
        data_sub_samps = data_samps[:, range_begin:range_end]
        data_samps_mean = np.mean(data_sub_samps, axis=0)
        data_samps_std = np.std(data_sub_samps, axis=0)
        idx = np.random.randint(0, data_samps.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(5, 5), dpi=200, sharex=True)

        for i in range(data_sub_samps.shape[0]):
            axes[0].plot(xrange, data_sub_samps[i] * ptp[5] + mean[5], color=colors[-1], label="Sampled" if i == 0 else None)
        axes[0].plot(xrange, data_valid[data_valid_idx, range_begin:range_end] * ptp[5] + mean[5], color=colors[0], label="Real")
        axes[0].plot(xrange, data_samps_mean * ptp[5] + mean[5], color=colors[4], label="Sampled mean")
        for i in range(data_sub_samps.shape[0]):
            axes[1].plot(xrange, (data_sub_samps[i] - data_samps_mean) * ptp[5] + mean[5], color=colors[-1])
        axes[1].plot(xrange, (data_valid[data_valid_idx, range_begin:range_end] - data_samps_mean) * ptp[5] + mean[5], color=colors[0])
        axes[1].plot(xrange, np.zeros(256), color=colors[4])
        axes[0].set_title('Diode 1 signal (Volts)')
        axes[1].set_title('Real minus sampled')
        axes[1].set_xlabel('Time(ms)')
        axes[0].legend()
        plt.tight_layout()
        plt.savefig("experiments_msi/" + model_path + "/msi-single-" + model_num + ".png")

        plot_energy_histogram(data_samps)
        plt.savefig("experiments_msi/" + model_path + "/energies-single-" + model_num + ".png")
        plt.close('all')
    except Exception as e:
        print("Generating plots failed for " + f)
        print(e, e.args)
