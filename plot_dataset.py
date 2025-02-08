import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary  # Print model / parameter details

import numpy as np
from tqdm import tqdm
import wandb
import os
import datetime
import time
import shutil
import importlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani

from modular_ebm_diagnostics import *
from modular_ebm_sampling import *

# Pretty tracebacks
import rich.traceback
rich.traceback.install()

import sys
import signal
from multiprocessing import shared_memory as sm

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(['dark_background'])
mpl.rcParams['axes.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['figure.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
mpl.rcParams['legend.facecolor'] = (50 / 256, 50 / 256, 50 / 256)
mpl.rcParams['savefig.facecolor'] = (27 / 256, 27 / 256, 27 / 256)
data_color = (0.2, 0.7, 1.0, 1.0)
samp_color = (1.0, 0.7, 0.1, 0.6)
aux_color = (0, 0.9, 0.0, 0.6)
colors = ['#ff005ca0', '#eb6a7d', '#cd98a1', '#9dbfc6', '#00ff80ff', '#00e2ed0b']


def load_data_old(path):
    data_signals = np.load(path)['signals']
    # Encode positions later
    # data_positions = np.load(path)['positions']
    data_bad_table = np.load(path)['bad_table']

    complete_data = np.where(np.sum(data_bad_table, axis=1) == 0)[0]
    data_signals = data_signals[complete_data]
    data = torch.tensor(data_signals, dtype=torch.float)

    return data


def load_data(path):
    datafile = np.load(path)
    data_signals = datafile['signals']
    data = np.concatenate((data_signals['discharge_current'],
                           data_signals['discharge_voltage'],
                           data_signals['interferometer'],
                           data_signals['diode_0'],
                           data_signals['diode_1'],
                           data_signals['diode_2'],
                           data_signals['diode_3'],
                           data_signals['diode_4'],
                           data_signals['magnet_profile'],
                           np.nan_to_num(data_signals['pressures'], nan=-0.9)), axis=1)
    del data_signals

    enabled_mask = np.ones((data.shape[0], 10), dtype=bool)
    enabled_mask[datafile['discharge_current_cut'].astype('i4'), 0] = False
    enabled_mask[datafile['discharge_voltage_cut'].astype('i4'), 1] = False
    enabled_mask[datafile['interferometer_cut'].astype('i4'), 2] = False
    enabled_mask[datafile['diode_0_cut'].astype('i4'), 3] = False
    enabled_mask[datafile['diode_1_cut'].astype('i4'), 4] = False
    enabled_mask[datafile['diode_2_cut'].astype('i4'), 5] = False
    enabled_mask[datafile['diode_3_cut'].astype('i4'), 6] = False
    enabled_mask[datafile['diode_4_cut'].astype('i4'), 7] = False
    enabled_mask[datafile['magnet_profile_cut'].astype('i4'), 8] = False
    enabled_mask[datafile['pressures_cut'].astype('i4'), 9] = False
    data = np.concatenate((data, enabled_mask), axis=1)
    del enabled_mask

    data = torch.tensor(data, dtype=torch.float)

    return data


class ModularWithRNNBackbone(torch.nn.Module):
    def __init__(self):
        super(ModularWithRNNBackbone, self).__init__()

        # Set model sizes
        self.seq_length = seq_length = 32
        self.embed_dim = embed_dim = 128
        self.dense_width = dense_width = 128

        num_heads = 4
        num_hidden = 128
        num_msi_attn = 1
        # num_mem_attn = 3
        # num_sum_attn = 3
        energy_embed_dim = self.embed_dim
        energy_num_heads = 4
        energy_num_hidden = 128
        energy_num_attn = 2

        self.memory_iterations = 1

        # seq_length, num_msi_attn, num_mem_attn, num_sum_attn
        self.dishargeI = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                             num_msi_attn).cuda()
        self.dishargeV = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                             num_msi_attn).cuda()
        self.interferometer = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                                  num_msi_attn).cuda()
        self.diodes = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                          num_msi_attn).cuda()
        self.diode_HeII = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                              num_msi_attn).cuda()
        self.magnets = MagneticFieldModule(dense_width, seq_length, embed_dim, num_heads, num_hidden,
                                           num_msi_attn).cuda()
        self.RGA = RGAPressureModule(dense_width, seq_length, embed_dim, num_heads, num_hidden,
                                     num_msi_attn).cuda()

        self.seqPosEnc = SequencePositionalEncoding(energy_embed_dim, seq_length * 10).cuda()
        self.attnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length * 10, energy_embed_dim, energy_num_heads, energy_num_hidden).cuda()
            for i in range(energy_num_attn)])

        self.realCoordEnc = DiagnosticPositionModule(3, 32, seq_length).cuda()

        # kernel_size = 4
        # self.dishargeI = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.dishargeV = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.interferometer = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.diodes = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.diode1 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.diode2 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.diode3 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.diode_HeII = MSICNNModule(seq_length, embed_dim, kernel_size)
        # self.magnets = RGADenseModule(seq_length, 64, embed_dim, kernel_size)
        # self.RGA = RGADenseModule(seq_length, 64, embed_dim, kernel_size)

        # self.memBlock1 = CNNResidualBlock(seq_length, embed_dim, kernel_size)
        # self.memBlock2 = CNNResidualBlock(seq_length, embed_dim, kernel_size)

        self.softmax = torch.nn.Softmax(dim=1)  # softmax over the seq_length
        self.linear = torch.nn.LazyLinear(1)

    def forward(self, x):
        device = getDeviceString(x.get_device())

        I_x = x[:, 32 * 0:32 * 1]
        V_x = x[:, 32 * 1:32 * 2]
        n_x = x[:, 32 * 2:32 * 3]
        d0_x = x[:, 32 * 3:32 * 4]
        d1_x = x[:, 32 * 4:32 * 5]
        d2_x = x[:, 32 * 5:32 * 6]
        d3_x = x[:, 32 * 6:32 * 7]
        d4_x = x[:, 32 * 7:32 * 8]
        B_x = x[:, 32 * 8:32 * 8 + 64]
        p_x = x[:, 32 * 8 + 64:32 * 8 + 64 + 51]

        key_mask = torch.reshape(x[:, -10:], (-1, 10, 1)).repeat((1, 1, self.seq_length))
        key_mask = key_mask.reshape((-1, 10 * self.seq_length))

        batch_size = I_x.shape[0]

        d0_pos = self.realCoordEnc(torch.tensor([0.12, ]).to(device))
        d1_pos = self.realCoordEnc(torch.tensor([0.22, ]).to(device))
        d2_pos = self.realCoordEnc(torch.tensor([0.34, ]).to(device))
        d3_pos = self.realCoordEnc(torch.tensor([0.46, ]).to(device))
        d4_pos = self.realCoordEnc(torch.tensor([0.10, ]).to(device))

        # shared_memory = torch.zeros((batch_size, self.seq_length, self.embed_dim)).to(device)
        # for i in range(self.memory_iterations):
        shared_memory = torch.cat([self.dishargeI(I_x),
                                   self.dishargeV(V_x),
                                   self.interferometer(n_x),
                                   self.diodes(d0_x + d0_pos),
                                   self.diodes(d1_x + d1_pos),
                                   self.diodes(d2_x + d2_pos),
                                   self.diodes(d3_x + d3_pos),
                                   self.diode_HeII(d4_x + d4_pos),
                                   self.magnets(B_x),
                                   self.RGA(p_x)], dim=1)

        shared_memory = self.seqPosEnc(shared_memory)
        for i, block in enumerate(self.attnBlocks):
            if i == 0:
                shared_memory = block(shared_memory, key_mask=key_mask)
            else:
                shared_memory = block(shared_memory)

        # shared_memory = self.memBlock1(shared_memory_temp)
        # shared_memory = self.memBlock2(shared_memory)

        shared_memory = self.softmax(shared_memory)
        shared_memory = self.linear(shared_memory.reshape(batch_size, -1))

        return shared_memory


# A comprehensive guide to distributed data parallel:
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '24163'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)
    sh_mem = sm.SharedMemory(name="exit_mem")
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    project_name = "modular_ebm"
    exp_path = "experiments_modular/"
    path = exp_path + identifier
    if rank == 0:
        os.mkdir(path)
        os.mkdir(path + "/checkpoints")
        os.mkdir(path + "/plots")
        shutil.copy(project_name + ".py", path + "/" + project_name + "_copy.py")
        shutil.copy(project_name + "_sampling.py", path + "/" + project_name + "_sampling.py")
        shutil.copy(project_name + "_diagnostics.py", path + "/" + project_name + "_diagnostics.py")

    hyperparams = {
        "num_epochs": 1,
        "reg_amount": 1e0,
        "replay_frac": 0.99,
        "replay_size": 8192,

        "sample_steps": 50,
        "step_size": 1e2,
        "noise_scale": 5e-3,
        "augment_data": True,

        "batch_size_max": 42,
        "lr": 1.5e-5,

        "kl_weight_energy": 1e0,
        "kl_weight_entropy": 1e0,
        "kl_backprop_steps": 1,

        "weight_decay": 1e-1,
        "identifier": identifier,
        "resume": False,
        # "resume_path": "2022-08-22_22h-29m-07s",
        # "resume_version": "checkpoints/model-0-7672"
    }

    num_epochs = hyperparams["num_epochs"]
    reg_amount = hyperparams["reg_amount"]
    replay_frac = hyperparams["replay_frac"]
    replay_size = hyperparams["replay_size"]
    sample_steps = hyperparams["sample_steps"]
    step_size = hyperparams["step_size"]
    noise_scale = hyperparams["noise_scale"]
    augment_data = hyperparams["augment_data"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]
    kl_weight_energy = hyperparams["kl_weight_energy"]
    kl_weight_entropy = hyperparams["kl_weight_entropy"]
    weight_decay = hyperparams["weight_decay"]
    resume = hyperparams["resume"]
    if resume:
        resume_path = hyperparams["resume_path"]
        resume_version = hyperparams["resume_version"]

    writer = SummaryWriter(log_dir=path)
    model = ModularWithRNNBackbone()
    if resume:
        spec = importlib.util.spec_from_file_location(project_name + "_copy", exp_path +
                                                      resume_path + "/" + project_name + "_copy.py")
        ebm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ebm)
        model = ebm.ModularWithRNNBackbone()

    if rank == 0:
        print("Loading data: " + path, flush=True)
    data_path = "data/data-MSI-all_2022-5-22_17-6-24-train.npz"
    data = load_data(data_path)
    num_examples = data.shape[0]
    data_size = data.shape[1]
    if rank == 0:
        print("Data shape: ", end="")
        print(data.shape, flush=True)

    # initialze for lazy layers so that the num_parameters works properly
    model = model.to(rank)
    model(torch.zeros((2, data_size)).to(rank))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size,
                                                              rank=rank, shuffle=True,
                                                              drop_last=False)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                             batch_size=batch_size_max, shuffle=False,
                                             num_workers=4, pin_memory=True, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.0, 0.999))
    del data

    replay_buffer_init_data = torch.cat((torch.rand((replay_size, data_size - 10)) * 2 - 1,
                                         torch.ones((replay_size, 10))), dim=1).to(rank)
    replay_buffer = ReplayBuffer(replay_size, replay_buffer_init_data)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt", map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        replay_buffer.sample_list = ckpt['replay_buffer_list']

    num_batches = len(dataloader)

    # Warmup schedule for the model
    def lr_func(x):
        if x <= 60:
            return 1.0
        else:
            return 1 / torch.sqrt(torch.tensor(x - 60)).to(rank)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    if resume:
        lrScheduler.load_state_dict(ckpt['lrScheduler_state_dict'])

    if rank == 0:
        # summary(model, (data_size,), batch_size=batch_size_max)
        num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("Parameters: {}".format(num_parameters))
        hyperparams['num_parameters'] = num_parameters
        wandb.init(project="modular-ebm", entity='phil',
                   group="", job_type="",
                   config=hyperparams)

    t_start0 = t_start1 = t_start2 = t_start_autoclose = time.time()
    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    if resume:
        batch_iteration = ckpt['batch_iteration']
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

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

    ptp = np.load(data_path)['scale']

    for epoch in range(num_epochs):
        # with torch.cuda.amp.autocast():s
        sampler.set_epoch(epoch)
        loss_avg = 0
        reg_avg = 0
        kl_loss_avg = 0

        energy_list_size = batch_size_max * 4 if num_examples > batch_size_max * 4 else num_examples
        energy_pos_list = torch.zeros((energy_list_size, 1)).to(rank)
        energy_neg_list = torch.zeros((energy_list_size, 1)).to(rank)
        energy_kl_list = torch.zeros((energy_list_size, 1)).to(rank)

        batch_pbar = tqdm(total=num_batches)
        for pos_x, i in zip(dataloader, range(num_batches)):
            # So that data that's been seen before is skipped and get to the right number of batch iterations
            

            try:
                pos_x = pos_x[0].to(rank)
                data_samps = pos_x.detach().cpu().numpy()
                # data_samps_mean = np.mean(data_sub_samps, axis=0)
                # data_samps_std = np.std(data_sub_samps, axis=0)

                fig, axes = plt.subplots(5, 2, figsize=(8, 7), dpi=200)

                for ax in axes.flatten():
                    ax.set_autoscale_on(False)

                for k in range(data_samps.shape[0]):
                    axes[0, 0].plot(xrange, data_samps[k, 32 * 0:32 * 1] * ptp['discharge_current'], color=colors[-1], label="Sampled")
                    axes[0, 1].plot(xrange, data_samps[k, 32 * 1:32 * 2] * ptp['discharge_voltage'], color=colors[-1])
                    axes[1, 0].plot(xrange, data_samps[k, 32 * 2:32 * 3] * ptp['interferometer'], color=colors[-1])
                    axes[1, 1].plot(xrange, data_samps[k, 32 * 3:32 * 4] * ptp['diode_0'], color=colors[-1])
                    axes[2, 0].plot(xrange, data_samps[k, 32 * 4:32 * 5] * ptp['diode_1'], color=colors[-1])
                    axes[2, 1].plot(xrange, data_samps[k, 32 * 5:32 * 6] * ptp['diode_2'], color=colors[-1])
                    axes[3, 0].plot(xrange, data_samps[k, 32 * 6:32 * 7] * ptp['diode_3'], color=colors[-1])
                    axes[3, 1].plot(xrange, data_samps[k, 32 * 7:32 * 8] * ptp['diode_4'], color=colors[-1])
                    axes[4, 0].plot(B_xrange, data_samps[k, 32 * 8:32 * 8 + 64] * ptp['magnet_profile'], color=colors[-1])
                    axes[4, 1].scatter(np.arange(51), data_samps[k, 32 * 8 + 64:32 * 8 + 64 + 51] * ptp['pressures'], color=colors[-1])

                for ax in axes.flatten():
                    ax.set_autoscale_on(True)

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
                plt.savefig(path + '/plots/data-batch-{}.png'.format(i))
                plt.close()

            except Exception as e:
                print(e)

            # if i + epoch * num_batches < batch_iteration:
            #     batch_pbar.update(1)
            #     continue
            # # print("Rank: {}\tBatchnum: {}".format(rank, i))
            # # with torch.autograd.detect_anomaly():
            # with torch.cuda.amp.autocast(enabled=False):
            #     optimizer.zero_grad()
            #     pos_x = pos_x[0].to(rank)
            #     pos_x.requires_grad = True
            #     batch_size = pos_x.shape[0]

            #     # print(pos_x[0, -11:])

            #     neg_x = replay_buffer.sample(int(batch_size * replay_frac))
            #     if augment_data:
            #         neg_x = perturb_samples(neg_x)

            #     # print(neg_x[0, -11:])

            #     # Refill part of the replay buffer with random data
            #     neg_x_rand = torch.cat((torch.rand(batch_size - neg_x.shape[0], pos_x.shape[1] - 10) * 2 - 1,
            #                             torch.ones((batch_size - neg_x.shape[0], 10))), dim=1).to(rank)
            #     neg_x = torch.cat([neg_x, neg_x_rand], 0)
            #     # neg_x = torch.Tensor(neg_x).to(rank)
            #     # neg_x = torch.Tensor(neg_x)
            #     # neg_x.requires_grad = True  # Needed if not using Langevin_KL sampling

            #     # print(neg_x[-1, -11:])

            #     # For calculating the KL loss later
            #     num_kl_samples = 100
            #     kl_samp = replay_buffer.sample(num_kl_samples)

            #     # print(kl_samp[0, -11:])

            #     # ------ PUT DIAGNOSTIC DROPOUT FOR SAMPLES HERE ------ #

            #     # Run Langevin dynamics on sample
            #     neg_x, kl_x = sample_langevin_KL_cuda(neg_x, model, sample_steps=sample_steps,
            #                                           kl_backprop_steps=hyperparams["kl_backprop_steps"],
            #                                           step_size=step_size, noise_scale=noise_scale)
            #     replay_buffer.add(neg_x)  # neg_x already detached in Langevin call above

            #     # print(neg_x[0, -11:])
            #     # print(kl_x[0, -11:])

            #     # KL loss -- energy part
            #     # Don't accumulate grads in the model parameters for the KL loss
            #     model.requires_grad_(False)
            #     kl_energy = model(kl_x)
            #     model.requires_grad_(True)
            #     kl_loss = kl_energy.mean() * kl_weight_energy

            #     # KL loss -- entropy part
            #     # This uses a nearest-neighbor estimation of the entropy of Langevin'd samples
            #     kl_x = kl_x.view(batch_size, -1)
            #     kl_entropy = kl_x[:, None, :] - kl_samp[None, :, :]
            #     kl_entropy = torch.norm(kl_entropy, p=2, dim=2)
            #     kl_entropy = torch.min(kl_entropy, dim=1)[0]  # Min returns a tuple
            #     kl_entropy = -torch.log(kl_entropy + 1e-8).mean()  # Technically missing + ln(2) + C_E
            #     kl_loss += kl_entropy * kl_weight_entropy

            #     # Backwards pass...
            #     optimizer.zero_grad()
            #     pos_energy = model(pos_x)
            #     neg_energy = model(neg_x.to(rank))
            #     # neg_energy = model(neg_x)
            #     energy_regularization = reg_amount * (pos_energy.square() + neg_energy.square()).mean()

            #     loss = ((pos_energy - neg_energy).mean() + energy_regularization + kl_loss)
            # # loss.backward()
            # scaler.scale(loss).backward()

            # # For debugging missing gradients error
            # # for name, p in model.named_parameters():
            # #     if p.grad is None:
            # #         print("found unused param: ")
            # #         print(name)
            # #         print(p)
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # # optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # lrScheduler.step()

            # # print(linearScheduler.get_last_lr())
            # # print(sqrtScheduler.get_last_lr())

            # if rank == 0:
            #     loss_avg += loss.detach() * batch_size / num_examples
            #     reg_avg += energy_regularization.detach() * batch_size / num_examples
            #     # print("\n")
            #     # print(reg_avg)
            #     kl_loss_avg += kl_loss.detach() * batch_size / num_examples

            #     pos_energy = pos_energy.detach()
            #     neg_energy = neg_energy.detach()
            #     kl_energy = kl_energy.detach()
            #     kl_loss = kl_loss.detach()
            #     loss = loss.detach()
            #     energy_regularization = energy_regularization.detach()

            #     # Restricting size *dramatically* improves performance
            #     if i < 4 and energy_list_size >= batch_size_max * i + batch_size:
            #         energy_pos_list[i * batch_size_max:i * batch_size_max + batch_size] = pos_energy
            #         energy_neg_list[i * batch_size_max:i * batch_size_max + batch_size] = neg_energy
            #         energy_kl_list[i * batch_size_max:i * batch_size_max + batch_size] = kl_energy

            #     pos_energy = pos_energy.mean()
            #     neg_energy = neg_energy.mean()
            #     kl_energy = kl_energy.mean()
            #     if i % 20 == 0:
            #         tqdm.write("#: {} // L: {:.2e} // (+): {:.2e} // (-): {:.2e} // "
            #                    "(-)-(+): {:.2e}".format(i, loss, pos_energy,
            #                                             neg_energy,
            #                                             neg_energy - pos_energy))
            #     wandb.log({"loss/total": loss,
            #                "energy/reg": energy_regularization,
            #                "energy/positive": pos_energy,
            #                "energy/negative": neg_energy,
            #                "energy/negative_relative": neg_energy - pos_energy,
            #                "energy/kl_energy": kl_energy,
            #                "loss/kl_loss": kl_loss,
            #                "loss/max_likelihood": loss - kl_loss,
            #                "batch_num": batch_iteration,
            #                "epoch": epoch})

            # # Longer-term metrics
            # if rank == 0:
            #     # End training after fixed amount of time
            #     if time.time() - t_start_autoclose > 3600 * 120:
            #         sh_mem.buf[0] = 1

            #     # scalars
            #     avg_energy_pos = energy_pos_list.mean()
            #     avg_energy_neg = energy_neg_list.mean()
            #     avg_energy_kl = energy_kl_list.mean()

            #     # histograms
            #     energy_pos_list -= avg_energy_pos
            #     energy_neg_list -= avg_energy_pos
            #     energy_kl_list -= avg_energy_kl

            #     # Log to tensorboard every 5 min
            #     if (epoch == 0 and i == 3) or time.time() - t_start0 > 300 or sh_mem.buf[0] == 1:
            #         t_start0 = time.time()
            #         # write scalars and histograms
            #         writer.add_scalar("loss/total", loss_avg, batch_iteration)
            #         writer.add_scalar('energy/reg', reg_avg, batch_iteration)
            #         writer.add_scalar("energy/positive", avg_energy_pos, batch_iteration)
            #         writer.add_scalar("energy/negative", avg_energy_neg, batch_iteration)
            #         writer.add_scalar("energy/negative_relative", avg_energy_neg - avg_energy_pos, batch_iteration)
            #         writer.add_scalar("energy/kl_energy", avg_energy_kl, batch_iteration)
            #         writer.add_scalar("loss/kl_loss", kl_loss_avg, batch_iteration)
            #         writer.add_scalar("loss/max_likelihood", loss_avg - kl_loss_avg, batch_iteration)
            #     # wandb.log({"loss/total": loss_avg,
            #     #            "energy/reg": reg_avg,
            #     #            "energy/positive": avg_energy_pos,
            #     #            "energy/negative": avg_energy_neg,
            #     #            "energy/negative_relative": avg_energy_neg - avg_energy_pos,
            #     #            "energy/kl_energy": avg_energy_kl,
            #     #            "loss/kl_loss": kl_loss_avg,
            #     #            "loss/max_likelihood": loss_avg - kl_loss_avg,
            #     #            "epoch": epoch})

            #     # Add histogram every 20 min
            #     if (epoch == 0 and i == 3) or time.time() - t_start1 > 1200 or sh_mem.buf[0] == 1:
            #         t_start1 = time.time()
            #         try:
            #             writer.add_histogram("energy/pos_relative", energy_pos_list, batch_iteration)
            #             writer.add_histogram("energy/neg_relative", energy_neg_list, batch_iteration)
            #             writer.add_histogram("energy_kl_list", energy_kl_list, batch_iteration)
            #         except Exception as e:
            #             print(e)
            #         try:
            #             for name, weight in model.named_parameters():
            #                 writer.add_histogram("w/" + name, weight, batch_iteration)
            #                 writer.add_histogram(f'g/{name}.grad', weight.grad, batch_iteration)
            #         except Exception as e:
            #             print(e)
            #         writer.flush()
            #         tqdm.write("E: {} // L: {:.2e} // (+): {:.2e} // (-): {:.2e} // "
            #                    "(-)-(+): {:.2e}".format(epoch, loss_avg, avg_energy_pos,
            #                                             avg_energy_neg,
            #                                             avg_energy_neg - avg_energy_pos))

            #     # Save checkpoint every hour
            #     if ((epoch == 0 and i == 3) or (epoch == num_epochs - 1 and i == num_batches - 1)
            #         or time.time() - t_start2 > 3600 or sh_mem.buf[0] == 1):
            #         t_start2 = time.time()
            #         torch.save({'epoch': epoch,
            #                     'batch_iteration': batch_iteration,
            #                     'model_state_dict': model.state_dict(),
            #                     'optimizer_state_dict': optimizer.state_dict(),
            #                     'lrScheduler_state_dict': lrScheduler.state_dict(),
            #                     'replay_buffer_list': replay_buffer.sample(128)
            #                     },
            #                    path + "/checkpoints/model-{}-{}.pt".format(epoch, i))

            batch_iteration += 1
            batch_pbar.update(1)

            if sh_mem.buf[0] == 1:
                print("\n ---- Exiting process ----\n")
                break
        batch_pbar.close()
        pbar.update(1)
        if sh_mem.buf[0] == 1:
            break
    pbar.close()
    sh_mem.close()
    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    world_size = 1
    sh_mem = sm.SharedMemory(name="exit_mem", create=True, size=1)
    sh_mem.buf[0] = 0
    try:
        proc_context = torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=False)
        proc_context.join()
    except KeyboardInterrupt:
        sh_mem.buf[0] = 1
        proc_context.join(timeout=30)
        sh_mem.unlink()
    else:
        sh_mem.unlink()
