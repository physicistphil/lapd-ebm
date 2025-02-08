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
import argparse
import json
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani

from modular_ebm_diagnostics import *
from modular_ebm_sampling import *

# Pretty tracebacks
# import rich.traceback
# rich.traceback.install()

import sys
import signal
from multiprocessing import shared_memory as sm


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

    disabled_mask = np.zeros((data.shape[0], 10), dtype=bool)
    disabled_mask[datafile['discharge_current_cut'].astype('i4'), 0] = True
    disabled_mask[datafile['discharge_voltage_cut'].astype('i4'), 1] = True
    disabled_mask[datafile['interferometer_cut'].astype('i4'), 2] = True
    disabled_mask[datafile['diode_0_cut'].astype('i4'), 3] = True
    disabled_mask[datafile['diode_1_cut'].astype('i4'), 4] = True
    disabled_mask[datafile['diode_2_cut'].astype('i4'), 5] = True
    disabled_mask[datafile['diode_3_cut'].astype('i4'), 6] = True
    disabled_mask[datafile['diode_4_cut'].astype('i4'), 7] = True
    disabled_mask[datafile['magnet_profile_cut'].astype('i4'), 8] = True
    disabled_mask[datafile['pressures_cut'].astype('i4'), 9] = True
    data = np.concatenate((data, disabled_mask), axis=1)
    del disabled_mask

    data = torch.tensor(data, dtype=torch.float)

    return data


class ModularWithRNNBackbone(torch.nn.Module):
    def __init__(self, hyperparams):
        super(ModularWithRNNBackbone, self).__init__()

        # Set model sizes
        # self.seq_length = seq_length = hyperparams["seq_length"]
        # self.embed_dim = out_channels = embed_dim = hyperparams["embed_dim"]
        # self.dense_width = dense_width = hyperparams["dense_width"]

        # num_heads = hyperparams["num_heads"]
        # num_hidden = hyperparams["num_hidden"]
        # self.num_msi_attn = num_msi_attn = hyperparams["num_msi_attn"]
        # # num_mem_attn = 3
        # # num_sum_attn = 3
        # energy_embed_dim = self.embed_dim
        # energy_num_heads = hyperparams["energy_num_heads"]
        # energy_num_hidden = hyperparams["energy_num_hidden"]
        # energy_num_attn = hyperparams["energy_num_attn"]

        # # self.realCoordEnc = DiagnosticPositionModule(3, 32, seq_length).cuda()

        # # kernel_size = 4
        # # self.dishargeI = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.dishargeV = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.interferometer = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.diode0 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.diode1 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.diode2 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.diode3 = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.diode_HeII = MSICNNModule(seq_length, embed_dim, kernel_size)
        # # self.magnets = MagneticCNNModule(seq_length, embed_dim, kernel_size)
        # # self.RGA = RGADenseModule(seq_length, 64, embed_dim, kernel_size)

        # # self.seqPosEnc = SequencePositionalEncoding(embed_dim * 10 + 8, seq_length).cuda()

        # # self.combiner = CombinedDiagnosticsCNN(seq_length, embed_dim, kernel_size)

        #  # seq_length, num_msi_attn, num_mem_attn, num_sum_attn

        # # Build embedding for each diagnostic
        # kernel_size = 4
        # self.dishargeI_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.dishargeV_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.interferometer_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.diode0_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.diode1_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.diode2_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.diode3_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.diode_HeII_embed = CNNDenseNetBlock(seq_length, embed_dim, kernel_size)
        # self.magnets_embed = MagneticCNNModule(seq_length, embed_dim, kernel_size)
        # self.RGA_embed = RGADenseModule(seq_length, 64, seq_length * embed_dim, kernel_size)

        # self.dishargeI = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.dishargeV = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.interferometer = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.diode0 = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.diode1 = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.diode2 = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.diode3 = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.diode_HeII = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.magnets = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])
        # self.RGA = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
        #     for i in range(num_msi_attn)])

        # self.seqPosEnc = SequencePositionalEncoding(energy_embed_dim, seq_length * 10).cuda()
        # self.attnBlocks = torch.nn.ModuleList([
        #     ResidualAttnBlock(seq_length * 10, energy_embed_dim, energy_num_heads, energy_num_hidden).cuda()
        #     for i in range(energy_num_attn)])

        # # self.realCoordEnc = DiagnosticPositionModule(3, 32, seq_length).cuda()

        # self.softmax = torch.nn.Softmax(dim=2)  # dim=1: seq_length, dim=2: embedding
        self.hidden = torch.nn.LazyLinear(1024)
        self.relu = torch.nn.Tanh()
        self.linear = torch.nn.LazyLinear(1)


    def forward(self, x):
        device = getDeviceString(x.get_device())

        # I_x = x[:, 32 * 0:32 * 1]
        # V_x = x[:, 32 * 1:32 * 2]
        # n_x = x[:, 32 * 2:32 * 3]
        # d0_x = x[:, 32 * 3:32 * 4]
        # d1_x = x[:, 32 * 4:32 * 5]
        # d2_x = x[:, 32 * 5:32 * 6]
        # d3_x = x[:, 32 * 6:32 * 7]
        # d4_x = x[:, 32 * 7:32 * 8]
        # B_x = x[:, 32 * 8:32 * 8 + 64]
        # p_x = x[:, 32 * 8 + 64:32 * 8 + 64 + 51]

        # key_mask = torch.reshape(x[:, -10:], (-1, 10, 1)).repeat((1, 1, self.seq_length))
        # key_mask = key_mask.reshape((-1, 10 * self.seq_length))

        # batch_size = I_x.shape[0]
        # # print(self.dishargeI_embed(torch.unsqueeze(I_x, 1)).transpose(1, 2).shape)
        # # print(self.magnets_embed(B_x).shape)
        # # print(self.magnets_embed(B_x).reshape(-1, self.seq_length, self.embed_dim).shape)
        # # print(self.RGA_embed(p_x).unsqueeze(2).reshape(-1, self.seq_length, self.embed_dim).shape)

        # I_x = self.dishargeI_embed(torch.unsqueeze(I_x, 1)).transpose(1, 2)
        # V_x = self.dishargeV_embed(torch.unsqueeze(V_x, 1)).transpose(1, 2)
        # n_x = self.interferometer_embed(torch.unsqueeze(n_x, 1)).transpose(1, 2)
        # d0_x = self.diode0_embed(torch.unsqueeze(d0_x, 1)).transpose(1, 2)
        # d1_x = self.diode1_embed(torch.unsqueeze(d1_x, 1)).transpose(1, 2)
        # d2_x = self.diode2_embed(torch.unsqueeze(d2_x, 1)).transpose(1, 2)
        # d3_x = self.diode3_embed(torch.unsqueeze(d3_x, 1)).transpose(1, 2)
        # d4_x = self.diode_HeII_embed(torch.unsqueeze(d4_x, 1)).transpose(1, 2)
        # B_x = self.magnets_embed(B_x).reshape(-1, self.seq_length, self.embed_dim)
        # p_x = self.RGA_embed(p_x).reshape(-1, self.seq_length, self.embed_dim)

        # # print(I_x.shape)
        # # print(B_x.shape)
        # # print(p_x.shape)

        # all_diagnostics = torch.cat([I_x,
        #                              V_x,
        #                              n_x,
        #                              d0_x,
        #                              d1_x,
        #                              d2_x,
        #                              d3_x,
        #                              d4_x,
        #                              B_x,
        #                              p_x],
        #                             dim=1)
        # # print(all_diagnostics.shape)

        # for i in range(self.num_msi_attn):
        #     I_x = self.dishargeI[i](I_x, x_full=all_diagnostics)
        #     V_x = self.dishargeV[i](V_x, x_full=all_diagnostics)
        #     n_x = self.interferometer[i](n_x, x_full=all_diagnostics)
        #     d0_x = self.diode0[i](d0_x, x_full=all_diagnostics)
        #     d1_x = self.diode1[i](d1_x, x_full=all_diagnostics)
        #     d2_x = self.diode2[i](d2_x, x_full=all_diagnostics)
        #     d3_x = self.diode3[i](d3_x, x_full=all_diagnostics)
        #     d4_x = self.diode_HeII[i](d4_x, x_full=all_diagnostics)
        #     B_x = self.magnets[i](B_x, x_full=all_diagnostics)
        #     p_x = self.RGA[i](p_x, x_full=all_diagnostics)

        #     all_diagnostics = torch.cat([I_x,
        #                                  V_x,
        #                                  n_x,
        #                                  d0_x,
        #                                  d1_x,
        #                                  d2_x,
        #                                  d3_x,
        #                                  d4_x,
        #                                  B_x,
        #                                  p_x],
        #                                 dim=1)

        # # shape should be (batch size, out_channels * 10, seq_length)

        # # So it knows where it is in the sequence
        # # all_diagnostics = self.seqPosEnc(all_diagnostics.transpose(1, 2)).transpose(1, 2)
        # all_diagnostics = self.seqPosEnc(all_diagnostics)
        # # all_diagnostics = self.combiner(all_diagnostics)

        # for i, block in enumerate(self.attnBlocks):
        #     all_diagnostics = block(all_diagnostics)

        # all_diagnostics = self.softmax(all_diagnostics)
        batch_size = x.shape[0]
        all_diagnostics = x[0:32 * 8 + 64 + 51]
        all_diagnostics = self.hidden(all_diagnostics)
        all_diagnostics = self.relu(all_diagnostics)
        all_diagnostics = self.linear(all_diagnostics.reshape(batch_size, -1))

        return all_diagnostics


# A comprehensive guide to distributed data parallel:
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def main(rank, world_size, hyperparams, port):
    setup(rank, world_size, port)
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getppid()),)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    hyperparams['identifier'] = identifier
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

        with open(path + "/" + "hyperparams.json", 'w') as json_f:
            json.dump(hyperparams, json_f)

    num_epochs = hyperparams["num_epochs"]
    reg_amount = hyperparams["reg_amount"]
    replay_frac = hyperparams["replay_frac"]
    replay_size = hyperparams["replay_size"]
    replay_cyclic = hyperparams["replay_cyclic"]
    sample_steps = hyperparams["sample_steps"]
    step_size = hyperparams["step_size"]
    noise_scale = hyperparams["noise_scale"]
    augment_data = hyperparams["augment_data"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]
    momentum = hyperparams["momentum"]
    kl_weight_energy = hyperparams["kl_weight_energy"]
    kl_weight_entropy = hyperparams["kl_weight_entropy"]
    weight_decay = hyperparams["weight_decay"]
    resume = hyperparams["resume"]
    if resume:
        resume_path = hyperparams["resume_path"]
        resume_version = hyperparams["resume_version"]

    writer = SummaryWriter(log_dir=path)
    model = ModularWithRNNBackbone(hyperparams)
    if resume:
        with open(exp_path + resume_path + "/" + "hyperparams.json") as json_f:
            hyperparams_temp = json.loads(json_f.read())
        spec = importlib.util.spec_from_file_location(project_name + "_copy", exp_path +
                                                      resume_path + "/" + project_name + "_copy.py")
        ebm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ebm)
        model = ebm.ModularWithRNNBackbone(hyperparams_temp)

    if rank == 0:
        print("Loading data: " + path, flush=True)
    data_path = "data/" + hyperparams["dataset"]
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
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum, nesterov=False)
    switch_opt_flag = True
    del data

    replay_buffer_init_data = torch.cat((torch.rand((replay_size, data_size - 10)) * 2 - 1,
                                         torch.ones((replay_size, 10))), dim=1).to(rank)
    replay_buffer = ReplayBuffer(replay_size, replay_buffer_init_data)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt", map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        ckpt_buffer_list = ckpt['replay_buffer_list']
        replay_buffer.sample_list[0:len(ckpt_buffer_list)] = ckpt['replay_buffer_list']

    num_batches = len(dataloader)

    # Warmup schedule for the model
    def lr_func(x):
        # return 1.0
        if x <= 500:
            return 1.0
        else:
            return 1 / torch.sqrt(torch.tensor(x - 500)).to(rank)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    if resume:
        lrScheduler.load_state_dict(ckpt['lrScheduler_state_dict'])

    if rank == 0:
        # summary(model, (data_size,), batch_size=batch_size_max)
        num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("Parameters: {}".format(num_parameters))
        for name, module in model.named_modules():
            print(name, sum(param.numel() for param in module.parameters()))

        hyperparams['num_parameters'] = num_parameters
        wandb.init(project="modular-ebm", entity='phil',
                   group="", job_type="",
                   config=hyperparams)

    t_start0 = t_start1 = t_start2 = t_start_autoclose = time.time()
    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    grad_mag_list = []
    if resume:
        batch_iteration = ckpt['batch_iteration']
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
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
            # print("Rank: {}\tBatchnum: {}".format(rank, i))
            # with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast(enabled=False):
                optimizer.zero_grad()
                pos_x = pos_x[0].to(rank)
                pos_x.requires_grad = True
                batch_size = pos_x.shape[0]

                # print(pos_x[0, -11:])

                if replay_cyclic:
                    with torch.no_grad():
                        replay_buffer.sample_list[batch_iteration % replay_size] = torch.cat((torch.rand(pos_x.shape[1] - 10) * 4 - 2,
                                                                  torch.ones(10)), dim=0).to(rank)
                    if batch_size == batch_size_max:
                        neg_x = replay_buffer.sample_list
                    else:
                        neg_x = replay_buffer.sample_list[batch_size_max - batch_size:batch_size_max] 

                else:
                    # This needs to be fixed to account for the thrown away fraction when rounded
                    neg_x = replay_buffer.sample(int(batch_size * replay_frac))
                    if augment_data:
                        neg_x = perturb_samples(neg_x)

                    # Refill part of the replay buffer with random data
                    neg_x_rand = torch.cat((torch.rand(batch_size - neg_x.shape[0], pos_x.shape[1] - 10) * 2 - 1,
                                            torch.ones((batch_size - neg_x.shape[0], 10))), dim=1).to(rank)
                    neg_x = torch.cat([neg_x, neg_x_rand], 0)
                    # neg_x = torch.Tensor(neg_x).to(rank)
                    # neg_x = torch.Tensor(neg_x)
                    # neg_x.requires_grad = True  # Needed if not using Langevin_KL sampling

#                 print(neg_x.shape)
                
                # For calculating the KL loss later
                num_kl_samples = 100
                kl_samp = replay_buffer.sample(num_kl_samples)

                # ------ PUT DIAGNOSTIC DROPOUT FOR SAMPLES HERE ------ #

                # Run Langevin dynamics on sample
                neg_x, kl_x, avg_grad_mag, avg_noise_mag = sample_langevin_KL_cuda(neg_x, model, sample_steps=sample_steps,
                                                      kl_backprop_steps=hyperparams["kl_backprop_steps"],
                                                      step_size=step_size, noise_scale=noise_scale)
                if replay_cyclic:
                    with torch.no_grad():
                        replay_buffer.sample_list[batch_size_max - batch_size:batch_size_max] = neg_x
                else:
                    replay_buffer.add(neg_x)  # neg_x already detached in Langevin call above

                # avg_grad_mag = avg_grad_mag.cpu().numpy()
                # avg_noise_mag = avg_noise_mag.cpu().numpy()

                # KL loss -- energy part
                # Don't accumulate grads in the model parameters for the KL loss
                model.requires_grad_(False)
                kl_energy = model(kl_x)
                model.requires_grad_(True)
                kl_loss = kl_energy.mean() * kl_weight_energy

                # KL loss -- entropy part
                # This uses a nearest-neighbor estimation of the entropy of Langevin'd samples
                kl_x = kl_x.view(batch_size, -1)
                kl_entropy = kl_x[:, None, :] - kl_samp[None, :, :]
                kl_entropy = torch.norm(kl_entropy, p=2, dim=2)
                kl_entropy = torch.min(kl_entropy, dim=1)[0]  # Min returns a tuple
                kl_entropy = -torch.log(kl_entropy + 1e-8).mean()  # Technically missing + ln(2) + C_E
                kl_loss += kl_entropy * kl_weight_entropy

                # Backwards pass...
                optimizer.zero_grad()

                # pos_noise = torch.cat((torch.randn_like(pos_x[:, :-10]) * step_size, torch.zeros_like(pos_x[:, :10])), dim=1)
                # pos_energy = model(pos_x + pos_noise * step_size)
                pos_energy = model(pos_x)
                neg_energy = model(neg_x.to(rank))
                # neg_energy = model(neg_x)
                energy_regularization = reg_amount * (pos_energy.square() + neg_energy.square()).mean()

                loss = ((pos_energy.mean() - neg_energy.mean()) + energy_regularization + kl_loss)
            # loss.backward()
            scaler.scale(loss).backward()

            # For debugging missing gradients error
            # for name, p in model.named_parameters():
            #     if p.grad is None:
            #         print("found unused param: ")
            #         print(name)
            #         print(p)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            lrScheduler.step()

            # print(linearScheduler.get_last_lr())
            # print(sqrtScheduler.get_last_lr())

            if rank == 0:
                loss_avg += loss.detach() * batch_size / num_examples
                reg_avg += energy_regularization.detach() * batch_size / num_examples
                # print("\n")
                # print(reg_avg)
                kl_loss_avg += kl_loss.detach() * batch_size / num_examples

                pos_energy = pos_energy.detach()
                neg_energy = neg_energy.detach()
                kl_energy = kl_energy.detach()
                kl_loss = kl_loss.detach()
                loss = loss.detach()
                energy_regularization = energy_regularization.detach()

                # Restricting size *dramatically* improves performance
                if i < 4 and energy_list_size >= batch_size_max * i + batch_size:
                    energy_pos_list[i * batch_size_max:i * batch_size_max + batch_size] = pos_energy
                    energy_neg_list[i * batch_size_max:i * batch_size_max + batch_size] = neg_energy
                    energy_kl_list[i * batch_size_max:i * batch_size_max + batch_size] = kl_energy

                pos_energy = pos_energy.mean()
                neg_energy = neg_energy.mean()
                kl_energy = kl_energy.mean()
                if i % 20 == 0:
                    tqdm.write("#: {} // L: {:.2e} // (+): {:.2e} // (-): {:.2e} // "
                               "(-)-(+): {:.2e} // |dU|: {:.2e} // |N|: {:.2e}".format(i, loss, pos_energy,
                                                        neg_energy,
                                                        neg_energy - pos_energy,
                                                        avg_grad_mag, avg_noise_mag))
                wandb.log({"loss/total": loss,
                           "energy/reg": energy_regularization,
                           "energy/positive": pos_energy,
                           "energy/negative": neg_energy,
                           "energy/negative_relative": neg_energy - pos_energy,
                           "energy/kl_energy": kl_energy,
                           "mcmc/avg_grad": avg_grad_mag,
                           "mcmc/avg_noise": avg_noise_mag,
                           "loss/kl_loss": kl_loss,
                           "loss/max_likelihood": loss - kl_loss,
                           "batch_num": batch_iteration,
                           "epoch": epoch})

            # Longer-term metrics
            if rank == 0:
                # End training after fixed amount of time
                if time.time() - t_start_autoclose > 3600 * hyperparams['time_limit']:
                    sh_mem.buf[0] = 1

                # scalars
                avg_energy_pos = energy_pos_list.mean()
                avg_energy_neg = energy_neg_list.mean()
                avg_energy_kl = energy_kl_list.mean()

                # histograms
                energy_pos_list -= avg_energy_pos
                energy_neg_list -= avg_energy_pos
                energy_kl_list -= avg_energy_kl

                # Log to tensorboard every 5 min
                if (epoch == 0 and i == 3) or time.time() - t_start0 > 300 or sh_mem.buf[0] == 1:
                    t_start0 = time.time()
                    # write scalars and histograms
                    writer.add_scalar("loss/total", loss_avg, batch_iteration)
                    writer.add_scalar('energy/reg', reg_avg, batch_iteration)
                    writer.add_scalar("energy/positive", avg_energy_pos, batch_iteration)
                    writer.add_scalar("energy/negative", avg_energy_neg, batch_iteration)
                    writer.add_scalar("energy/negative_relative", avg_energy_neg - avg_energy_pos, batch_iteration)
                    writer.add_scalar("energy/kl_energy", avg_energy_kl, batch_iteration)
                    writer.add_scalar("mcmc/avg_grad", avg_grad_mag)
                    writer.add_scalar("mcmc/avg_noise", avg_noise_mag)
                    writer.add_scalar("loss/kl_loss", kl_loss_avg, batch_iteration)
                    writer.add_scalar("loss/max_likelihood", loss_avg - kl_loss_avg, batch_iteration)
                # wandb.log({"loss/total": loss_avg,
                #            "energy/reg": reg_avg,
                #            "energy/positive": avg_energy_pos,
                #            "energy/negative": avg_energy_neg,
                #            "energy/negative_relative": avg_energy_neg - avg_energy_pos,
                #            "energy/kl_energy": avg_energy_kl,
                #            "loss/kl_loss": kl_loss_avg,
                #            "loss/max_likelihood": loss_avg - kl_loss_avg,
                #            "epoch": epoch})

                # Add histogram every 20 min
                if (epoch == 0 and i == 3) or time.time() - t_start1 > 1200 or sh_mem.buf[0] == 1:
                    t_start1 = time.time()
                    try:
                        writer.add_histogram("energy/pos_relative", energy_pos_list, batch_iteration)
                        writer.add_histogram("energy/neg_relative", energy_neg_list, batch_iteration)
                        writer.add_histogram("energy_kl_list", energy_kl_list, batch_iteration)
                    except Exception as e:
                        print(e)
                    try:
                        for name, weight in model.named_parameters():
                            writer.add_histogram("w/" + name, weight, batch_iteration)
                            writer.add_histogram(f'g/{name}.grad', weight.grad, batch_iteration)
                    except Exception as e:
                        print(e)
                    writer.flush()
                    tqdm.write("E: {} // L: {:.2e} // (+): {:.2e} // (-): {:.2e} // "
                               "(-)-(+): {:.2e}".format(epoch, loss_avg, avg_energy_pos,
                                                        avg_energy_neg,
                                                        avg_energy_neg - avg_energy_pos))

                # EVERY FIVE MIN
                # Save checkpoint every hour
                if ((epoch == 0 and i == 3) or (epoch == num_epochs - 1 and i == num_batches - 1)
                    or time.time() - t_start2 > 300 or sh_mem.buf[0] == 1):
                    t_start2 = time.time()
                    torch.save({'epoch': epoch,
                                'batch_iteration': batch_iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lrScheduler_state_dict': lrScheduler.state_dict(),
                                'replay_buffer_list': replay_buffer.sample(batch_size_max)
                                },
                               path + "/checkpoints/model-{}-{}.pt".format(epoch, i))

            batch_iteration += 1
            batch_pbar.update(1)
            
#             grad_mag_list.append(avg_grad_mag)
#             if batch_iteration > 40 and switch_opt_flag:
#                 if torch.tensor(grad_mag_list[-10:]).mean() > 0.9 * noise_scale:
#                     optimizer = optimizer_SGD
#                     switch_opt_flag = False

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
    parser = argparse.ArgumentParser(description="Train the modular EBM")
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--reg_amount', type=float)
    parser.add_argument('--replay_frac', type=float)
    parser.add_argument('--replay_size', type=int)
    parser.add_argument('--replay_cyclic', type=bool)

    parser.add_argument('--sample_steps', type=int)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--noise_scale', type=float)
    parser.add_argument('--augment_data', type=bool)

    parser.add_argument('--batch_size_max', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)

    parser.add_argument('--kl_weight_energy', type=float)
    parser.add_argument('--kl_weight_entropy', type=float)
    parser.add_argument('--kl_backprop_steps', type=int)

    parser.add_argument('--weight_decay', type=float)
    # parser.add_argument('--identifier')

    parser.add_argument('--resume', type=bool)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_version', type=str)

    parser.add_argument('--dataset', type=str)

    # parser.add_argument('--seq_length', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--dense_width', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--num_msi_attn', type=int)
    parser.add_argument('--energy_num_heads', type=int)
    parser.add_argument('--energy_num_hidden', type=int)
    parser.add_argument('--energy_num_attn', type=int)

    parser.add_argument('--time_limit', type=float, default=-1,
                        help='Time limit (in hours). -1 for unlimited')
    parser.add_argument('--port', type=int, default=26000)
    args = parser.parse_args()

    hyperparams = {
        "num_epochs": 10,
        "reg_amount": 1e0,
        "replay_frac": 0.99,
        "replay_size": 8192,
        "replay_cyclic": False,

        "sample_steps": 50,
        "step_size": 1e2,
        "noise_scale": 5e-3,
        "augment_data": True,

        "batch_size_max": 32,
        "lr": 1e-5,
        "momentum": 0,

        "kl_weight_energy": 1e0,
        "kl_weight_entropy": 1e0,
        "kl_backprop_steps": 1,

        "weight_decay": 1e-1,
        # "identifier": identifier,
        "resume": False,
        "resume_path": None,
        "resume_version": None,
        'time_limit': -1,

        'dataset': "data-MSI-mini_2022-9-28_sets-1-train.npz",

        # Model settings
        'seq_length': 32,
        'embed_dim': 256,
        'dense_width': 128,
        'num_heads': 8,
        'num_hidden': 128,
        'num_msi_attn': 2,
        'energy_num_heads': 8,
        'energy_num_hidden': 256,
        'energy_num_attn': 2,
    }

    for key in vars(args).keys():
        if vars(args)[key] is not None and key != 'port':
            hyperparams[key] = vars(args)[key]

    world_size = 1
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getpid()), create=True, size=1)
    sh_mem.buf[0] = 0
    try:
        proc_context = torch.multiprocessing.spawn(main, args=(world_size, hyperparams, args.port),
                                                   nprocs=world_size, join=False)
        proc_context.join()
    except KeyboardInterrupt:
        sh_mem.buf[0] = 1
        proc_context.join(timeout=30)
        sh_mem.unlink()
    else:
        sh_mem.unlink()
