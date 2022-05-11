import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary

import numpy as np
from tqdm import tqdm
import wandb
import os
import datetime
import shutil
import importlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani

from modular_ebm_diagnostics import *
from modular_ebm_sampling import *


def load_data(path):
    data_signals = np.load(path)['signals']
    # Encode positions later
    # data_positions = np.load(path)['positions']
    data_bad_table = np.load(path)['bad_table']

    complete_data = np.where(np.sum(data_bad_table, axis=1) == 0)[0]
    data_signals = data_signals[complete_data]
    data = torch.tensor(data_signals, dtype=torch.float)

    return data


class ModularWithRNNBackbone(torch.nn.Module):
    def __init__(self):
        super(ModularWithRNNBackbone, self).__init__()

        seq_length = 256

        # seq_length, num_msi_attn, num_mem_attn, num_sum_attn
        self.dishargeI = MSITimeSeriesModule(seq_length, 1, 1, 1).cuda()
        self.dishargeV = MSITimeSeriesModule(seq_length, 1, 1, 1).cuda()
        self.interferometer = MSITimeSeriesModule(seq_length, 1, 1, 1).cuda()
        self.diodes = MSITimeSeriesModule(seq_length, 1, 1, 1).cuda()
        self.magnets = MagneticFieldModule(seq_length, 1, 1, 1).cuda()
        self.RGA = RGAPressureModule(seq_length, 1, 1, 1).cuda()

        embed_dim = 16
        num_heads = 4
        num_hidden = 128
        num_attn = 2

        self.seqPosEnc = SequencePositionalEncoding(embed_dim, seq_length).cuda()
        self.attnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(embed_dim, num_heads, num_hidden).cuda()
            for i in range(num_attn)])

        self.softmax = torch.nn.Softmax(dim=1)  # softmax over the seq_length
        self.linear = torch.nn.LazyLinear(1)

    def forward(self, x):
        device = getDeviceString(x.get_device())

        iterations = 1

        I_x = x[:, 256 * 0:256 * 1]
        V_x = x[:, 256 * 1:256 * 2]
        n_x = x[:, 256 * 2:256 * 3]
        d0_x = x[:, 256 * 3:256 * 4]
        d1_x = x[:, 256 * 4:256 * 5]
        d2_x = x[:, 256 * 5:256 * 6]
        B_x = x[:, 1536:1536 + 128]
        p_x = x[:, 1664:1664 + 51]

        batch_size = I_x.shape[0]

        shared_memory_temp = torch.zeros((batch_size, 256, 16)).to(device)
        shared_memory = torch.zeros((batch_size, 256, 16)).to(device)
        for i in range(iterations):
            shared_memory += self.dishargeI(I_x, shared_memory_temp)
            shared_memory += self.dishargeV(V_x, shared_memory_temp)
            shared_memory += self.interferometer(n_x, shared_memory_temp)
            shared_memory += self.diodes(d0_x, shared_memory_temp)
            shared_memory += self.diodes(d1_x, shared_memory_temp)
            shared_memory += self.diodes(d2_x, shared_memory_temp)
            shared_memory += self.magnets(B_x, shared_memory_temp)
            shared_memory += self.RGA(p_x, shared_memory_temp)
            shared_memory_temp = shared_memory

        shared_memory = self.seqPosEnc(shared_memory)

        for i, block in enumerate(self.attnBlocks):
            shared_memory = block(shared_memory)

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

    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    project_name = "modular_ebm"
    exp_path = "experiments_modular/"
    path = exp_path + identifier
    if rank == 0:
        os.mkdir(path)
        os.mkdir(path + "/checkpoints")
        shutil.copy(project_name + ".py", path + "/" + project_name + "_copy.py")
        shutil.copy(project_name + "_sampling.py", path + "/" + project_name + "sampling_copy.py")
        shutil.copy(project_name + "_diagnostics.py", path + "/" + project_name + "diagnostics_copy.py")

    hyperparams = {
        "num_epochs": 51,
        "reg_amount": 1e0,
        "replay_frac": 0.99,
        "replay_size": 8192,

        "sample_steps": 50,
        "step_size": 1e2,
        "noise_scale": 5e-3,
        "augment_data": True,

        "batch_size_max": 20,
        "lr": 1e-4,

        "kl_weight_energy": 1e-1,
        "kl_weight_entropy": 1e-1,
        "kl_backprop_steps": 1,

        "weight_decay": 1e-1,
        "identifier": identifier,
        "resume": False,
        # "resume_path": "2022-03-16_22h-00m-14s",
        # "resume_version": "checkpoints/model-50"
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

    data_path = "data/data-MSI-hairpin_002-train.npz"
    data = load_data(data_path)

    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size,
                                                              rank=rank, shuffle=True,
                                                              drop_last=False)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                             batch_size=batch_size_max, shuffle=False,
                                             num_workers=0, pin_memory=False, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.0, 0.999))
    replay_buffer = ReplayBuffer(replay_size, torch.rand((replay_size, data.shape[1])).to(rank) * 2 - 1)

    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        replay_buffer.sample_list = ckpt['replay_buffer_list']

    num_data = data.shape[0]
    num_batches = len(dataloader)

    # initialze for lazy layers so that the num_parameters works properly
    model = model.to(rank)
    model(torch.zeros((2, 1715)).to(rank))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if rank == 0:
        summary(model, (1715,))
        num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("Parameters: {}".format(num_parameters))
        hyperparams['num_parameters'] = num_parameters
        wandb.init(project="modular-ebm", entity='phil',
                   group="", job_type="",
                   config=hyperparams)

    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(init_scale=1e6)
    for epoch in range(num_epochs):
        # with torch.cuda.amp.autocast():s
        sampler.set_epoch(epoch)
        loss_avg = 0
        reg_avg = 0
        kl_loss_avg = 0

        energy_list_size = batch_size_max * 4 if data.shape[0] > batch_size_max * 4 else data.shape[0]
        energy_pos_list = torch.zeros((energy_list_size, 1)).to(rank)
        energy_neg_list = torch.zeros((energy_list_size, 1)).to(rank)
        energy_kl_list = torch.zeros((energy_list_size, 1)).to(rank)

        batch_pbar = tqdm(total=num_batches)
        for pos_x, i in zip(dataloader, range(num_batches)):
            # print("Rank: {}\tBatchnum: {}".format(rank, i))
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                pos_x = pos_x[0].to(rank)
                pos_x.requires_grad = True
                batch_size = pos_x.shape[0]

                neg_x = replay_buffer.sample(int(batch_size * replay_frac))
                if augment_data:
                    neg_x = perturb_samples(neg_x)
                neg_x_rand = (torch.rand(batch_size - neg_x.shape[0], *list(pos_x.shape[1:])) *
                              2 - 1).to(rank)
                neg_x = torch.cat([neg_x, neg_x_rand], 0)
                # neg_x = torch.Tensor(neg_x).to(rank)
                # neg_x = torch.Tensor(neg_x)
                # neg_x.requires_grad = True  # Needed if not using Langevin_KL sampling

                # For calculating the KL loss later
                num_kl_samples = 100
                kl_samp = replay_buffer.sample(num_kl_samples)

                # Run Langevin dynamics on sample
                neg_x, kl_x = sample_langevin_KL_cuda(neg_x, model, sample_steps=sample_steps,
                                                      kl_backprop_steps=hyperparams["kl_backprop_steps"],
                                                      step_size=step_size, noise_scale=noise_scale)
                replay_buffer.add(neg_x)  # neg_x already detached in Langevin call above

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
            pos_energy = model(pos_x)
            neg_energy = model(neg_x.to(rank))
            # neg_energy = model(neg_x)
            energy_regularization = reg_amount * (pos_energy.square() + neg_energy.square()).mean()

            loss = ((pos_energy - neg_energy).mean() + energy_regularization + kl_loss)
            # loss.backward()
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                loss_avg += loss.detach() * batch_size / num_data
                reg_avg += energy_regularization.detach() * batch_size / num_data
                # print("\n")
                # print(reg_avg)
                kl_loss_avg += kl_loss.detach() * batch_size / num_data

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
                               "(-)-(+): {:.2e}".format(i, loss, pos_energy,
                                                        neg_energy,
                                                        neg_energy - pos_energy))
                wandb.log({"loss/total": loss,
                           "energy/reg": energy_regularization,
                           "energy/positive": pos_energy,
                           "energy/negative": neg_energy,
                           "energy/negative_relative": neg_energy - pos_energy,
                           "energy/kl_energy": kl_energy,
                           "loss/kl_loss": kl_loss,
                           "batch_num": batch_iteration,
                           "epoch": epoch})
                batch_iteration += 1
            batch_pbar.update(1)
        batch_pbar.close()

        if rank == 0:
            # scalars
            avg_energy_pos = energy_pos_list.mean()
            avg_energy_neg = energy_neg_list.mean()
            avg_energy_kl = energy_kl_list.mean()

            # histograms
            energy_pos_list -= avg_energy_pos
            energy_neg_list -= avg_energy_pos
            energy_kl_list -= avg_energy_kl

            # write scalars and histograms
            writer.add_scalar("loss/total", loss_avg, epoch)
            writer.add_scalar('energy/reg', reg_avg, epoch)
            writer.add_scalar("energy/positive", avg_energy_pos, epoch)
            writer.add_scalar("energy/negative", avg_energy_neg, epoch)
            writer.add_scalar("energy/negative_relative", avg_energy_neg - avg_energy_pos, epoch)
            writer.add_scalar("energy/kl_energy", avg_energy_kl, epoch)
            writer.add_scalar("loss/kl_loss", kl_loss_avg, epoch)
            wandb.log({"loss/total": loss_avg,
                       "energy/reg": reg_avg,
                       "energy/positive": avg_energy_pos,
                       "energy/negative": avg_energy_neg,
                       "energy/negative_relative": avg_energy_neg - avg_energy_pos,
                       "energy/kl_energy": avg_energy_kl,
                       "loss/kl_loss": kl_loss_avg,
                       "epoch": epoch})

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                writer.add_histogram("energy/pos_relative", energy_pos_list, epoch)
                writer.add_histogram("energy/neg_relative", energy_neg_list, epoch)
                writer.add_histogram("energy_kl_list", energy_kl_list, epoch)
                for name, weight in model.named_parameters():
                    writer.add_histogram("w/" + name, weight, epoch)
                    writer.add_histogram(f'g/{name}.grad', weight.grad, epoch)
                writer.flush()
                tqdm.write("E: {} // L: {:.2e} // (+): {:.2e} // (-): {:.2e} // "
                           "(-)-(+): {:.2e}".format(epoch, loss_avg, avg_energy_pos,
                                                    avg_energy_neg,
                                                    avg_energy_neg - avg_energy_pos))

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'replay_buffer_list': replay_buffer.sample_list
                            },
                           path + "/checkpoints/model-{}.pt".format(epoch))

        pbar.update(1)
    pbar.close()

    cleanup()


if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
