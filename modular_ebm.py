import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani
from tqdm import tqdm
import wandb
import os
import datetime
import shutil
import importlib


def load_data(path):
    data_signals = np.load(path)['signals']
    # Encode positions later
    # data_positions = np.load(path)['positions']
    data_bad_table = np.load(path)['bad_table']

    complete_data = np.where(np.sum(data_bad_table, axis=1) == 0)[0]
    data_signals = data_signals[complete_data]
    data = torch.tensor(data_signals, dtype=torch.float)

    return data


class ReplayBuffer():
    def __init__(self, max_size, init_data):
        self.sample_list = init_data
        self.max_size = max_size

    # This starts populated with noise and then is eventually replaced with generated samples
    def add(self, samples):
        self.sample_list = torch.cat([self.sample_list, samples], 0)
        buffer_len = self.sample_list.shape[0]
        if buffer_len > self.max_size:
            self.sample_list = self.sample_list[buffer_len - self.max_size:]

    def sample(self, num_samples):
        buffer_len = self.sample_list.shape[0]
        indicies = torch.randint(0, buffer_len,
                                 (num_samples if buffer_len > num_samples else buffer_len,))
        return self.sample_list[indicies]


def sample_langevin(x, model, sample_steps=10, step_size=10, noise_scale=0.005, return_list=False):
    sample_list = []
    sample_list.append(x.detach())
    for _ in range(sample_steps):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x + noise)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size
        sample_list.append(x.detach().cpu())
    if return_list:
        return sample_list
    else:
        return sample_list[-1]


def sample_langevin_cuda(x, model, sample_steps=10, step_size=10, noise_scale=0.005):
    for _ in range(sample_steps):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x + noise)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size
    return x


def sample_langevin_KL_cuda(x, model, sample_steps=10, kl_backprop_steps=5, step_size=10, noise_scale=0.005):
    for i in range(sample_steps):
        x.requires_grad_(True)  # So gradients are saved
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x + noise)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]

        # Backpropping through 5 steps
        # kl_backprop_steps = 5
        if i == sample_steps - kl_backprop_steps:
            x_KL = x
        if i >= sample_steps - kl_backprop_steps:
            # We're not going to detach so we retain the gradients
            kl_model_output = model(x_KL + noise)
            kl_gradient = torch.autograd.grad(kl_model_output.sum(), x_KL,
                                              only_inputs=True, create_graph=True)[0]
            x_KL = x_KL - kl_gradient * step_size
        x = (x - gradient * step_size).detach()  # Remove the gradients

    return x, x_KL


def perturb_samples(samples):
    # Constants chosen because they made sense on 11/1/2021
    rand_gaussian = torch.randn_like(samples) * 0.2
    rand_mulitplier = torch.rand_like(samples[:, 0])[:, None] * 1.0 + 0.5

    # Not sure what the order of operations should be here
    return samples * rand_mulitplier + rand_gaussian


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # spec_norm = torch.nn.utils.spectral_norm
        ModuleList = torch.nn.ModuleList

        # Conv model
        self.I_conv = ModuleList([
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        

    def forward(self, x):
        
        return x


if __name__ == "__main__":
    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    project_name = "msi_ebm"
    exp_path = "experiments_msi/"
    path = exp_path + identifier
    os.mkdir(path)
    os.mkdir(path + "/checkpoints")
    shutil.copy(project_name + ".py", path + "/" + project_name + "_copy.py")

    hyperparams = {
        "num_epochs": 301,
        "reg_amount": 1e0,
        "replay_frac": 0.99,
        "replay_size": 8192,

        "sample_steps": 50,
        "step_size": 1e2,
        "noise_scale": 5e-3,
        "augment_data": True,

        "batch_size_max": 1024,
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
    model = NeuralNet().cuda()
    if resume:
        spec = importlib.util.spec_from_file_location(project_name + "_copy", exp_path +
                                                      resume_path + "/" + project_name + "_copy.py")
        ebm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ebm)
        model = ebm.NeuralNet().cuda()

    data_path = "data/data-MSI-hairpin_002-train.npz"
    data = load_data(data_path)

    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                             batch_size=batch_size_max, shuffle=True,
                                             num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.0, 0.999))
    replay_buffer = ReplayBuffer(replay_size, torch.rand((replay_size, data.shape[1])).cuda() * 2 - 1)

    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        replay_buffer.sample_list = ckpt['replay_buffer_list']

    num_data = data.shape[0]
    num_batches = int(np.ceil(num_data / batch_size_max))

    # initialze for lazy layers so that the num_parameters works properly
    model(torch.zeros((2, 1715)).cuda())
    num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Parameters: {}".format(num_parameters))
    hyperparams['num_parameters'] = num_parameters
    wandb.init(project="msi-ebm", entity='phil',
               group="Copy of green dot", job_type="downsampled branch, data aug",
               config=hyperparams)

    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(init_scale=1e6)
    for epoch in range(num_epochs):
        # with torch.cuda.amp.autocast():
        loss_avg = 0
        reg_avg = 0
        kl_loss_avg = 0

        energy_list_size = batch_size_max * 4 if data.shape[0] > batch_size_max * 4 else data.shape[0]
        energy_pos_list = torch.zeros((energy_list_size, 1)).cuda()
        energy_neg_list = torch.zeros((energy_list_size, 1)).cuda()
        energy_kl_list = torch.zeros((energy_list_size, 1)).cuda()

        batch_pbar = tqdm(total=num_batches)
        for pos_x, i in zip(dataloader, range(num_batches)):
            # with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            pos_x = pos_x[0].cuda()
            pos_x.requires_grad = True
            batch_size = pos_x.shape[0]

            neg_x = replay_buffer.sample(int(batch_size * replay_frac))
            if augment_data:
                neg_x = perturb_samples(neg_x)
            neg_x_rand = (torch.rand(batch_size - neg_x.shape[0], *list(pos_x.shape[1:])) *
                          2 - 1).cuda()
            neg_x = torch.cat([neg_x, neg_x_rand], 0)
            # neg_x = torch.Tensor(neg_x).cuda()
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
            neg_energy = model(neg_x.cuda())
            # neg_energy = model(neg_x)
            energy_regularization = reg_amount * (pos_energy.square() + neg_energy.square()).mean()

            loss = ((pos_energy - neg_energy).mean() + energy_regularization + kl_loss)
            loss.backward()
            # scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

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
