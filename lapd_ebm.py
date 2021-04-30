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


class ReplayBuffer():
    def __init__(self, max_size, init_data):
        self.sample_list = init_data
        self.max_size = max_size

    # This starts populated with noise and then is eventually replaced with generated samples
    def add(self, samples):
        self.sample_list = np.concatenate([self.sample_list, samples.numpy()], axis=0)
        buffer_len = self.sample_list.shape[0]
        if buffer_len > self.max_size:
            self.sample_list = np.delete(self.sample_list, np.s_[0:buffer_len - self.max_size], 0)

    def sample(self, num_samples):
        buffer_len = self.sample_list.shape[0]
        indicies = np.random.randint(0, buffer_len,
                                     num_samples if buffer_len > num_samples else buffer_len)
        return self.sample_list[indicies]


def sample_langevin(x, model, sample_steps=10, step_size=10, noise_scale=0.005, return_list=False):
    sample_list = []
    sample_list.append(x.detach())
    for _ in range(sample_steps):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size + noise
        sample_list.append(x.detach().cpu())
    if return_list:
        return sample_list
    else:
        return sample_list[-1]


def sample_langevin_cuda(x, model, sample_steps=10, step_size=10, noise_scale=0.005):
    for _ in range(sample_steps):
        noise = torch.randn_like(x) * noise_scale
        model_output = model(x)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0]
        x = x - gradient * step_size + noise
    return x


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.dense1 = torch.nn.Linear(10, 128)
        self.dense2 = torch.nn.Linear(128, 128)
        self.dense3 = torch.nn.Linear(128, 128)
        self.denseEnd = torch.nn.Linear(128, 1)

        self.dense1 = torch.nn.utils.spectral_norm(self.dense1)
        self.dense2 = torch.nn.utils.spectral_norm(self.dense2)
        self.dense3 = torch.nn.utils.spectral_norm(self.dense3)
        self.denseEnd = torch.nn.utils.spectral_norm(self.denseEnd)

    def forward(self, x):
        SiLU = torch.nn.functional.silu

        x = SiLU(self.dense1(x))
        x = SiLU(self.dense2(x))
        x = SiLU(self.dense3(x))
        x = self.denseEnd(x)

        return x


if __name__ == "__main__":
    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    path = "experiments/" + identifier
    os.mkdir(path)
    shutil.copy("lapd_ebm.py", path + "/lapd_ebm_copy.py")

    hyperparams = {
        "num_epochs": 10001,
        "reg_amount": 1e0,
        "replay_frac": 0.95,
        "replay_size": 8192,
        "sample_steps": 10,
        "step_size": 1e1,
        "noise_scale": 5e-3,
        "batch_size_max": 128,
        "lr": 1e-4,
        "identifier": identifier
    }

    wandb.init(project='lapd-ebm', entity='phil', config=hyperparams)

    num_epochs = hyperparams["num_epochs"]
    reg_amount = hyperparams["reg_amount"]
    replay_frac = hyperparams["replay_frac"]
    replay_size = hyperparams["replay_size"]
    sample_steps = hyperparams["sample_steps"]
    step_size = hyperparams["step_size"]
    noise_scale = hyperparams["noise_scale"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]

    writer = SummaryWriter(log_dir=path)
    model = NeuralNet().cuda()
    # model = NeuralNet()
    data = torch.tensor(np.load("data/isat_downsampled_8.npz")['arr_0'].reshape(-1, 10)).float()
    writer.add_graph(model, data[0:10].cuda())

    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                             batch_size=batch_size_max, shuffle=True,
                                             num_workers=24, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.0, 0.999))
    replay_buffer = ReplayBuffer(replay_size, np.random.randn(*data.shape))

    num_data = data.shape[0]
    num_batches = int(np.ceil(num_data / 128))

    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        loss_avg = 0
        reg_avg = 0
        energy_pos_list = torch.zeros((num_data, 1)).cuda()
        energy_neg_list = torch.zeros((num_data, 1)).cuda()

        for pos_x, i in zip(dataloader, range(num_batches)):
            pos_x = torch.Tensor(pos_x[0]).cuda()
    #         pos_x = torch.Tensor(pos_x)
            pos_x.requires_grad = True
            batch_size = pos_x.shape[0]

            neg_x = replay_buffer.sample(int(batch_size * replay_frac))
            neg_x_rand = np.random.randn(batch_size - neg_x.shape[0], *list(pos_x.shape[1:]))
            neg_x = np.concatenate([neg_x, neg_x_rand], axis=0)
            neg_x = torch.Tensor(neg_x).cuda()
    #         neg_x = torch.Tensor(neg_x)
            neg_x.requires_grad = True

            neg_x = sample_langevin_cuda(neg_x, model, sample_steps=sample_steps,
                                         step_size=step_size, noise_scale=noise_scale)
            replay_buffer.add(neg_x.detach().cpu())

            optimizer.zero_grad()

            pos_energy = model(pos_x)
            neg_energy = model(neg_x.cuda())
    #         neg_energy = model(neg_x)
            energy_regularization = reg_amount * (pos_energy.square() + neg_energy.square()).mean()
            loss = (pos_energy - neg_energy).mean() + energy_regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            loss_avg += loss * batch_size / num_data
            reg_avg += energy_regularization * batch_size / num_data
            energy_pos_list[i * batch_size_max:i * batch_size_max + batch_size] = pos_energy
            energy_neg_list[i * batch_size_max:i * batch_size_max + batch_size] = neg_energy

        # scalars
        avg_energy_pos = energy_pos_list.mean()
        avg_energy_neg = energy_neg_list.mean()

        # histograms
        energy_pos_list -= avg_energy_pos
        energy_neg_list -= avg_energy_pos

        # write scalars and histograms
        writer.add_scalar("energy/loss", loss_avg, epoch)
        writer.add_scalar('energy/reg', reg_avg, epoch)
        writer.add_scalar("energy/positive", avg_energy_pos, epoch)
        writer.add_scalar("energy/negative", avg_energy_neg, epoch)
        writer.add_scalar("energy/negative_relative", avg_energy_neg - avg_energy_pos, epoch)
        wandb.log({"energy/loss": loss_avg,
                   "energy/reg": reg_avg,
                   "energy/positive": avg_energy_pos,
                   "energy/negative": avg_energy_neg,
                   "energy/negative_relative": avg_energy_neg - avg_energy_pos})

        if epoch % 10 == 0:
            writer.add_histogram("energy/pos_relative", energy_pos_list, epoch)
            writer.add_histogram("energy/neg_relative", energy_neg_list, epoch)
            for name, weight in model.named_parameters():
                writer.add_histogram("w/" + name, weight, epoch)
                writer.add_histogram(f'g/{name}.grad', weight.grad, epoch)
            writer.flush()
            tqdm.write("Epoch: {} // Loss: {:.3e} // Pos: {:.3e} // Neg: {:.3e} // "
                       "Neg_relative: {:.3e}".format(epoch, loss_avg, avg_energy_pos,
                                                     avg_energy_neg,
                                                     avg_energy_neg - avg_energy_pos))

        if epoch % 100 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'replay_buffer_list': replay_buffer.sample_list},
                       path + "/model-{}.pt".format(epoch))

        pbar.update(1)
    pbar.close()
