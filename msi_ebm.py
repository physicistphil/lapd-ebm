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
    data = torch.tensor(data_signals).float()

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


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        spec_norm = torch.nn.utils.spectral_norm
        ModuleList = torch.nn.ModuleList
        f_num = 4
        stride = 1
        k_len = 16
        pad = 8
        pad_mode = 'replicate'

        # Conv model
        self.I_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.I_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.V_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.V_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.n_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.n_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.d0_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d0_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.d1_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d1_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.d2_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d2_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (256), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.B_conv = ModuleList([
            spec_norm(torch.nn.Conv1d(1, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            spec_norm(torch.nn.Conv1d(f_num, f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.B_dense = ModuleList([
            spec_norm(torch.nn.Linear(f_num * (128), 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.p_layers = ModuleList([
            spec_norm(torch.nn.Linear(51, 32)),
            spec_norm(torch.nn.Linear(32, 32)),
            spec_norm(torch.nn.Linear(32, 32)),
            spec_norm(torch.nn.Linear(32, 32))])

        self.energy_layers = ModuleList([
            spec_norm(torch.nn.Linear(32 * 8, 32)),
            spec_norm(torch.nn.Linear(32, 32)),
            spec_norm(torch.nn.Linear(32, 32))])
        self.energy_final = torch.nn.Linear(32, 1)

    def forward(self, x):
        SiLU = torch.nn.functional.silu
        # ELU = torch.nn.functional.elu

        batch_size = x.shape[0]

        I_x = x[:, 256 * 0:256 * 1].unsqueeze(dim=1)
        V_x = x[:, 256 * 1:256 * 2].unsqueeze(dim=1)
        n_x = x[:, 256 * 2:256 * 3].unsqueeze(dim=1)
        d0_x = x[:, 256 * 3:256 * 4].unsqueeze(dim=1)
        d1_x = x[:, 256 * 4:256 * 5].unsqueeze(dim=1)
        d2_x = x[:, 256 * 5:256 * 6].unsqueeze(dim=1)
        B_x = x[:, 1536:1536 + 128].unsqueeze(dim=1)
        p_x = x[:, 1664:1664 + 51]  # don't need unsqueeze because no conv ops

        for layer in self.I_conv:
            # print(I_x.shape)
            I_x = SiLU(layer(I_x))[:, :, 0:-1]
        I_x = (I_x).reshape(batch_size, -1)
        # print('\n')
        for layer in self.I_dense:
            # print(I_x.shape)
            I_x = SiLU(layer(I_x))

        for layer in self.V_conv:
            V_x = SiLU(layer(V_x))[:, :, 0:-1]
        V_x = (V_x).reshape(batch_size, -1)
        for layer in self.V_dense:
            V_x = SiLU(layer(V_x))

        for layer in self.n_conv:
            n_x = SiLU(layer(n_x))[:, :, 0:-1]
        n_x = (n_x).reshape(batch_size, -1)
        for layer in self.n_dense:
            n_x = SiLU(layer(n_x))

        for layer in self.d0_conv:
            d0_x = SiLU(layer(d0_x))[:, :, 0:-1]
        d0_x = (d0_x).reshape(batch_size, -1)
        for layer in self.d0_dense:
            d0_x = SiLU(layer(d0_x))

        for layer in self.d1_conv:
            d1_x = SiLU(layer(d1_x))[:, :, 0:-1]
        d1_x = (d1_x).reshape(batch_size, -1)
        for layer in self.d1_dense:
            d1_x = SiLU(layer(d1_x))

        for layer in self.d2_conv:
            d2_x = SiLU(layer(d2_x))[:, :, 0:-1]
        d2_x = (d2_x).reshape(batch_size, -1)
        for layer in self.d2_dense:
            d2_x = SiLU(layer(d2_x))

        for layer in self.B_conv:
            B_x = SiLU(layer(B_x))[:, :, 0:-1]
        B_x = (B_x).reshape(batch_size, -1)
        for layer in self.B_dense:
            B_x = SiLU(layer(B_x))

        for layer in self.p_layers:
            p_x = SiLU(layer(p_x))
        # p_x = torch.squeeze(p_x)
        # for layer in self.p_dense:
            # p_x = SiLU(layer(p_x))
        x = torch.cat((I_x, V_x, n_x, d0_x, d1_x, d2_x, B_x, p_x), 1)

        for layer in self.energy_layers:
            x = SiLU(layer(x))
        x = self.energy_final(x)

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
        "num_epochs": 51,
        "reg_amount": 1e0,
        "replay_frac": 0.98,
        "replay_size": 8192,
        "sample_steps": 10,
        "step_size": 1e2,
        "noise_scale": 5e-3,
        "batch_size_max": 1024,
        "lr": 1e-4,
        "weight_decay": 1e-1,
        "identifier": identifier,
        "resume": False,
        # "resume_path": "2021-04-30_18h-39m-50s",
        # "resume_version": "checkpoints/model-75000"
    }

    num_epochs = hyperparams["num_epochs"]
    reg_amount = hyperparams["reg_amount"]
    replay_frac = hyperparams["replay_frac"]
    replay_size = hyperparams["replay_size"]
    sample_steps = hyperparams["sample_steps"]
    step_size = hyperparams["step_size"]
    noise_scale = hyperparams["noise_scale"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]
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
        # sample_langevin = ebm.sample_langevin
        # sample_langevin_cuda = ebm.sample_langevin_cuda
        # ReplayBuffer = ebm.ReplayBuffer
        model = ebm.NeuralNet().cuda()

    data_path = "data/data-MSI-hairpin_001.npz"
    data = load_data(data_path)

    # writer.add_graph(model, data[0:1].cuda())

    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                                             batch_size=batch_size_max, shuffle=True,
                                             num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.0, 0.999))
    replay_buffer = ReplayBuffer(replay_size, torch.randn(*data.shape).cuda())

    if resume:
        # sample_langevin = lapd_ebm.sample_langevin
        # sample_langevin_cuda = lapd_ebm.sample_langevin_cuda
        # ReplayBuffer = lapd_ebm.ReplayBuffer
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # data = torch.tensor(np.load("data/isat_downsampled_8_div3.npz")['arr_0'].reshape(-1, 10)).float()
        # data = torch.tensor(np.load("data/isat_downsampled_8.npz")['arr_0'].reshape(-1, 10)).float()
        # replay_buffer = ReplayBuffer(ckpt['replay_buffer_list'].shape[0],
                                     # np.random.randn(*data.shape))
        replay_buffer.sample_list = ckpt['replay_buffer_list']

    num_data = data.shape[0]
    num_batches = int(np.ceil(num_data / batch_size_max))

    num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Parameters: {}".format(num_parameters))
    hyperparams['num_parameters'] = num_parameters
    wandb.init(project="msi-ebm", entity='phil',
               group="Smaller Conv", job_type="mixed precision testing",
               config=hyperparams)

    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        with torch.cuda.amp.autocast():
            loss_avg = 0
            reg_avg = 0
            energy_pos_list = torch.zeros((num_data, 1)).cuda()
            energy_neg_list = torch.zeros((num_data, 1)).cuda()

            for pos_x, i in zip(dataloader, range(num_batches)):
                optimizer.zero_grad()
                pos_x = torch.Tensor(pos_x[0]).cuda()
        #         pos_x = torch.Tensor(pos_x)
                pos_x.requires_grad = True
                batch_size = pos_x.shape[0]

                neg_x = replay_buffer.sample(int(batch_size * replay_frac))
                neg_x_rand = torch.randn(batch_size - neg_x.shape[0], *list(pos_x.shape[1:])).cuda()
                neg_x = torch.cat([neg_x, neg_x_rand], 0)
                # neg_x = torch.Tensor(neg_x).cuda()
        #         neg_x = torch.Tensor(neg_x)
                neg_x.requires_grad = True

                neg_x = sample_langevin_cuda(neg_x, model, sample_steps=sample_steps,
                                             step_size=step_size, noise_scale=noise_scale)
                replay_buffer.add(neg_x.detach())
                neg_x = neg_x.detach()

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

                if i % 20 == 0:
                    tqdm.write("Batch: {} // Loss: {:.3e} // Pos: {:.3e} // Neg: {:.3e} // "
                               "Neg_relative: {:.3e}".format(i, loss.mean(), pos_energy.mean(),
                                                             neg_energy.mean(),
                                                             neg_energy.mean() - pos_energy.mean()))


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

        if epoch % 5 == 0 or epoch == num_epochs - 1:
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

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'replay_buffer_list': replay_buffer.sample_list},
                       path + "/checkpoints/model-{}.pt".format(epoch))

        pbar.update(1)
    pbar.close()
