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
        f_num = 4
        stride = 1
        k_len = 16
        pad = 16
        pad_mode = 'zeros'

        # self.ts_conv = ModuleList([
        #     torch.nn.LazyConv2d(f_num, (6, k_len), padding=(0, pad), padding_mode=pad_mode),
        #     torch.nn.LazyConv2d(f_num, (1, k_len), padding=(0, pad), padding_mode=pad_mode),
        #     torch.nn.LazyConv2d(f_num, (1, k_len), padding=(0, pad), padding_mode=pad_mode),
        #     torch.nn.LazyConv2d(f_num, (1, k_len), padding=(0, pad), padding_mode=pad_mode)])

        # self.ts_dense = ModuleList([
        #     torch.nn.LazyLinear(256),
        #     torch.nn.LazyLinear(128)])

        # Conv model
        self.I_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.I_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.I_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.V_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.V_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.V_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.n_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.n_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.n_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.d0_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d0_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.d0_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.d1_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d1_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.d1_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.d2_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.d2_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.d2_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.B_conv = ModuleList([
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            # (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode)),
            (torch.nn.LazyConv1d(f_num, k_len, stride=stride, padding=pad, padding_mode=pad_mode))])
        self.B_dense = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8))])

        self.B_downsampled = ModuleList([
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.p_layers = ModuleList([
            # (torch.nn.LazyLinear(32)),
            (torch.nn.LazyLinear(64)),
            (torch.nn.LazyLinear(8)),
            (torch.nn.LazyLinear(8))])

        self.energy_layers = ModuleList([
            # (torch.nn.LazyLinear(256)),
            # (torch.nn.LazyLinear(128)),
            # (torch.nn.LazyLinear(128)),
            # (torch.nn.LazyLinear(64)),
            # (torch.nn.LazyLinear(64)),
            (torch.nn.LazyLinear(16)),
            # (torch.nn.LazyLinear(32)),
            # (torch.nn.LazyLinear(16)),
            (torch.nn.LazyLinear(8))])
        self.energy_final = torch.nn.LazyLinear(1)

        # self.energy_ds_layers = ModuleList([
        #     (torch.nn.LazyLinear(256)),
        #     (torch.nn.LazyLinear(128)),
        #     (torch.nn.LazyLinear(64)),
        #     (torch.nn.LazyLinear(32)),
        #     (torch.nn.LazyLinear(16))])
        # self.energy_ds_final = torch.nn.LazyLinear(1)

    def forward(self, x):
        SiLU = torch.nn.functional.silu
        # ELU = torch.nn.functional.elu

        batch_size = x.shape[0]

        # ts_x = x[:, 0:256 * 6].reshape(batch_size, 1, 6, 256)

        I_x = x[:, 256 * 0:256 * 1].unsqueeze(dim=1)
        V_x = x[:, 256 * 1:256 * 2].unsqueeze(dim=1)
        n_x = x[:, 256 * 2:256 * 3].unsqueeze(dim=1)
        d0_x = x[:, 256 * 3:256 * 4].unsqueeze(dim=1)
        d1_x = x[:, 256 * 4:256 * 5].unsqueeze(dim=1)
        d2_x = x[:, 256 * 5:256 * 6].unsqueeze(dim=1)
        B_x = x[:, 1536:1536 + 128].unsqueeze(dim=1)
        p_x = x[:, 1664:1664 + 51]  # don't need unsqueeze because no conv ops

        I_ds_x = I_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.I_downsampled:
            I_ds_x = SiLU(layer(I_ds_x))
        for layer in self.I_conv:
            I_x = SiLU(layer(I_x))
        I_x = (I_x).reshape(batch_size, -1)
        I_x = torch.cat([I_x, I_ds_x], 1)
        for layer in self.I_dense:
            I_x = SiLU(layer(I_x))

        V_ds_x = V_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.V_downsampled:
            V_ds_x = SiLU(layer(V_ds_x))
        for layer in self.V_conv:
            V_x = SiLU(layer(V_x))
        V_x = (V_x).reshape(batch_size, -1)
        V_x = torch.cat([V_x, V_ds_x], 1)
        for layer in self.V_dense:
            V_x = SiLU(layer(V_x))

        n_ds_x = n_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.n_downsampled:
            n_ds_x = SiLU(layer(n_ds_x))
        for layer in self.n_conv:
            n_x = SiLU(layer(n_x))
        n_x = (n_x).reshape(batch_size, -1)
        n_x = torch.cat([n_x, n_ds_x], 1)
        for layer in self.n_dense:
            n_x = SiLU(layer(n_x))

        d0_ds_x = d0_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.d0_downsampled:
            d0_ds_x = SiLU(layer(d0_ds_x))
        for layer in self.d0_conv:
            d0_x = SiLU(layer(d0_x))
        d0_x = (d0_x).reshape(batch_size, -1)
        d0_x = torch.cat([d0_x, d0_ds_x], 1)
        for layer in self.d0_dense:
            d0_x = SiLU(layer(d0_x))

        d1_ds_x = d1_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.d1_downsampled:
            d1_ds_x = SiLU(layer(d1_ds_x))
        for layer in self.d1_conv:
            d1_x = SiLU(layer(d1_x))
        d1_x = (d1_x).reshape(batch_size, -1)
        d1_x = torch.cat([d1_x, d1_ds_x], 1)
        for layer in self.d1_dense:
            d1_x = SiLU(layer(d1_x))

        d2_ds_x = d2_x.reshape(-1, 16, 16).mean(dim=2)
        for layer in self.d2_downsampled:
            d2_ds_x = SiLU(layer(d2_ds_x))
        for layer in self.d2_conv:
            d2_x = SiLU(layer(d2_x))
        d2_x = (d2_x).reshape(batch_size, -1)
        d2_x = torch.cat([d2_x, d2_ds_x], 1)
        for layer in self.d2_dense:
            d2_x = SiLU(layer(d2_x))

        B_ds_x = B_x.reshape(-1, 16, 8).mean(dim=2)
        for layer in self.B_downsampled:
            B_ds_x = SiLU(layer(B_ds_x))
        for layer in self.B_conv:
            B_x = SiLU(layer(B_x))
        B_x = (B_x).reshape(batch_size, -1)
        B_x = torch.cat([B_x, B_ds_x], 1)
        for layer in self.B_dense:
            B_x = SiLU(layer(B_x))

        for layer in self.p_layers:
            p_x = SiLU(layer(p_x))

        x = torch.cat((I_x, V_x, n_x, d0_x, d1_x, d2_x, B_x, p_x), 1)

        # for layer in self.ts_conv:
        #     ts_x = SiLU(layer(ts_x))
        # ts_x = ts_x.reshape(batch_size, -1)
        # for layer in self.ts_dense:
        #     ts_x = SiLU(layer(ts_x))

        # x = torch.cat([ts_x, B_x, p_x], 1)

        for layer in self.energy_layers:
            x = SiLU(layer(x))
        x = self.energy_final(x)

        # ds_x = torch.cat((I_ds_x, V_ds_x, n_ds_x, d0_ds_x, d1_ds_x, d2_ds_x, B_ds_x, p_x), 1)
        # for layer in self.energy_ds_layers:
            # ds_x = SiLU(layer(ds_x))
        # ds_x = self.energy_ds_final(ds_x)

        # return x + ds_x
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
                        'replay_buffer_list': replay_buffer.sample_list},
                       path + "/checkpoints/model-{}.pt".format(epoch))

        pbar.update(1)
    pbar.close()
