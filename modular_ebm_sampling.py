import torch


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
    gradient_enabled_mask = torch.cat([torch.ones((1, x.shape[1] - 10), device=x.get_device()),
                                       torch.zeros((1, 10), device=x.get_device())], dim=1)
    avg_grad_mag = 0
    avg_noise_mag = 0
    for i in range(sample_steps):
        x.requires_grad_(True)  # So gradients are saved

        # Noise / implementation in the style of Du 2019
        # noise = torch.randn_like(x) * noise_scale
        # model_output = model(x + noise)
        # Noise / implementation in the style of Nijkamp 2020
        noise = torch.randn_like(x) * step_size
        model_output = model(x)
        # Only inputs so that only grad wrt x is calculated (and not all remaining vars)
        gradient = torch.autograd.grad(model_output.sum(), x, only_inputs=True)[0] * gradient_enabled_mask

        # For diagnostics purposes
        # Calculate average (over samples) gradient magnitude
        avg_grad_mag += torch.sqrt(torch.mean((gradient * (step_size ** 2) / 2).square(), dim=1)).mean().detach()
        # Similar calculation, should be equal to step_size on average
        avg_noise_mag += torch.sqrt(torch.mean(noise.square(), dim=1)).mean().detach()

        # Backpropping through 5 steps
        # kl_backprop_steps = 5
        if i == sample_steps - kl_backprop_steps:
            x_KL = x
        if i >= sample_steps - kl_backprop_steps:
            # We're not going to detach so we retain the gradients
            kl_model_output = model(x_KL + noise)
            kl_gradient = torch.autograd.grad(kl_model_output.sum(), x_KL,
                                              only_inputs=True, create_graph=True)[0]
            # Noise / implementation in the style of Du 2019
            # x_KL = x_KL - kl_gradient * step_size * gradient_enabled_mask
            # Noise / implementation in the style of Nijkamp 2020
            x_KL = x_KL - (kl_gradient * (step_size ** 2) / 2 - noise) * gradient_enabled_mask
        # Noise / implementation in the style of Du 2019
        # x = x - gradient * step_size * gradient_enabled_mask
        # Noise / implementation in the style of Nijkamp 2020
        x = x - (gradient * (step_size ** 2) / 2 - noise * gradient_enabled_mask)
        x = x.detach()  # Remove the gradients

    avg_grad_mag = avg_grad_mag / (sample_steps + 1)
    avg_noise_mag = avg_noise_mag / (sample_steps + 1)

    return x, x_KL, avg_grad_mag, avg_noise_mag


def perturb_samples(samples):
    # Constants chosen because they made sense on 11/1/2021
    rand_gaussian = torch.randn_like(samples) * 0.2
    rand_mulitplier = torch.rand_like(samples[:, 0])[:, None] * 1.0 + 0.5

    # Not sure what the order of operations should be here
    samples[:, 0:-10] = samples[:, 0:-10] * rand_mulitplier[:] + rand_gaussian[:, 0:-10]
    return samples
