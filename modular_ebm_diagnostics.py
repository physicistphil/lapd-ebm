import torch


class DiagnosticPositionModule(torch.nn.Module):
    def __init__(self):
        super(DiagnosticPositionModule, self).__init__()

    def forward(self, x):
        pass


class DiagnosticOrientationModule(torch.nn.Module):
    def __init__(self):
        super(DiagnosticOrientationModule, self).__init__()

    def forward(self, x):
        pass


class ProbePositionModule(torch.nn.Module):
    def __init__(self):
        super(ProbePositionModule, self).__init__()

    def forward(self, x):
        pass


class AddAndNorm(torch.nn.Module)
    def __init__(self):
        super(AddAndNorm, self).__init__()
        self.layerNorm = torch.nn.LayerNorm()

    def forward(self, x1, x2):
        x = self.layerNorm(x1 + x2)
        return x


# Input shape is (batch size, num_channels/features, seq_length)
# Implemented based on "Attention is all you need"
class PositionWiseFFNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(PositionWiseFFNN, self).__init__()
        
        self.dense1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        torch.transpose(x, 1, 2)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        torch.transpose(x, 1, 2)
        return x


class ResidualAttnBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_hidden):
        super(ResidualAttnBlock, self).__init__()

        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads,
                                                batch_first=True, need_weight=False)
        self.addNorm1 = AddAndNorm()
        self.posWiseNN = PositionWiseFFNN(embed_dim, num_hidden, embed_dim)
        self.addNorm2 = AddAndNorm()

    def forward(self, x_resid):
        x = self.attn(x_resid)
        x_resid = self.addNorm1(x, x_resid)
        x = self.posWiseNN(x_resid)
        x = self.addNorm2(x, x_resid)
        return x


class MSITimeSeriesModule(torch.nn.Module):
    def __init__(self, num_msi_attn_blocks, num_mem_attn_blocks, ):
        super(MSITimeSeriesModule, self).__init__()

        # CNN setup
        out_channels = 8
        kernel_size = 8
        self.zeroPad = torch.nn.constantPad1d(0, (4, 3))
        # Attention setup
        num_heads = 4
        num_hidden = 256

        # Signal processing branch
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv2 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.attnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden)
            for i in range(num_attn_blocks)])

    def forward(self, x, shared_memory, pos_module, orient_module):
        

        # x is shape (batch_size, seq_length)
        x = torch.unsqueeze(x, 1)  # Make into shape (batch_size, channels, seq_length)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv1(x)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv2(x)

        
# class DischargeCurrentModule(torch.nn.Module):
#     def __init__(self):
#         super(DischargeCurrentModule, self).__init__()

#     def forward(self, x):
#         pass


# class DischargeVoltageModule(torch.nn.Module):
#     def __init__(self):
#         super(DischargeVoltageModule, self).__init__()

#     def forward(self, x):
#         pass


# class DiodeModule(torch.nn.Module):
#     def __init__(self):
#         super(DiodeModule, self).__init__()

#     def forward(self, x):
#         pass


# class DiodeModuleHeII(torch.nn.Module):
#     def __init__(self):
#         super(DiodeModuleHeII, self).__init__()

#     def forward(self, x):
#         pass


# class InterferometerModule(torch.nn.Module):
#     def __init__(self):
#         super(InterferometerModule, self).__init__()

#     def forward(self, x):
#         pass


class RGAPressureModule(torch.nn.Module):
    def __init__(self):
        super(RGAPressureModule, self).__init__()

    def forward(self, x):
        pass


class MagneticFieldModule(torch.nn.Module):
    def __init__(self):
        super(MagneticFieldModule, self).__init__()

    def forward(self, x):
        pass


class ProbeTimeSeriesModule(torch.nn.Module):
    def __init__(self):
        super(ProbeTimeSeriesModule, self).__init__()

    def forward(self, x):
        pass
