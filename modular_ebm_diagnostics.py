import torch
from functools import partial

# class DiagnosticPositionModule(torch.nn.Module):
#     def __init__(self):
#         super(DiagnosticPositionModule, self).__init__()

#     def forward(self, x):
#         pass


# class DiagnosticOrientationModule(torch.nn.Module):
#     def __init__(self):
#         super(DiagnosticOrientationModule, self).__init__()

#     def forward(self, x):
#         pass


# class ProbePositionModule(torch.nn.Module):
#     def __init__(self):
#         super(ProbePositionModule, self).__init__()

#     def forward(self, x):
#         pass

def getDeviceString(device_num):
    if device_num <= -1:
        return 'cpu'
    if device_num >= 0:
        return 'cuda:{}'.format(device_num)


class AddAndNorm(torch.nn.Module):
    def __init__(self, seq_length, embed_dim):
        super(AddAndNorm, self).__init__()
        # These should automatically move to GPU since they're registered
        self.layerNorm = torch.nn.LayerNorm([seq_length, embed_dim])

    def forward(self, x1, x2):
        x = self.layerNorm(x1 + x2)
        return x


# Input shape is (batch size, seq_length, num_channels/features)
# Implemented based on "Attention is all you need"
class PositionWiseFFNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(PositionWiseFFNN, self).__init__()

        # These should automatically move to GPU since they're registered
        # Don't need bias -- layer normalization should take care of that
        self.dense1 = torch.nn.Linear(num_inputs, num_hidden, bias=False)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(num_hidden, num_outputs, bias=False)

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = x.transpose(1, 2)
        return x


class ResidualAttnBlock(torch.nn.Module):
    def __init__(self, seq_length, embed_dim, num_heads, num_hidden):
        super(ResidualAttnBlock, self).__init__()

        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads,
                                                batch_first=True)
        self.addNorm1 = AddAndNorm(seq_length, embed_dim)
        self.posWiseNN = PositionWiseFFNN(embed_dim, num_hidden, embed_dim)
        self.addNorm2 = AddAndNorm(seq_length, embed_dim)

    def forward(self, x_resid):
        x = self.attn(x_resid, x_resid, x_resid)[0]  # Just the output, no weights
        x_resid = self.addNorm1(x, x_resid)
        x = self.posWiseNN(x_resid)
        x = self.addNorm2(x, x_resid)
        return x


class SequencePositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, seq_length):
        super(SequencePositionalEncoding, self).__init__()

        # I'm defining tensors here so they will NOT automatically be moved to the correct GPU.
        # I'll fix this on the forward pass.
        self.posEncoding = torch.zeros((1, seq_length, embed_dim))
        i = torch.arange(0, seq_length).reshape(seq_length, 1)
        j2 = torch.arange(0, embed_dim, 2).reshape(1, embed_dim // 2)
        posVal = i / torch.pow(10000, j2 / embed_dim)

        self.posEncoding[:, :, 0::2] = torch.sin(posVal)
        self.posEncoding[:, :, 1::2] = torch.cos(posVal)

    def forward(self, x):
        device = getDeviceString(x.get_device())

        x = x + self.posEncoding.to(device)
        return x


class DiagnosticPositionModule(torch.nn.Module):
    def __init__(self, num_layers, width_layer, out_size):
        super(DiagnosticPositionModule, self).__init__()

        self.num_layers = num_layers
        self.width_layer = width_layer
        self.out_size = out_size

        self.relu = torch.nn.ReLU()
        self.dense = partial(torch.nn.LazyLinear, self.width_layer, bias=True)
        self.dense_out = torch.nn.LazyLinear(self.out_size)

        self.layers = torch.nn.ModuleList([self.dense() for i in range(self.num_layers)])

    # x is the position of the diagnostic. Maybe include which direction it's pointing?
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            x = self.relu(x)
        x = self.dense_out(x)

        return x


# memory is shape batch x seq_length x embed_dim
class MSITimeSeriesModule(torch.nn.Module):
    def __init__(self, seq_length, embed_dim, num_heads, num_hidden,
                 num_msi_attn, num_mem_attn, num_sum_attn):
        super(MSITimeSeriesModule, self).__init__()

        # CNN setup
        out_channels = embed_dim
        kernel_size = 4
        self.zeroPad = torch.nn.ConstantPad1d((1, 2), 0)
        # Attention setup
        # num_heads = 4
        # num_hidden = 256

        self.seqPosEnc = SequencePositionalEncoding(out_channels, seq_length)

        # Signal processing branch
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv2 = torch.nn.LazyConv1d(out_channels, kernel_size)

        self.msiAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
            for i in range(num_msi_attn)])

        self.memAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
            for i in range(num_mem_attn)])

        self.sumAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, out_channels, num_heads, num_hidden)
            for i in range(num_sum_attn)])

    # shared_memory: a tensor that this module will read/write to
    # pos_module: module encoding the physical position of the diagnostic
    # ^ these will be included later
    # orient_module: module encoding the orientation of the diagnostic
    def forward(self, x, shared_memory, unsqueeze=True):
        # x is shape (batch_size, seq_length)
        if unsqueeze:
            x = torch.unsqueeze(x, 1)  # Make into shape (batch_size, channels, seq_length)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv1(x)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv2(x)
        x = x.transpose(1, 2)

        # self attention time :)
        x = self.seqPosEnc(x)  # encode the position for the self-attention blocks
        for i, block in enumerate(self.msiAttnBlocks):
            x = block(x)
        # should now have shape (batch_size, channels, seq_length), which is the same as memory

        # process the memory using attention
        x_mem = self.seqPosEnc(shared_memory)
        for i, block in enumerate(self.memAttnBlocks):
            x_mem = block(x_mem)

        # add the memory and the processed diagnostic together
        x = x + x_mem

        # figure out which items to write to memory
        x = self.seqPosEnc(x)
        for i, block in enumerate(self.sumAttnBlocks):
            x = block(x)

        return x


class CNNResidualBlock(torch.nn.Module):
    def __init__(self, seq_length, embed_dim, kernel_size):
        super(CNNResidualBlock, self).__init__()

        # CNN setup
        self.seq_length = seq_length
        self.embed_dim = out_channels = embed_dim
        self.kernel_size = kernel_size

        self.relu = torch.nn.ReLU()
        self.zeroPad = torch.nn.ConstantPad1d((1, 2), 0)
        self.conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv2 = torch.nn.LazyConv1d(out_channels, 1)
        self.conv3 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv4 = torch.nn.LazyConv1d(out_channels, 1)
        self.conv5 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv6 = torch.nn.LazyConv1d(out_channels, 1)
        self.dense1 = torch.nn.LazyLinear(64)
        self.dense2 = torch.nn.LazyLinear(seq_length * embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)

        x_temp = x
        x = self.zeroPad(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + x_temp

        x_temp = x
        x = self.zeroPad(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = x + x_temp

        x_temp = x
        x = self.zeroPad(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        # x = x + x_temp

        x = x.transpose(1, 2)

        x = self.dense1(x.reshape(-1, self.seq_length * self.embed_dim))
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = x.reshape((-1, self.seq_length, self.embed_dim))

        return x


class MSICNNModule(torch.nn.Module):
    def __init__(self, seq_length, embed_dim, kernel_size):
        super(MSICNNModule, self).__init__()

        # CNN setup
        self.seq_length = seq_length
        self.embed_dim = out_channels = embed_dim
        self.kernel_size = kernel_size
        self.zeroPad = torch.nn.ConstantPad1d((1, 2), 0)

        # Signal processing branch
        # self.relu = torch.nn.ReLU()
        # self.conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        # self.conv2 = torch.nn.LazyConv1d(out_channels, kernel_size)
        # self.dense1 = torch.nn.LazyLinear(seq_length * embed_dim, bias=True)

        self.msiBlock = CNNResidualBlock(seq_length, embed_dim, kernel_size)
        self.memBlock = CNNResidualBlock(seq_length, embed_dim, kernel_size)
        self.sumBlock = CNNResidualBlock(seq_length, embed_dim, kernel_size)

    def forward(self, x, shared_memory):
        # x is shape (batch_size, seq_length)
        # x = torch.unsqueeze(x, 1)  # Make into shape (batch_size, channels, seq_length)
        # x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = x.transpose(1, 2)
        # x = self.pw_ffnn(x)
        # x = self.relu(x)
        # x = self.dense1(x.reshape((-1, self.seq_length * self.embed_dim))).reshape((-1, self.seq_length, self.embed_dim))
        # x = x.transpose(1, 2)

        x = torch.unsqueeze(x, 2)
        x = self.msiBlock(x)
        x_mem = self.memBlock(shared_memory)
        x = self.sumBlock(x + x_mem)

        # x = x + x_mem
        return x


class RGADenseModule(torch.nn.Module):
    def __init__(self, seq_length, dense_size, embed_dim, kernel_size):
        super(RGADenseModule, self).__init__()
        self.embed_dim = out_channels = embed_dim
        self.kernel_size = kernel_size

        self.seq_length = seq_length
        self.dense_size = dense_size

        self.zeroPad = torch.nn.ConstantPad1d((1, 2), 0)
        self.relu = torch.nn.ReLU()
        self.dense1 = torch.nn.LazyLinear(dense_size, bias=True)
        self.dense2 = torch.nn.LazyLinear(dense_size, bias=True)
        self.dense3 = torch.nn.LazyLinear(seq_length * embed_dim, bias=True)
        # self.mem_conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        # self.mem_conv2 = torch.nn.LazyConv1d(out_channels, kernel_size)
        # self.mem_pw_ffnn = PositionWiseFFNN(out_channels, 32, out_channels)

        self.memBlock = CNNResidualBlock(seq_length, embed_dim, kernel_size)
        self.sumBlock = CNNResidualBlock(seq_length, embed_dim, kernel_size)

    def forward(self, x, shared_memory):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x).reshape((-1, self.seq_length, self.embed_dim))

        # x_mem = shared_memory.transpose(1, 2)
        # x_mem = self.zeroPad(x_mem)  # Add padding to get CNN output seq length to 256
        # x_mem = self.mem_conv1(x_mem)
        # x_mem = self.relu(x_mem)
        # x_mem = self.zeroPad(x_mem)  # Add padding to get CNN output seq length to 256
        # x_mem = self.mem_conv2(x_mem)
        # x_mem = self.relu(x_mem)
        # x_mem = x_mem.transpose(1, 2)
        # x_mem = self.mem_pw_ffnn(x_mem)

        x_mem = self.memBlock(shared_memory)
        x = self.sumBlock(x + x_mem)

        # x = x + x_mem
        return x


class RGAPressureModule(torch.nn.Module):
    def __init__(self, dense_width, seq_length, embed_dim, num_heads, num_hidden,
                 num_msi_attn, num_mem_attn, num_sum_attn):
        super(RGAPressureModule, self).__init__()

        self.dense_width = dense_width
        self.seq_length = seq_length
        # embed_dim = 16
        # Attention setup
        # num_heads = 4
        # num_hidden = 256

        self.seqPosEnc = SequencePositionalEncoding(embed_dim, seq_length)

        # Signal processing branch
        self.relu = torch.nn.ReLU()
        self.dense1 = torch.nn.LazyLinear(dense_width)
        self.dense2 = torch.nn.LazyLinear(dense_width)
        # self.linearExpander = torch.nn.LazyLinear(seq_length, bias=False)
        self.posWiseNN = PositionWiseFFNN(dense_width // seq_length, num_hidden, embed_dim)

        self.msiAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, embed_dim, num_heads, num_hidden)
            for i in range(num_msi_attn)])

        self.memAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, embed_dim, num_heads, num_hidden)
            for i in range(num_mem_attn)])

        self.sumAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(seq_length, embed_dim, num_heads, num_hidden)
            for i in range(num_sum_attn)])

    def forward(self, x, shared_memory):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = self.linearExpander(x)
        x = self.posWiseNN(x.reshape(-1, self.seq_length, self.dense_width // self.seq_length))
        # x = x.reshape(-1, self.seq_length, 16)
        x = self.seqPosEnc(x)

        for i, block in enumerate(self.msiAttnBlocks):
            x = block(x)
        # should now have shape (batch_size, channels, seq_length), which is the same as memory

        # process the memory using attention
        x_mem = self.seqPosEnc(shared_memory)
        for i, block in enumerate(self.memAttnBlocks):
            x_mem = block(x_mem)

        # add the memory and the processed diagnostic together
        x = x + x_mem

        # figure out which items to write to memory
        x = self.seqPosEnc(x)
        for i, block in enumerate(self.sumAttnBlocks):
            x = block(x)

        return x


class MagneticFieldModule(torch.nn.Module):
    def __init__(self, dense_width, seq_length, embed_dim, num_heads, num_hidden,
                 num_msi_attn, num_mem_attn, num_sum_attn):
        super(MagneticFieldModule, self).__init__()

        self.dense_width = dense_width
        self.seq_length = seq_length

        self.relu = torch.nn.ReLU()
        self.dense1 = torch.nn.LazyLinear(dense_width)
        self.dense2 = torch.nn.LazyLinear(dense_width)
        # Don't need to multiply by embed_dim because the time series module takes care of that
        # self.linearExpander = torch.nn.LazyLinear(seq_length, bias=False)
        self.msiProcessor = MSITimeSeriesModule(seq_length, embed_dim, num_heads, num_hidden,
                                                num_msi_attn, num_mem_attn, num_sum_attn)

    def forward(self, x, shared_memory):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = self.linearExpander(x)
        x = x.reshape(-1, self.dense_width // self.seq_length, self.seq_length)
        x = self.msiProcessor(x, shared_memory, unsqueeze=False)
        return x


# class ProbeTimeSeriesModule(torch.nn.Module):
#     def __init__(self):
#         super(ProbeTimeSeriesModule, self).__init__()

#     def forward(self, x):
#         pass
