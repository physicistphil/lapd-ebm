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
                                                batch_first=True)
        self.addNorm1 = AddAndNorm()
        self.posWiseNN = PositionWiseFFNN(embed_dim, num_hidden, embed_dim)
        self.addNorm2 = AddAndNorm()

    def forward(self, x_resid):
        x = self.attn(x_resid, x_resid, x_resid, need_weight=False)
        x_resid = self.addNorm1(x, x_resid)
        x = self.posWiseNN(x_resid)
        x = self.addNorm2(x, x_resid)
        return x


class SequencePositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, seq_length):
        super(SequencePositionalEncoding, self).__init__()

        self.posEncoding = torch.zeros((1, seq_length, embed_dim))

        i = torch.arange(0, seq_length).reshape(seq_length, 1)
        j2 = torch.arange(0, embed_dim, 2).reshape(1, embed_dim // 2)
        posVal = i / torch.pow(10000, j2 / embed_dim)

        self.posEncoding[:, :, 0::2] = torch.sin(posVal)
        self.posEncoding[:, :, 1::2] = torch.cos(posVal)

    def forward(self, x):
        x = x + self.posEncoding
        return x


# memory is shape batch x 16 x 256 
class MSITimeSeriesModule(torch.nn.Module):
    def __init__(self, seq_length, num_msi_attn, num_mem_attn, num_sum_attn):
        super(MSITimeSeriesModule, self).__init__()

        # CNN setup
        out_channels = 16
        kernel_size = 4
        self.zeroPad = torch.nn.ConstantPad1d(0, (1, 2))
        # Attention setup
        num_heads = 4
        num_hidden = 256

        self.seqPosEnc = SequencePositionalEncoding(out_channels, seq_length)

        # Signal processing branch
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.LazyConv1d(out_channels, kernel_size)
        self.conv2 = torch.nn.LazyConv1d(out_channels, kernel_size)

        self.msiAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden)
            for i in range(num_msi_attn)])

        self.memAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden
            for i in range(num_mem_attn))])

        self.sumAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden
            for i in range(num_sum_attn))])

    # shared_memory: a tensor that this module will read/write to
    # pos_module: module encoding the physical position of the diagnostic 
    # ^ these will be included later
    # orient_module: module encoding the orientation of the diagnostic
    def forward(self, x, shared_memory):
        # x is shape (batch_size, seq_length)
        x = torch.unsqueeze(x, 1)  # Make into shape (batch_size, channels, seq_length)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv1(x)
        x = self.zeroPad(x)  # Add padding to get CNN output seq length to 256
        x = self.conv2(x)
        x = self.seqPosEnc(x)  # encode the position for the self-attention blocks
        for i, block in enumerate(self.msiAttnBlocks):
            x = block(x)
        # should now have shape (batch_size, channels, seq_length), which is the same as memory

        # process the memory using attention
        for i, block in enumerate(self.memAttnBlocks):
            x_mem = block(shared_memory)

        # add the memory and the processed diagnostic together
        x = x + x_mem

        # figure out which items to write to memory
        for i, block in enumerate(self.sumAttnBlocks):
            x = block(x)

        return x


class RGAPressureModule(torch.nn.Module):
    def __init__(self, seq_length, num_msi_attn, num_mem_attn, num_sum_attn):
        super(RGAPressureModule, self).__init__()
        
        # Attention setup
        num_heads = 4
        num_hidden = 256

        self.seqPosEnc = SequencePositionalEncoding(out_channels, seq_length)

        # Signal processing branch
        self.relu = torch.nn.ReLU()
        self.dense1 = torch.nn.LazyLinear(seq_length * 16)
        self.dense2 = torch.nn.LazyLinear(seq_length * 16)

        self.msiAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden)
            for i in range(num_msi_attn)])

        self.memAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden
            for i in range(num_mem_attn))])

        self.sumAttnBlocks = torch.nn.ModuleList([
            ResidualAttnBlock(out_channels, num_heads, num_hidden
            for i in range(num_sum_attn))])

    def forward(self, x, shared_memory):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = x.reshape(-1, 16, seq_length)
        x = self.seqPosEnc(x)

        for i, block in enumerate(self.msiAttnBlocks):
            x = block(x)
        # should now have shape (batch_size, channels, seq_length), which is the same as memory

        # process the memory using attention
        for i, block in enumerate(self.memAttnBlocks):
            x_mem = block(shared_memory)

        # add the memory and the processed diagnostic together
        x = x + x_mem

        # figure out which items to write to memory
        for i, block in enumerate(self.sumAttnBlocks):
            x = block(x)

        return x


class MagneticFieldModule(torch.nn.Module):
    def __init__(self, seq_length, num_msi_attn, num_mem_attn, num_sum_attn):
        super(MagneticFieldModule, self).__init__()

        self.linearExpander = torch.nn.LazyLinear(seq_length * 16)
        self.msiProcessor = MSITimeSeriesModule(seq_length, num_msi_attn,
                                                num_mem_attn, num_sum_attn)

    def forward(self, x, shared_memory):
        x = self.linearExpander(x)
        x = x.reshape(-1, 16, seq_length)
        x = self.msiProcessor(x, shared_memory)
        return x
        


class ProbeTimeSeriesModule(torch.nn.Module):
    def __init__(self):
        super(ProbeTimeSeriesModule, self).__init__()

    def forward(self, x):
        pass
