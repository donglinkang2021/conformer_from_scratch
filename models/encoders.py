# PyTorch
import torch
import torch.nn as nn

# Blocks
from models.blocks import (
    ConformerBlock
)

# Modules
from models.modules import (
    AudioPreprocessing,
    SpecAugment,
    Conv2dSubsampling
)

# Positional Encodings and Masks
from models.attentions import (
    SinusoidalPositionalEncoding,
    StreamingMask
)

###############################################################################
# Encoder Models
###############################################################################

class ConformerEncoder(nn.Module):

    def __init__(self, params):
        super(ConformerEncoder, self).__init__()

        # Audio Preprocessing
        self.preprocessing = AudioPreprocessing(
            **params["prepocess_params"]
        )
        
        # Spec Augment
        self.augment = SpecAugment(
            **params["specaug_params"]
        )

        # Subsampling Module
        self.subsampling_module = Conv2dSubsampling(
            **params["conv2d_params"]
        )
        
        # Padding Mask
        self.padding_mask = StreamingMask(
            left_context = params["max_pos_encoding"], 
            right_context = params["max_pos_encoding"]
        )

        # Linear Proj
        self.linear = nn.Linear(
            params["conv2d_params"]["filters"][-1] * params["prepocess_params"]["n_mels"] // 2**params["conv2d_params"]["num_layers"], 
            params["dim_model"]
        )

        # Dropout
        self.dropout = nn.Dropout(p=params["Pdrop"])

        # Sinusoidal Positional Encodings
        self.pos_enc = None

        # Conformer Blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim_model = params["dim_model"],
                dim_expand = params["dim_model"],
                ff_ratio = params["ff_ratio"],
                num_heads = params["num_heads"], 
                kernel_size = params["kernel_size"], 
                att_group_size =  1,
                att_kernel_size = None,
                linear_att = False,
                Pdrop = params["Pdrop"], 
                relative_pos_enc = params["relative_pos_enc"], 
                max_pos_encoding = params["max_pos_encoding"],
                conv_stride =  1,
                att_stride =  1,
                causal = False
            ) for block_id in range(params["num_blocks"])
        ])

    def forward(self, x, x_len=None):

        # Audio Preprocessing
        x, x_len = self.preprocessing(x, x_len)

        # Spec Augment
        if self.training:
            x = self.augment(x, x_len)

        # Subsampling Module
        x, x_len = self.subsampling_module(x, x_len)

        # Padding Mask
        mask = self.padding_mask(x, x_len)

        # Transpose (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)

        # Linear Projection
        x = self.linear(x)

        # Dropout
        x = self.dropout(x)

        # Conformer Blocks
        attentions = []
        for block in self.blocks:
            x, attention, hidden = block(x, mask)
            attentions.append(attention)

            # Strided Block
            if block.stride > 1:

                # Stride Mask (B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if x_len is not None:
                    x_len = torch.div(x_len - 1, block.stride, rounding_mode='floor') + 1

        return x, x_len, attentions
