import torch
from torch import nn
from torchaudio.models import Conformer


class ConformerTorchAudio(Conformer):
    def __init__(self
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        n_tokens: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False
    ):
        super().__init__(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first
        )
        self.proj = nn.Linear(input_dim, n_tokens)
    
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, lenght = super().forward(input, lengths)
        predictions = self.proj(output)
        return predictions, lenght
