import torch
from torch import Tensor, nn

from src.model.rope import MultiHeadSelfAttention, precompute_freqs_cis


class FeedForwardBlock(nn.Module):
    """
    Feed forward part of Conformer layer

    Args:
        n_features: frequencies in spectrogram recorded 
        n_hidden: n_features in intermediate representation
        dropout: dropout probability
    
    """
    def __init__(self, n_features: int, n_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, n_hidden, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_features, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        """
        Args:
            spectrogram: tensor with shape [batch_size, seq_len, n_features]
        Returns:
            spectrogram: tensor with same shape as input
        """
        return self.ffn(spectrogram)


class MultiHeadSelfAttentonBlock(nn.Module):
    """
    Self-attention part of Conformer layer

    Args:
        n_features: frequencies in spectrogram recorded
        n_heads: number of attention heads
        dropout: dropout probability
    """
    def __init__(self, n_features: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_features)
        self.self_attention = MultiHeadSelfAttention(
            n_features=n_features,
            n_heads=n_heads,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spectrogram: Tensor, freq: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            spectrogram: tensor with shape [batch_size, seq_len, n_features]
            freq: tensor of torch.complex64 with shape [n_features // n_heads, seq_len]
            padding_mask: tensor with shape [batch_size, seq_len]
        Returns:
            spectrogram: tensor with same shape as input
        """
        spectrogram = self.layer_norm(spectrogram)
        spectrogram = self.self_attention(spectrogram, freq, padding_mask)
        return self.dropout(spectrogram)


class ConvolutionBlock(nn.Module):
    """
    Convolution part of Conformer layer

    Args:
        n_features: frequencies in spectrogram recorded
        n_channels: number of depthwise convolution layer input channels
        depthwise_kernel_size: kernel size of depthwise convolution layer
        dropout: dropout probability
    """
    def __init__(self, n_features: int, n_channels: int, depthwise_kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_features)
        self.conv = nn.Sequential(
            # Pointwise conv
            nn.Conv1d(
                in_channels=n_features, 
                out_channels=2 * n_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.GLU(dim=1),
            # Depthwise conv
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=n_channels
            ),
            nn.BatchNorm1d(n_channels),
            nn.SiLU(),
            # Poitnwise conv
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_features,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Dropout(dropout)
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        """
        Args:
            spectrogram: tensor with shape [batch_size, seq_len, n_features]
        Returns:
            spectrogram: tensor with same shape as input
        """
        spectrogram = self.layer_norm(spectrogram)
        # Make [batch_size, n_features, seq_len]
        spectrogram = spectrogram.transpose(1, 2)
        spectrogram = self.conv(spectrogram)
        return spectrogram.transpose(1, 2)


class ConformerBlock(nn.Module):
    """
    Conformer base building block

    Args:
        n_features: frequencies in spectrogram recorded
        n_hidden: n_features in intermediate representation for FeedForward block
        n_heads: number of attention heads in MultiHeadSelfAttention block
        depthwise_conv_kernel_size: kernel size of depthwise convolution in Convolution block 
        dropout: dropout probability
    """
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardBlock(n_features, n_hidden, dropout)
        self.attention = MultiHeadSelfAttentonBlock(n_features, n_heads, dropout)
        self.conv = ConvolutionBlock(n_features, n_features, depthwise_conv_kernel_size, dropout)
        self.ffn2 = FeedForwardBlock(n_features, n_hidden, dropout)
        self.layer_norm = nn.LayerNorm(n_features)


    def forward(self, spectrogram: Tensor, freq: Tensor, padding_mask: Tensor | None = None, **batch) -> Tensor:
        """
        Args:
            spectrogram: tensor [batch_size, n_features, seq_len]
            freq: tensor of torch.complex64 with shape [n_features // n_heads, seq_len] for MultiHeadSelfAttention block
            padding_mask: padding mask for MultiHeadSelfAttention block
        """
        spectrogram = spectrogram * 0.5 + self.ffn1(spectrogram)
        spectrogram = spectrogram * 1.0 + self.attention(spectrogram, freq, padding_mask)
        spectrogram = spectrogram * 1.0 + self.conv(spectrogram)
        spectrogram = spectrogram * 0.5 + self.ffn2(spectrogram)
        spectrogram = self.layer_norm(spectrogram)
        return spectrogram


class SubsamplingBlock(nn.Module):
    """
    Convolution subsampling layer for Conformer

    Args:
        out_channels: output channels count
    """
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
    
    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            spectrogram: tensor with shape [batch_size, seq_len, n_features]
            spectrogram_length: tensor with shape [batch_size]
        """
        # [batch_size, in_channels, seq_len, n_features] where in_channels = 1
        spectrogram = spectrogram.unsqueeze(1)
        # [batch_size, out_channels, seq_len // 4, n_features // 4]
        spectrogram = self.subsampling(spectrogram)
        # [batch_size, seq_len // 4, out_channels, n_features // 4]
        spectrogram = spectrogram.transpose(1, 2)
        batch_size, new_seq_len, out_channels, new_n_features = spectrogram.shape
        # [batch_size, new_seq_len, out_channels * new_n_features ]
        spectrogram = spectrogram.contiguous().view(batch_size, new_seq_len, out_channels * new_n_features)
        return spectrogram, ((spectrogram_length + 1) // 2 + 1) // 2


class Conformer(nn.Module):
    """
    Conformer neural network for ASR
    https://arxiv.org/pdf/2005.08100

    Args:
        n_input_features: frequencies in spectrogram recorded
        n_encoder_features: number of features after subsampling
        n_ffn_hidden: n_features in intermediate representation for FeedForward block
        n_heads: number of attention heads in MultiHeadSelfAttention block
        n_layers: number of Conformer block to stack
        depthwise_conv_kernel_size: kernel size of depthwise convolution in Convolution block
        max_seq_len: maximun sequence lenght for rotary embeddings in MultiHeadSelfAttention block
        n_tokens: vocab size
        theta: base for rotary embeddings in MultiHeadSelfAttention block
        dropout: dropout probability
    """
    def __init__(
        self,
        n_input_features: int,
        n_encoder_features: int,
        n_ffn_hidden: int,
        n_heads: int,
        n_layers: int,
        depthwise_conv_kernel_size: int,
        max_seq_len: int,
        n_tokens: int,
        theta: float = 10000.0,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.subsampling = SubsamplingBlock(n_encoder_features)
        self.projector = nn.Linear(
            in_features=n_encoder_features * n_input_features // 4,
            out_features=n_encoder_features
        )
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    n_encoder_features,
                    n_ffn_hidden,
                    n_heads,
                    depthwise_conv_kernel_size,
                    dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.logits = nn.Linear(
            in_features=n_encoder_features,
            out_features=n_tokens,
        )
        freq = precompute_freqs_cis(n_encoder_features // n_heads, max_seq_len * 2, theta=theta)
        self.freq = nn.Parameter(freq, requires_grad=False)
    
    @staticmethod
    def _lengths_to_padding_mask(lengths: Tensor) -> Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
            batch_size, max_length
        ) < lengths.unsqueeze(1)
        return padding_mask

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch) -> dict:
        """
        Args:
            spectrogram: input spectrogram with shape [batch_size, n_features, seq_len]
            spectrogram_length: spectrogram original lengths with shape [batch_size]
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.transpose(1, 2)

        spectrogram, spectrogram_length = self.subsampling(spectrogram, spectrogram_length)
        spectrogram = self.projector(spectrogram)

        seq_len = spectrogram.shape[1]
        freq = self.freq[:seq_len, :]
        padding_mask = self._lengths_to_padding_mask(spectrogram_length).to(device=spectrogram.device)

        for layer in self.layers:
            spectrogram = layer(spectrogram, freq, padding_mask)
        
        # [batch_size, seq_len, n_tokens]
        logits = self.logits(spectrogram)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": spectrogram_length}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
