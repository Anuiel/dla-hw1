import torch
from torch import Tensor, nn

from src.model.relative_attention import ConformerMultiHeadedSelfAttentionModule as AttentionBlock
from src.model.conformer import FeedForwardBlock, ConvolutionBlock, SubsamplingBlock


class ConformerRelativeBlock(nn.Module):
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
        self.attention = AttentionBlock(n_features, n_heads, dropout)
        self.conv = ConvolutionBlock(n_features, n_features, depthwise_conv_kernel_size, dropout)
        self.ffn2 = FeedForwardBlock(n_features, n_hidden, dropout)
        self.layer_norm = nn.LayerNorm(n_features)


    def forward(self, spectrogram: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            spectrogram: tensor [batch_size, n_features, seq_len]
            freq: tensor of torch.complex64 with shape [n_features // n_heads, seq_len] for MultiHeadSelfAttention block
            padding_mask: padding mask for MultiHeadSelfAttention block
        """
        spectrogram = spectrogram * 0.5 + self.ffn1(spectrogram)
        spectrogram = spectrogram * 1.0 + self.attention(spectrogram, padding_mask)
        spectrogram = spectrogram * 1.0 + self.conv(spectrogram)
        spectrogram = spectrogram * 0.5 + self.ffn2(spectrogram)
        spectrogram = self.layer_norm(spectrogram)
        return spectrogram


class ConformerRelative(nn.Module):
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
        n_tokens: int,
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
                ConformerRelativeBlock(
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
        padding_mask = self._lengths_to_padding_mask(spectrogram_length).to(device=spectrogram.device)

        for layer in self.layers:
            spectrogram = layer(spectrogram, padding_mask)
        
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
