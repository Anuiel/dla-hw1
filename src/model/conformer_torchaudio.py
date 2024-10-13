import torch
from torch import nn
from torchaudio.models import Conformer

from src.model.conformer import SubsamplingBlock


class ConformerTorchAudio(Conformer):
    def __init__(
        self,
        n_input_features: int,
        n_encoder_features: int,
        n_ffn_hidden: int,
        n_heads: int,
        n_layers: int,
        depthwise_conv_kernel_size: int,
        n_tokens: int,
        dropout: float,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__(
            input_dim=n_encoder_features,
            num_heads=n_heads,
            ffn_dim=n_ffn_hidden,
            num_layers=n_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )
        self.subsampling = SubsamplingBlock(n_encoder_features)
        self.input_projector = nn.Linear(
            in_features=n_encoder_features * n_input_features // 4,
            out_features=n_encoder_features,
        )
        self.output_projector = nn.Linear(n_encoder_features, n_tokens)

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch
    ) -> dict:
        """
        Args:
            spectrogram: spectrogram spectrogram with shape [batch_size, n_features, seq_len]
            spectrogram_length: spectrogram original lengths with shape [batch_size]
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.transpose(1, 2)
        spectrogram, spectrogram_length = self.subsampling(
            spectrogram, spectrogram_length
        )
        spectrogram = self.input_projector(spectrogram)

        output, spectrogram_length = super().forward(spectrogram, spectrogram_length)
        predictions = self.output_projector(output)
        log_probs = nn.functional.log_softmax(predictions, dim=-1)
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
