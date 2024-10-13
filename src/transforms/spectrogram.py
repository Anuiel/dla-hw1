import torch
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(MelSpectrogram):
    def __init__(self, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = super().forward(waveform)
        return torch.log(spec + self.eps)
