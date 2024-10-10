from abc import ABC, abstractmethod

import torch

class CTCBaseDecoder(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class CTCArgmaxDecoder(CTCBaseDecoder):
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)
