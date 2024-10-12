import math
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from string import ascii_lowercase

import torch
from torchaudio.models.decoder import ctc_decoder

from src.datasets.collate import pad_sequence

INF = float('inf')


class CTCBaseDecoder(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class CTCArgmaxDecoder(CTCBaseDecoder):
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        ids = logits.argmax(dim=-1)
        return pad_sequence(
            [
                torch.tensor([
                    x for x, _ in itertools.groupby(sample)
                ])
                for sample in ids
            ],
            padding_item=0
        )


class CTCBeamSearchDecoderFast(CTCBaseDecoder):
    def  __init__(self, beam_size: int = 50) -> None:
        # TODO: tell this thing what is an actual vocab
        vocab = [""] + list(ascii_lowercase + " ")
        self.decoder = ctc_decoder(
            lexicon=None,
            tokens=vocab,
            blank_token=vocab[0],
            sil_token=" ",
            beam_size=beam_size
        )
    
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.detach().cpu()
        result = self.decoder(logits)
        return pad_sequence(
            [
                res[0].tokens
                for res in result
            ],
            padding_item=0
        )


class CTCBeamSearchDecoder(CTCBaseDecoder):
    def __init__(self, beam_size: int = 100, blank: int = 0) -> None:
        self.beam_size = beam_size
        self.blank = blank

    @staticmethod
    def stable_log_prob_sum(*args: float) -> float:
        """
        Calculates log(\\sum_{k=1}^n exp(a_k)) numericly stable 
        """
        if all(a == -INF for a in args):
            return -INF
        maximum = max(args)
        return maximum + math.log(sum(math.exp(x - maximum) for x in args))
    
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.detach().cpu()
        if logits.ndim == 3:
            return pad_sequence(
                [
                    self.decode(logits_sinlge_batch)
                    for logits_sinlge_batch in logits
                ],
                padding_item=0
            )

        current_beams = [
            (tuple(), (0.0, -INF))
        ]

        for t in range(logits.shape[0]):
            next_beams = defaultdict(lambda: (-INF, -INF)) # so if no such prefix exists then all probs are 0
            for c in range(logits.shape[1]):
                p = logits[t, c].item()
                for prefix, (p_bl, p_n_bl) in current_beams:
                    if c == self.blank:
                        next_p_bl, next_p_n_bl = next_beams[prefix]
                        # for this prefix with black at the end we have new 2 valid paths
                        next_p_bl = self.stable_log_prob_sum(next_p_bl, p_bl + p, p_n_bl + p)
                        next_beams[prefix] = (next_p_bl, next_p_n_bl)
                    else: # c != blank
                        end_prefix_char = prefix[-1] if prefix else None
                        new_prefix = prefix + (c,)

                        next_p_bl, next_p_n_bl = next_beams[new_prefix]

                        if end_prefix_char == c:
                            # if chars are same we can add only to prefix with blank
                            next_p_n_bl = self.stable_log_prob_sum(next_p_n_bl, p_bl + p)

                            # But we can actually append char, that only update old prefix
                            old_p_bl, old_p_n_bl = next_beams[prefix]
                            old_p_n_bl = self.stable_log_prob_sum(old_p_n_bl, p_n_bl + p)
                            next_beams[prefix] = (old_p_bl, old_p_n_bl)
                        else:
                            # no merge would happend, 
                            next_p_n_bl = self.stable_log_prob_sum(next_p_n_bl, p_bl + p, p_n_bl + p)

                        next_beams[new_prefix] = (next_p_bl, next_p_n_bl)
            current_beams = sorted(
                next_beams.items(),
                key=lambda x : self.stable_log_prob_sum(*x[1]),
                reverse=True
            )
            current_beams = current_beams[:self.beam_size]

        best, (p_bl, p_n_bl) = current_beams[0]
        return torch.tensor(best)
