from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.text_encoder import CTCBaseDecoder, CTCTextEncoder


class CERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, token_decoder: CTCBaseDecoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.token_decoder = token_decoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = self.token_decoder.decode(log_probs.cpu()).numpy()
        lengths = log_probs_length.cpu().detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
