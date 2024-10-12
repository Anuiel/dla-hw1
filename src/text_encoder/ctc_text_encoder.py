import re
import typing as tp
from enum import Enum
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier



class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.EMPTY_IND = self.char2ind[self.EMPTY_TOK]

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )
        
    def decode(self, inds: list[int]) -> str:
        """
        Args:
            inds (list): list of tokens.
        Returns:
            text (str): raw text with empty tokens and repetitions.
        """
        return "".join(self.ind2char[int(x)] for x in inds if x != self.EMPTY_IND).strip()

    def decode_raw(self, inds: list[int]) -> str:
        """
        Args:
            inds (list): list of tokens.
        Returns:
            text (str): raw text with empty tokens and repetitions.
        """
        return "".join(self.ind2char[int(x)] for x in inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
