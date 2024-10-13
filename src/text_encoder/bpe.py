import io
import re
import typing as tp
from enum import Enum
from string import ascii_lowercase

import sentencepiece as spm
import torch

from src.utils.io_utils import ROOT_PATH


class BPETextEncoder:
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, path: str, train_dataset=None, vocab_size: int | None = None):
        """
        Args:
            path: path to pretrained tokenizer. If None, new will be created and saved
            train_datatset (BaseDataset): dataset to train tokenizer on. Ignored if want to load existing
            vocab_size: vocab size to create tokenizer. Ignored if want to load existing
        """
        full_path = ROOT_PATH / path
        if full_path.is_file():
            print("Importing pretained tokenizer")
            self.tokenizer = spm.SentencePieceProcessor(
                model_file=str(ROOT_PATH / path)
            )
        else:
            print("Training new tokenizer")

            self.train(save_path=path, dataset=train_dataset, vocab_size=vocab_size)
            self.tokenizer = spm.SentencePieceProcessor(
                model_file=str(ROOT_PATH / path)
            )

    def __len__(self) -> int:
        return self.tokenizer.vocab_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.tokenizer.decode(item)

    def encode(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)

    def decode(self, inds: list[int]) -> str:
        """
        Args:
            inds (list): list of tokens.
        Returns:
            text (str): decoded text.
        """
        return self.tokenizer.decode(
            [int(x) for x in inds if int(x) != self.UNK_ID]
        ).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    @classmethod
    def dataset_iterator(cls, dataset):
        for x in dataset:
            yield cls.normalize_text(x["text"])

    @classmethod
    def train(
        cls, save_path: str, dataset, vocab_size: int
    ):  # should be typing there, but circular input
        model = io.BytesIO()

        spm.SentencePieceTrainer.train(
            sentence_iterator=cls.dataset_iterator(dataset),
            model_writer=model,
            model_type="bpe",
            vocab_size=vocab_size,
            pad_id=cls.PAD_ID,
            unk_id=cls.UNK_ID,
            # For easy check for LM scoring when word is ending
            treat_whitespace_as_suffix=True,
            # not needed in this task
            bos_id=-1,
            eos_id=-1,
        )

        with open(str(ROOT_PATH / save_path), "wb") as f:
            f.write(model.getvalue())
