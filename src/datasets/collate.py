import math

import torch


def pad_sequence(data: list[torch.Tensor], padding_item: float | int = 0) -> torch.Tensor:
    """
    Pad sequence of tensors [1, ..., variable_lenght] -> [batch_size, ..., max(variable_lenght)]

    Args:
        data: list if torch.Tensor with identical shapes except last one
    Returns:
        batch: torch.Tensor with shape [len(data), ..., max(data.shape[-1])]
    """
    max_lenght = max(item.shape[-1] for item in data)

    padded_data = []
    for item in data:
        time_padding = max_lenght - item.shape[-1]
        padded_item = torch.nn.functional.pad(item, (0, time_padding), mode="constant", value=padding_item).squeeze(0)
        padded_data.append(padded_item)
    return torch.stack(padded_data)



def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    spectrogram_length = torch.tensor([item["spectrogram"].shape[-1] for item in dataset_items])
    # 1e-6 because this is "default" value for 0 in old spectrogram
    # now i use log(spectrogram + 1e-6)
    # TODO: somehow tell this thing that eps = 1e-6
    spectrogram = pad_sequence([item["spectrogram"] for item in dataset_items], padding_item=math.log(1e-6))

    text_encoded_length = torch.tensor([item["text_encoded"].shape[-1] for item in dataset_items])
    # TODO: somehow tell this thing that padding token is EMPTY_IND
    text_encoded = pad_sequence([item["text_encoded"] for item in dataset_items], padding_item=0)
    return {
        "spectrogram_length": spectrogram_length,
        "spectrogram": spectrogram,
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,

        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
