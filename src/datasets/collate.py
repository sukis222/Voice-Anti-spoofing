import torch
from torch.nn.utils.rnn import pad_sequence


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
    wavs = []
    labels = []

    for el in dataset_items:
        if el['data_object'].squeeze().shape[0] > 100000:
            wavs.append(el['data_object'].squeeze()[:100000])
        else:
            wavs.append(el['data_object'].squeeze())
        labels.append(el['label'])

    x = 100000 - wavs[0].shape[0]
    wavs[0] = torch.nn.functional.pad(wavs[0], (0, x))

    wavs = pad_sequence(wavs, batch_first=True)
    labels = torch.Tensor(labels).long()
    return {
        "data_object": wavs,
        "labels": labels
    }
