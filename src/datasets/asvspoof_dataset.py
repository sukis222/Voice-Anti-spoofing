import logging
from typing import List, Dict
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
from src.utils.io_utils import ROOT_PATH, read_json, write_json


from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ASVSpoofDataset(BaseDataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
            self,
            asv_root_path: str,
            asv_protocol_path: str,
            access_type: str,  # 'LA' or 'PA'
            stage: str,  # 'train', 'dev', 'eval'
            *args,
            **kwargs,
    ):
        self.asv_root_path = Path(asv_root_path)
        self.asv_protocol_path = asv_protocol_path
        self.access_type = access_type
        self.stage = stage

        index = self.create_index()
        super().__init__(index, *args, **kwargs)

    def create_index(self) -> List[Dict]:
        index = []
        with open(self.asv_protocol_path, 'r') as protocol:
            for line in protocol:
                split_line = line.split()
                id = split_line[1]
                label_str = split_line[4]
                if label_str == "bonafide":
                    label = 0
                else:
                    label = 1

                key_path = self.asv_root_path / self.access_type / self.access_type / f"ASVspoof2019_{self.access_type}_{self.stage}" / "flac" / f"{id}.flac"

                index.append({
                    "path" : str(key_path),
                    "label" : label
                })

        return index

    def load_object(self, path:str):
        try:
            waveform, sample_rate = torchaudio.load(path)
            return waveform
        except Exception as e:
            raise e




