from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self, audio_dir, save_path=None, transcription_dir=None, *args, **kwargs
    ):
        data = []
        if save_path is not None:
            Path(save_path).mkdir(exist_ok=True, parents=True)
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["path"] = str(path)
                entry["audio_len"] = length
                if save_path:
                    entry["save_path"] = str(Path(save_path) / (path.stem + ".txt"))
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
