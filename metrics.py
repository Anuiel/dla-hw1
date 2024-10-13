import json
import warnings
from pathlib import Path

import click
import hydra
from hydra.utils import instantiate
from tqdm import tqdm

from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    """
    Script for easy CER/WER evaluation over existing predictions and ground truth

    Args:
        predictions_dir: Directory with predicted text
        ground_truth_dir: Directory with ground truth text
    """
    predictions_dir = Path(config.predictions_dir)
    ground_truth_dir = Path(config.ground_truth_dir)
    text_encoder = instantiate(config.text_encoder)

    if not predictions_dir.exists() or not predictions_dir.is_dir():
        print(
            f"Prediction dir {config.predictions_dir} is not existing or not a directory"
        )
        return

    if not ground_truth_dir.exists() or not ground_truth_dir.is_dir():
        print(
            f"Ground truth dir {config.ground_truth_dir} is not existing or not a directory"
        )
        return

    metric_tracker = MetricTracker("cer", "wer")
    for pred_path in tqdm(predictions_dir.iterdir()):
        name = pred_path.name
        gt_path = ground_truth_dir / name

        with pred_path.open() as f:
            pred_text = f.read().strip()
        if gt_path.exists():
            with gt_path.open() as f:
                gt_text = f.read().strip()
        else:
            print(pred_path, gt_path)
            print(f"Ground truth for {name} not found. Skipping.")
            continue

        pred_text = text_encoder.normalize_text(pred_text)
        gt_text = text_encoder.normalize_text(gt_text)
        metric_tracker.update("cer", calc_cer(pred_text, gt_text))
        metric_tracker.update("wer", calc_wer(pred_text, gt_text))

    print(json.dumps(metric_tracker.result(), indent=4))


if __name__ == "__main__":
    main()
