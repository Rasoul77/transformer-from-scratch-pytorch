# This file is covered by the LICENSE file in the root of this project.

"""
train.py — Training entry script for the Transformer model.

This script demonstrates how to train the `TransformerModel` class
for machine translation using the `TransformerTrainer` class.

Workflow:
    1. Parse command-line arguments (YAML config path, logging options).
    2. Load the configuration file and enrich it with runtime parameters.
    3. Initialize a `TransformerTrainer` with the loaded config.
    4. Run the training and evaluation loop across epochs.

The YAML configuration file defines all model hyperparameters,
training settings, dataset information, and checkpoint paths.
This separation of config and code allows reproducible experiments
and easy switching between datasets or model sizes.

Optional logging is supported through [Weights & Biases](https://wandb.ai/),
which can be enabled from the command line.

Typical usage:
    python3 train.py --config_path config/de_en/config_de_en.yaml
    python3 train.py --config_path config/de_en/config_de_en.yaml --use_wandb --wandb_project "Transformer-DE-EN"
"""

import argparse
import yaml

from argparse import Namespace

from trainer import TransformerTrainer


def get_args() -> Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Namespace containing:
          - config_path: Path to the YAML config file.
          - use_wandb: Flag indicating whether to enable Weights & Biases logging.
          - wandb_project: Project name for W&B logging.
    """

    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument(
        "--config_path", required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Shall use WandB for logging?",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="A descriptive name for wandb logging's project.",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    Main entry point for training.

    Steps:
      - Load the configuration YAML file.
      - Add runtime options (wandb usage, config path).
      - Initialize the TransformerTrainer with the config.
      - Call the trainer's run() method to start training.

    Args:
        args: Parsed command-line arguments.
    """

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    config["config_path"] = args.config_path
    config["wandb"] = {
        "enabled": args.use_wandb,
        "project_name": args.wandb_project,
    }

    # Run training
    trainer = TransformerTrainer(config)
    trainer.run()


if __name__ == "__main__":
    args = get_args()

    try:
        main(args)
    except KeyboardInterrupt:
        raise
