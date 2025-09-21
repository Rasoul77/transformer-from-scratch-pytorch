# This file is covered by the LICENSE file in the root of this project.

"""
This script demonstrates how to create SentencePiece tokenizers for both
the input and output languages of a translation dataset.
It loads training data, extracts raw sentences, and fits tokenizers
with the specified vocabulary size and model type.
"""

import argparse
import os
import yaml
from argparse import Namespace
from typing import Dict

import sentencepiece as spm
from datasets import load_dataset


class TokenizerBuilder:
    """
    TokenizerBuilder automates the process of creating SentencePiece
    tokenizers for machine translation tasks.

    The workflow includes:
      - Loading a parallel dataset for the specified language pair.
      - Extracting input and output sentences from the training split.
      - Saving them to temporary files for training SentencePiece.
      - Training separate tokenizers for input and output languages.
      - Saving the tokenizer models to the specified directory.
      - Cleaning up temporary files.

    This class makes it easier to prepare tokenizers consistently
    when experimenting with Transformer-based models.
    """

    def __init__(self, config: Dict):
        """
        Initialize the TokenizerBuilder with configuration settings.

        Args:
            config (Dict): Configuration dictionary loaded from YAML.
                           Must include dataset path, languages, vocab sizes,
                           tokenizer model type, and output directory.
        """

        self.config = config
        self.dataset_path = config["data"]["dataset_path"]
        self.input_lang = config["data"]["input_lang"]
        self.output_lang = config["data"]["output_lang"]
        self.input_vocab_size = config["data"]["input_vocab_size"]
        self.output_vocab_size = config["data"]["output_vocab_size"]
        self.input_model_type = config["data"]["input_tokenizer_model"]
        self.output_model_type = config["data"]["output_tokenizer_model"]
        self.tokenizer_dir = config["data"]["tokenizer_model_path"]

    def _load_dataset(self):
        """
        Load the parallel dataset for the configured language pair.

        Returns:
            The dataset object containing train/validation/test splits.
        """

        dataset_name = f"{self.dataset_path}-{self.input_lang}-{self.output_lang}"
        dataset = load_dataset(
            self.dataset_path,
            dataset_name,
            trust_remote_code=True,
        )
        return dataset

    def _save_sentences(self, sentences, filename: str) -> None:
        """
        Save a list of sentences to a temporary text file.

        Args:
            sentences: List of strings to be saved.
            filename: Path to the output text file.
        """

        with open(filename, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")

    def build_tokenizers(self) -> None:
        """
        Train SentencePiece tokenizers for both input and output languages.

        Steps:
          1. Load the dataset and extract sentences.
          2. Save sentences to temporary files.
          3. Train SentencePiece models for input and output.
          4. Save models to the configured directory.
          5. Remove temporary files.
        """

        dataset = self._load_dataset()
        train_data = dataset["train"]

        # Extract raw sentences
        input_sentences = [ex["translation"][self.input_lang] for ex in train_data]
        output_sentences = [ex["translation"][self.output_lang] for ex in train_data]

        # Save to temp files
        input_file = "temp_input_sentences.txt"
        output_file = "temp_output_sentences.txt"
        self._save_sentences(input_sentences, input_file)
        self._save_sentences(output_sentences, output_file)

        # Ensure output directory exists
        os.makedirs(self.tokenizer_dir, exist_ok=True)

        # Train input tokenizer
        input_prefix = os.path.join(
            self.tokenizer_dir, f"{self.dataset_path}_{self.input_lang}"
        )
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=input_prefix,
            vocab_size=self.input_vocab_size,
            model_type=self.input_model_type,
        )

        # Train output tokenizer
        output_prefix = os.path.join(
            self.tokenizer_dir, f"{self.dataset_path}_{self.output_lang}"
        )
        spm.SentencePieceTrainer.train(
            input=output_file,
            model_prefix=output_prefix,
            vocab_size=self.output_vocab_size,
            model_type=self.output_model_type,
        )

        # Clean up temp files
        os.remove(input_file)
        os.remove(output_file)


def get_args() -> Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Namespace containing:
          - config_path: Path to the YAML config file.
    """

    parser = argparse.ArgumentParser(description="Tokenizer Script")
    parser.add_argument(
        "--config_path", default=None, help="Path to the YAML config file."
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    Entry point for building tokenizers.

    Steps:
      - Load the configuration YAML.
      - Initialize a TokenizerBuilder with the config.
      - Run the build_tokenizers method to train and save models.

    Args:
        args: Namespace with config_path attribute.
    """
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    builder = TokenizerBuilder(config)
    builder.build_tokenizers()


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except KeyboardInterrupt:
        raise
