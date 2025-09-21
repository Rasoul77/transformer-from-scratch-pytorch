# This file is covered by the LICENSE file in the root of this project.

import argparse
import glob
import os
import torch
import yaml
from argparse import Namespace

import sentencepiece as spm
from model import TransformerModel


class Translator:
    """
    Translator class for performing sequence-to-sequence translation
    using a Transformer model. It loads model configuration from a
    YAML file, initializes tokenizers and the Transformer, and provides
    a translate method for inference.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initializes the Translator with the configuration from a YAML file.

        Args:
            checkpoint_path: Path to the checkpoint folder.
        """

        # === Find the config file path ===
        found_yaml_files = [
            x for x in glob.glob(os.path.join(checkpoint_path, "config*.yaml"))
        ]
        assert (
            len(found_yaml_files) == 1
        ), "Not a unique config .yaml file in the checkpoint folder!"

        config_path = found_yaml_files[0]
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # === Find the pretrained model file path ===
        found_pre_trained_files = [
            x for x in glob.glob(os.path.join(checkpoint_path, "*.pt"))
        ]
        assert (
            len(found_pre_trained_files) == 1
        ), "Not a unique pretrained .pt file in the checkpoint folder!"
        pre_trained_model_path = found_pre_trained_files[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Data config ===
        data_cfg = self.config["data"]
        tokenizer_path = data_cfg["tokenizer_model_path"]
        self.pad_id = data_cfg["pad_token_index"]
        self.vocab_size = data_cfg["input_vocab_size"]
        self.input_lang = data_cfg["input_lang"]
        self.output_lang = data_cfg["output_lang"]

        # === Tokenizers ===
        self.tokenizer_in = spm.SentencePieceProcessor(
            model_file=f"{tokenizer_path}/iwslt2017_{self.input_lang}.model"
        )
        self.tokenizer_out = spm.SentencePieceProcessor(
            model_file=f"{tokenizer_path}/iwslt2017_{self.output_lang}.model"
        )
        self.bos_id = self.tokenizer_in.bos_id()
        self.eos_id = self.tokenizer_in.eos_id()

        # === Model config ===
        model_cfg = self.config["model"]
        self.max_len = model_cfg["max_seq_length"]

        self.model = TransformerModel(
            input_vocab_size=self.vocab_size,
            output_vocab_size=self.vocab_size,
            max_length=self.max_len,
            n_layers=model_cfg["num_layers"],
            n_heads=model_cfg["num_heads"],
            dim_model=model_cfg["dim_model"],
            dim_k=model_cfg["dim_k"],
            dim_v=model_cfg["dim_v"],
            dim_ff=model_cfg["dim_ff"],
            dropout=model_cfg["dropout"],
        )
        self.model.load_state_dict(
            torch.load(pre_trained_model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def make_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Creates a padding mask for a batch of sequences.

        Args:
            seq: Tensor of shape (B, L) representing token IDs.

        Returns:
            Boolean mask of shape (B, L, L) where True marks valid positions.
        """

        pad_mask = seq != self.pad_id
        return pad_mask.unsqueeze(1).expand(-1, seq.size(1), -1)

    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        """
        Translates a sentence from the input language to the output language.

        Args:
            sentence: Input sentence as a string.

        Returns:
            Translated sentence as a string.
        """

        # === Encode input sentence ===
        x_ids = self.tokenizer_in.encode(sentence)[: self.max_len - 2]
        x_tensor = torch.tensor(
            [[self.bos_id] + x_ids + [self.eos_id]], device=self.device
        )  # (1, S)
        src_mask = self.make_padding_mask(x_tensor)  # (1, S, S)

        # === Greedy decoding ===
        y_ids = [self.bos_id]
        for _ in range(self.max_len):
            y_tensor = torch.tensor([y_ids], device=self.device)  # (1, T)
            tgt_mask = self.make_padding_mask(y_tensor)
            causal_mask = torch.tril(
                torch.ones(
                    (1, len(y_ids), len(y_ids)), device=self.device, dtype=torch.bool
                )
            )
            combined_mask = tgt_mask & causal_mask
            src_key_mask = x_tensor != self.pad_id  # shape: (1, S)
            crs_mask = src_key_mask.unsqueeze(1).expand(
                -1, len(y_ids), -1
            )  # shape: (1, T, S)

            logits = self.model(
                x_tensor,
                y_tensor,
                src_mask=src_mask,
                tgt_mask=combined_mask,
                crs_mask=crs_mask,
            )  # (1, T, V)

            next_token_id = logits[0, -1].argmax().item()
            if next_token_id == self.eos_id:
                break
            y_ids.append(next_token_id)

        # === Decode output sentence ===
        return self.tokenizer_out.decode(y_ids[1:])


def get_args() -> Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Namespace containing:
          - checkpoint_path: Path to the checkpoint folder.
    """

    parser = argparse.ArgumentParser(
        description="Translator - Based on a trained Transformer Model"
    )
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to the checkpoint folder."
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    Entry point for interactive translation.
    Loads Translator from YAML config and allows user
    to input sentences for the translation.
    """

    translator = Translator(args.checkpoint_path)

    print(
        f"{translator.input_lang.upper()}-to-{translator.output_lang.upper()} Translation — <Enter> to quit."
    )
    while True:
        sentence = input(f"{translator.input_lang.upper()}: ")
        if sentence.strip().lower() == "":
            break
        translation = translator.translate(sentence)
        print(f"{translator.output_lang.upper()}:", translation)


if __name__ == "__main__":
    args = get_args()

    try:
        main(args)
    except KeyboardInterrupt:
        raise
