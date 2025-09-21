# This file is covered by the LICENSE file in the root of this project.

import os
import datetime
import time
import shutil
import wandb

from collections import OrderedDict
from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import sentencepiece as spm

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model import TransformerModel
from utility.utils import *


class TransformerTrainer:
    """
    TransformerTrainer manages the full training and evaluation pipeline
    for a Transformer-based sequence-to-sequence model.

    This class handles:
      - Loading and storing configuration parameters.
      - Initializing the Transformer model, optimizer, scheduler, and loss function.
      - Loading tokenizers and datasets for training and evaluation.
      - Managing reproducibility (random seeds, deterministic execution).
      - Building the required attention masks for encoder, decoder, and cross-attention.
      - Training the model for one epoch and evaluating on validation data.
      - Logging training progress (optionally with Weights & Biases).
      - Saving checkpoints and keeping track of the best model according to
        validation loss.

    The overall design mirrors the workflow described in "Attention Is All You Need":
      1. Tokenize and encode input/output sentence pairs.
      2. Build batches with padding and masks.
      3. Run forward and backward passes to optimize the model.
      4. Evaluate periodically to measure generalization.
      5. Save the best-performing model checkpoint.

    Typical usage:
        trainer = TransformerTrainer(config)
        trainer.run()

    Args:
        config (dict): A dictionary containing nested training, model, data,
                       and evaluation parameters (e.g. batch size, learning rate,
                       model dimensions, dropout rate, checkpoint paths).
    """

    def __init__(self, config: dict):
        """
        Initialize all components needed for training.

        This includes:
          - Setting up checkpoint directories and saving the config file.
          - Initializing Weights & Biases for logging if enabled.
          - Fixing random seeds for reproducibility.
          - Building the Transformer model (and loading weights if specified).
          - Loading datasets and wrapping them in DataLoaders.
          - Preparing tokenizers for input and output languages.
          - Setting up optimizer, loss function, and optional learning rate scheduler.

        Args:
            config (dict): Configuration dictionary for model, training, and data.
        """

        self.config = DotDict(config)

        # Checkpoint Directory
        self.checkpoint_path = os.path.join(
            self.config.train.root_chkpt,
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.checkpoint_path)

        # Copy configuration to the checkpoint directory
        shutil.copy2(self.config.config_path, self.checkpoint_path)

        # Logging
        if self.config.wandb.enabled:
            wandb.login()
            self.wandb_exp = wandb.init(
                project=f"Attention Is All You Need",
                name=self.config.wandb.project_name,
                resume="allow",
            )
            wandb_url = wandb.run.get_url()
            with open(os.path.join(self.checkpoint_path, "wandb_url.txt"), "w") as f:
                f.write(wandb_url)
                f.write("\n")
                f.write(self.config.wandb.project_name)

        # Print config
        pprint(config)

        # Seed and Reproducibility
        random_seed = get_seed(config)
        print(f"Random seed: {random_seed}")
        set_random_seed(random_seed, deterministic=self.config.train.deterministic)
        self.np_rand_gen = np.random.default_rng()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TransformerModel
        self.model = TransformerModel(
            input_vocab_size=self.config.data.input_vocab_size,
            output_vocab_size=self.config.data.output_vocab_size,
            max_length=self.config.model.max_seq_length,
            n_layers=self.config.model.num_layers,
            n_heads=self.config.model.num_heads,
            dim_model=self.config.model.dim_model,
            dim_k=self.config.model.dim_k,
            dim_v=self.config.model.dim_v,
            dim_ff=self.config.model.dim_ff,
            dropout=self.config.model.dropout,
        ).to(self.device)

        if self.config.model.load:
            self.model.load_state_dict(torch.load(self.config.model.load))

        # Load Datasets
        dataset_path = config["data"]["dataset_path"]
        input_lang = config["data"]["input_lang"]
        output_lang = config["data"]["output_lang"]
        dataset_name = "{}-{}-{}".format(dataset_path, input_lang, output_lang)
        dataset = load_dataset(dataset_path, dataset_name, trust_remote_code=True)

        self.dataloader_train = DataLoader(
            dataset["train"],
            batch_size=self.config.train.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        self.dataloader_eval = DataLoader(
            dataset["validation"],
            batch_size=self.config.eval.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        # Load tokenizers
        input_model_prefix = os.path.join(
            self.config.data.tokenizer_model_path,
            "{}_{}".format(dataset_path, input_lang),
        )

        output_model_prefix = os.path.join(
            self.config["data"]["tokenizer_model_path"],
            "{}_{}".format(dataset_path, output_lang),
        )

        self.tokenizer_input = spm.SentencePieceProcessor(
            model_file=f"{input_model_prefix}.model"
        )
        self.tokenizer_output = spm.SentencePieceProcessor(
            model_file=f"{output_model_prefix}.model"
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.train.learning_rate,
        )

        # Loss Function
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.config.data.pad_token_index
        )

        # Learning Rate Scheduler
        self.scheduler = None
        if self.config.train.scheduler.enabled:
            num_steps_per_epoch = len(self.dataloader_train)
            warmup_steps = int(
                self.config.train.scheduler.num_warmup_epochs * num_steps_per_epoch
            )
            num_total_steps = self.config.train.num_epochs * num_steps_per_epoch
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_total_steps,
                num_cycles=self.config.train.scheduler.num_cycles,
            )

        self.best_loss = float("inf")
        self.epoch_idx = 1

    def _encode_pair(
        self, inp: str, out: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single pair of input and output sentences into tensors.

        Steps performed:
          - Tokenize input and output strings with SentencePiece.
          - Truncate to the maximum length allowed by the model (minus BOS/EOS).
          - Add special BOS (beginning of sentence) and EOS (end of sentence) tokens.
          - Prepare shifted output tensors for teacher forcing during training.

        Args:
            inp: Input string (source language).
            out: Output string (target language).

        Returns:
            A tuple of three tensors:
              - Encoded input tensor with BOS/EOS.
              - Encoded output tensor shifted right (used as decoder input).
              - Encoded output tensor shifted left (used as training target).
        """

        # Reserve room for BOS and EOS
        max_tokens = self.config.model.max_seq_length - 2

        # Truncate both sides
        inp_ids = self.tokenizer_input.encode(inp)[:max_tokens]
        out_ids = self.tokenizer_output.encode(out)[:max_tokens]

        # Add BOS and EOS
        inp_tensor = torch.tensor([1] + inp_ids + [2], dtype=torch.long)
        out_tensor = torch.tensor([1] + out_ids + [2], dtype=torch.long)

        return inp_tensor, out_tensor[:-1], out_tensor[1:]

    def collate_fn(
        self, batch: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom collation function to prepare a batch of examples.

        For each sample in the batch:
          - Encode input and output sentences into tensors.
          - Collect and pad them so that all sequences in the batch
            have the same length.

        Args:
            batch: A list of dataset samples, each containing a translation pair.

        Returns:
            A tuple of padded tensors:
              - Input batch (B, S).
              - Decoder input batch (B, T).
              - Decoder target batch (B, T).
        """

        x_list, y_in_list, y_out_list = [], [], []

        for sample in batch:
            x_tensor, y_in_tensor, y_out_tensor = self._encode_pair(
                sample["translation"][self.config.data.input_lang],
                sample["translation"][self.config.data.output_lang],
            )
            x_list.append(x_tensor)
            y_in_list.append(y_in_tensor)
            y_out_list.append(y_out_tensor)

        x_padded = torch.nn.utils.rnn.pad_sequence(
            x_list, batch_first=True, padding_value=0
        )
        y_in_padded = torch.nn.utils.rnn.pad_sequence(
            y_in_list, batch_first=True, padding_value=0
        )
        y_out_padded = torch.nn.utils.rnn.pad_sequence(
            y_out_list, batch_first=True, padding_value=0
        )

        return x_padded, y_in_padded, y_out_padded

    def train_one_epoch(self):
        """
        Train the model for one full epoch.

        Key steps:
          - Set the model to training mode.
          - Loop through the training DataLoader.
          - Construct encoder, decoder, and cross-attention masks
            to handle padding and causal dependencies.
          - Perform forward pass, compute loss, and backpropagate gradients.
          - Update model parameters with the optimizer (and scheduler if enabled).
          - Track and log running loss and learning rate.

        Returns:
            The average training loss for the epoch.
        """

        torch.cuda.empty_cache()
        self.model.train()

        epoch_loss = 0.0
        tqdm_desc = f"Train Epoch {self.epoch_idx:02}/{self.config.train.num_epochs}"
        with tqdm(self.dataloader_train, leave=False, desc=tqdm_desc) as train_bar:
            for batch_idx, (x, y_in, y_out) in enumerate(train_bar):
                x = x.to(self.device)  # (B, S)
                y_in = y_in.to(self.device)  # (B, T)
                y_out = y_out.to(self.device)

                B, T = y_in.shape

                self.optimizer.zero_grad()

                # ——————— Build encoder padding mask of shape (B, S, S) ————————
                # Initially, key‐mask is (B, S): 1 if token != 0, 0 if token == <pad>
                src_key_mask = x != 0  # (B, S)
                # Expand into (B, S, S) so that for each query position i, key position j is masked if j is pad.
                src_padding_mask = src_key_mask.unsqueeze(1).expand(
                    -1, x.size(1), -1
                )  # (B, S, S)

                # ——————— Build decoder padding mask of shape (B, T, T) ————————
                tgt_key_mask = y_in != 0  # (B, T)
                tgt_padding_mask = tgt_key_mask.unsqueeze(1).expand(
                    -1, y_in.size(1), -1
                )  # (B, T, T)

                # ——————— Build decoder casual mask of shape (B, T, T) ————————
                target_causal_mask = (
                    torch.tril(torch.ones((T, T), dtype=torch.bool, device=self.device))
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )  # (B, T, T)

                target_mask = target_causal_mask & tgt_padding_mask  # (B, T, T)

                # ——————— Build decoder cross attention mask of shape (B, T, S) ————————
                cross_attention_mask = src_key_mask.unsqueeze(1).expand(
                    -1, T, -1
                )  # (B, T, S)

                # ——————— Forward pass with both masks ————————
                logits = self.model(
                    x,
                    y_in,
                    src_mask=src_padding_mask,
                    tgt_mask=target_mask,
                    crs_mask=cross_attention_mask,
                )  # logits: (B, T, V)

                # ——————— Calculate loss ————————
                batch_loss = self.loss_function(
                    logits.view(-1, logits.size(-1)), y_out.view(-1)
                )

                epoch_loss += batch_loss.item()

                batch_loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                running_loss = epoch_loss / (batch_idx + 1)
                running_lr = self.optimizer.param_groups[0]["lr"]

                train_bar.set_postfix(
                    OrderedDict(
                        running_loss=f"{running_loss:.4f}", lr=f"{running_lr:.3e}"
                    )
                )

                if self.config.wandb.enabled:
                    self.wandb_exp.log(
                        {
                            "batch loss": batch_loss.item(),
                            "running loss": running_loss,
                            "epoch": self.epoch_idx,
                            "learning rate": running_lr,
                        }
                    )

        return running_loss

    def eval_one_epoch(self):
        """
        Evaluate the model on the validation set for one epoch.

        Key steps:
          - Set the model to evaluation mode.
          - Loop through the evaluation DataLoader without gradient computation.
          - Construct attention masks in the same way as during training.
          - Perform forward passes and compute loss.
          - Track the average evaluation loss across the entire validation set.

        Returns:
            The average evaluation loss for the epoch.
        """

        torch.cuda.empty_cache()
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            tqdm_desc = f"Eval Epoch {self.epoch_idx:02}/{self.config.train.num_epochs}"
            with tqdm(self.dataloader_eval, leave=False, desc=tqdm_desc) as eval_bar:
                for batch_idx, (x, y_in, y_out) in enumerate(eval_bar):
                    x = x.to(self.device)
                    y_in = y_in.to(self.device)
                    y_out = y_out.to(self.device)

                    B, T = y_in.shape

                    # ——————— Build encoder padding mask of shape (B, S, S) ————————
                    src_key_mask = x != 0  # (B, S)
                    src_padding_mask = src_key_mask.unsqueeze(1).expand(
                        -1, x.size(1), -1
                    )  # (B, S, S)

                    # ——————— Build decoder padding mask of shape (B, T, T) ————————
                    tgt_key_mask = y_in != 0  # (B, T)
                    tgt_padding_mask = tgt_key_mask.unsqueeze(1).expand(
                        -1, T, -1
                    )  # (B, T, T)

                    # ——————— Build decoder causal mask of shape (B, T, T) ————————
                    target_causal_mask = (
                        torch.tril(
                            torch.ones((T, T), dtype=torch.bool, device=self.device)
                        )
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )  # (B, T, T)

                    target_mask = target_causal_mask & tgt_padding_mask  # (B, T, T)

                    # ——————— Build decoder cross-attention mask of shape (B, T, S) ————————
                    cross_attention_mask = src_key_mask.unsqueeze(1).expand(
                        -1, T, -1
                    )  # (B, T, S)

                    # ——————— Forward pass with all masks ————————
                    logits = self.model(
                        x,
                        y_in,
                        src_mask=src_padding_mask,
                        tgt_mask=target_mask,
                        crs_mask=cross_attention_mask,
                    )

                    # ——————— Loss ————————
                    batch_loss = self.loss_function(
                        logits.view(-1, logits.size(-1)), y_out.view(-1)
                    )

                    total_loss += batch_loss.item() * y_out.numel()
                    total_tokens += y_out.numel()

                    eval_bar.set_postfix(
                        OrderedDict(
                            avg_loss=f"{total_loss / total_tokens:.4f}",
                        )
                    )

        avg_loss = total_loss / total_tokens

        return avg_loss

    def run(self) -> None:
        """
        Execute the full training loop across multiple epochs.

        For each epoch:
          - Call `train_one_epoch` to train the model.
          - Call `eval_one_epoch` to validate the model.
          - Log results (to console and optionally Weights & Biases).
          - Save the model if it achieves the best validation loss so far.

        This method acts as the high-level entry point for training.
        """

        for self.epoch_idx in range(1, self.config.train.num_epochs + 1):

            start_time = time.time()
            train_epoch_loss = self.train_one_epoch()
            train_epoch_time = get_formatted_time_execution(start_time)

            start_time = time.time()
            eval_epoch_loss = self.eval_one_epoch()
            eval_epoch_time = get_formatted_time_execution(start_time)

            if self.config.wandb.enabled:
                self.wandb_exp.log(
                    {
                        "eval loss": eval_epoch_loss,
                    }
                )

            print(
                f"Epoch{self.epoch_idx:02}/{self.config.train.num_epochs}"
                f" | Train > Loss: {train_epoch_loss:.4f} Time: {train_epoch_time}"
                f" | Eval > Loss: {eval_epoch_loss:.4f} Time: {eval_epoch_time}"
            )

            self.save_best_model(eval_loss=eval_epoch_loss)

    def save_best_model(self, eval_loss: float) -> None:
        """
        Save the model checkpoint if it achieves a new best validation loss.

        Steps performed:
          - Compare the current evaluation loss with the best so far.
          - If improved, save the model state and update the stored best loss.

        Args:
            eval_loss (float): The validation loss from the current epoch.
        """

        best_eval_loss = read_number_from_file(
            "eval_loss", self.checkpoint_path, default=np.inf
        )

        to_print = f"Epoch{self.epoch_idx:02}/{self.config.train.num_epochs} Best Eval"
        if eval_loss < best_eval_loss:
            to_print += f" | Loss : {best_eval_loss:.4f} -> {eval_loss:.4f}"
            save_model(self.model, "best_val_loss_model.pt", self.checkpoint_path)
            save_number_to_file("eval_loss", self.checkpoint_path, eval_loss)
            print(to_print)
