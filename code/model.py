# This file is covered by the LICENSE file in the root of this project.

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need". This module generates fixed
    position-dependent vectors that are added to word embeddings
    so that the model can take into account the order of tokens.
    A dropout is applied to the sum of token embeddings and
    positional encodings.
    """

    def __init__(self, dim_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            dim_model: Dimension of the embeddings.
            max_len: Maximum sequence length supported.
            dropout: Dropout probability applied after adding positional encodings.
        """

        super().__init__()

        # Create a long enough P matrix
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        # An efficient compution of 10_000^(2 * i / dim_model)
        # Note that torch.arrange(0, dim_model, 2) create {j: j = 2*i for i from 0 to dim_model//2}
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )  # (dim_model // 2)

        pe = torch.zeros(max_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, dim_model)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Args:
            x: Tensor of shape (batch_size, seq_len, dim_model) representing input embeddings.

        Returns:
            Tensor of the same shape with positional encodings added and dropout applied.
        """

        return self.dropout(x + self.pe[:, : x.size(1), :])


class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization, which normalizes
    inputs across the feature dimension. This stabilizes training
    and helps gradients flow more smoothly in deep networks.
    """

    def __init__(self, feature_dim: int, eps: float = 1e-5):
        """
        Args:
            feature_dim: Dimension of features to normalize.
            eps: Small constant added to variance for numerical stability.
        """

        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))  # scale
        self.beta = nn.Parameter(torch.zeros(feature_dim))  # shift
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Normalized tensor with the same shape.
        """

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class MaskedScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention with optional masking.
    This is the core operation of the Transformer: it computes
    attention weights between queries and keys, applies a mask if
    needed (e.g. for causal masking in the decoder), and uses
    them to weight the values.
    """

    def __init__(self, dropout: float = 0.1):
        """
        Args:
            dropout: Dropout probability applied to attention weights.
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for scaled dot-product attention.

        Args:
            q: Query tensor of shape (B, N, dim_k).
            k: Key tensor of shape (B, N, dim_k).
            v: Value tensor of shape (B, N, dim_v).
            mask: Optional boolean mask of shape (1, N, N) used to prevent
                  attention to certain positions (e.g. padding or future tokens).

        Returns:
            Tensor of shape (B, N, dim_v) representing attended values.
        """

        # MatMul
        x = torch.matmul(q, k.transpose(-2, -1))  # x: [B, N, N]
        # Scale
        x = x / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))
        # Apply causal mask
        if mask is not None:
            x = x.masked_fill(mask == 0, float("-inf"))
        # SoftMax
        x = torch.softmax(x, dim=-1)  # x: [B, N, N]
        x = self.dropout(x)
        # MatMul
        x = torch.matmul(x, v)  # x: [B, N, dim_v]
        # Result
        return x  # x: [B, N, dim_v]


class MaskedMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism. Splits the input into multiple
    attention heads, applies scaled dot-product attention in parallel,
    and concatenates the results. This allows the model to jointly
    attend to information from different representation subspaces.
    """

    def __init__(
        self,
        n_heads: int,
        dim_model: int,
        dim_k: int,
        dim_v: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_heads: Number of attention heads.
            dim_model: Dimension of the input embeddings.
            dim_k: Dimension of the key/query vectors.
            dim_v: Dimension of the value vectors.
            dropout: Dropout probability applied after attention and projection.
        """

        super().__init__()
        self.n_heads = n_heads

        self.multi_head_linear_projection = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q": nn.Linear(dim_model, dim_k),
                        "k": nn.Linear(dim_model, dim_k),
                        "v": nn.Linear(dim_model, dim_v),
                    }
                )
                for _ in range(n_heads)
            ]
        )

        self.masked_multi_head_scaled_dot_product_attention = nn.ModuleList(
            [MaskedScaledDotProductAttention(dropout=dropout) for _ in range(n_heads)]
        )

        self.output_linear = nn.Linear(n_heads * dim_v, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            q: Query tensor of shape (B, N, dim_model).
            k: Key tensor of shape (B, N, dim_model).
            v: Value tensor of shape (B, N, dim_model).
            mask: Optional mask tensor of shape (B, N, dim_model).

        Returns:
            Tensor of shape (B, N, dim_model) representing the attention output.
        """

        x = self.output_linear(
            torch.cat(
                [
                    self.masked_multi_head_scaled_dot_product_attention[head_idx](
                        q=self.multi_head_linear_projection[head_idx]["q"](q),
                        k=self.multi_head_linear_projection[head_idx]["k"](k),
                        v=self.multi_head_linear_projection[head_idx]["v"](v),
                        mask=mask,
                    )
                    for head_idx in range(self.n_heads)
                ],
                dim=-1,
            )
        )
        return self.dropout(x)


class EncoderSingleLayer(nn.Module):
    """
    A single layer of the Transformer encoder. Consists of a
    multi-head self-attention mechanism followed by a feed-forward
    neural network, with residual connections, dropout, and
    layer normalization applied around each sub-layer.
    """

    def __init__(
        self,
        n_heads: int,
        dim_model: int,
        dim_k: int,
        dim_v: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_heads: Number of attention heads.
            dim_model: Dimension of the input embeddings.
            dim_k: Dimension of key/query vectors.
            dim_v: Dimension of value vectors.
            dim_ff: Dimension of the hidden layer in feed-forward network.
            dropout: Dropout probability.
        """

        super().__init__()

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            n_heads=n_heads,
            dim_model=dim_model,
            dim_k=dim_k,
            dim_v=dim_v,
            dropout=dropout,
        )
        self.layer_norm_1 = LayerNorm(feature_dim=dim_model)
        self.layer_norm_2 = LayerNorm(feature_dim=dim_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for one encoder layer.

        Args:
            x: Tensor of shape (B, S, D) — encoder input.
            mask: Optional source mask of shape (B, S, S).

        Returns:
            Encoded tensor of shape (B, S, D).
        """

        x = self.layer_norm_1(
            x + self.dropout(self.masked_multi_head_attention(q=x, k=x, v=x, mask=mask))
        )
        x = self.layer_norm_2(x + self.dropout(self.feed_forward(x)))
        return x


class EncoderMultiLayer(nn.Module):
    """
    Stacks multiple encoder layers sequentially. Each layer
    refines the sequence representation, allowing the encoder
    to capture increasingly complex relationships between tokens.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        dim_model: int,
        dim_k: int,
        dim_v: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_layers: Number of encoder layers to stack.
            n_heads: Number of attention heads in each layer.
            dim_model: Dimension of input embeddings.
            dim_k: Dimension of key/query vectors.
            dim_v: Dimension of value vectors.
            dim_ff: Dimension of feed-forward hidden layer.
            dropout: Dropout probability.
        """

        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderSingleLayer(
                    n_heads=n_heads,
                    dim_model=dim_model,
                    dim_k=dim_k,
                    dim_v=dim_v,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the full encoder stack.

        Args:
            x: Input tensor of shape (B, S, D).
            mask: Optional source mask of shape (B, S, S).

        Returns:
            Encoded sequence of shape (B, S, D).
        """

        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return x


class DecoderSingleLayer(nn.Module):
    """
    A single layer of the Transformer decoder. It includes a
    masked self-attention mechanism, an encoder-decoder cross-attention,
    and a feed-forward network, each wrapped with residual connections,
    dropout, and layer normalization.
    """

    def __init__(
        self,
        n_heads: int,
        dim_model: int,
        dim_k: int,
        dim_v: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_heads: Number of attention heads.
            dim_model: Dimension of the input embeddings.
            dim_k: Dimension of key/query vectors.
            dim_v: Dimension of value vectors.
            dim_ff: Dimension of feed-forward hidden layer.
            dropout: Dropout probability.
        """

        super().__init__()

        self.self_attention = MaskedMultiHeadAttention(
            n_heads=n_heads,
            dim_model=dim_model,
            dim_k=dim_k,
            dim_v=dim_v,
            dropout=dropout,
        )

        self.cross_attention = MaskedMultiHeadAttention(
            n_heads=n_heads,
            dim_model=dim_model,
            dim_k=dim_k,
            dim_v=dim_v,
            dropout=dropout,
        )

        self.layer_norm_1 = LayerNorm(feature_dim=dim_model)
        self.layer_norm_2 = LayerNorm(feature_dim=dim_model)
        self.layer_norm_3 = LayerNorm(feature_dim=dim_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,  # (B, T, T)
        cross_mask: Optional[torch.Tensor] = None,  # (B, T, S)
    ):
        """
        Forward pass for one decoder layer.

        Args:
            x: Target input tensor of shape (B, T, D).
            context: Encoder output tensor of shape (B, S, D).
            self_mask: Optional mask for target tokens (causal + padding).
            cross_mask: Optional mask for source tokens (padding).

        Returns:
            Decoded tensor of shape (B, T, D).
        """

        x = self.layer_norm_1(
            x + self.dropout(self.self_attention(q=x, k=x, v=x, mask=self_mask))
        )

        x = self.layer_norm_2(
            x
            + self.dropout(
                self.cross_attention(q=x, k=context, v=context, mask=cross_mask)
            )
        )

        x = self.layer_norm_3(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderMultiLayer(nn.Module):
    """
    Stacks multiple decoder layers sequentially. Each layer refines
    the target sequence representation by combining masked self-attention,
    encoder-decoder attention, and feed-forward sublayers.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        dim_model: int,
        dim_k: int,
        dim_v: int,
        dim_ff: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_layers: Number of decoder layers to stack.
            n_heads: Number of attention heads per layer.
            dim_model: Dimension of embeddings.
            dim_k: Dimension of key/query vectors.
            dim_v: Dimension of value vectors.
            dim_ff: Dimension of feed-forward hidden layer.
            dropout: Dropout probability.
        """

        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderSingleLayer(
                    n_heads=n_heads,
                    dim_model=dim_model,
                    dim_k=dim_k,
                    dim_v=dim_v,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        y: torch.Tensor,  # The embedings of target tokens
        context: torch.Tensor,  # From encoder output
        self_mask: Optional[torch.Tensor] = None,  # For self-attention padding
        cross_mask: Optional[torch.Tensor] = None,  # For encoder padding
    ):
        """
        Forward pass through the full decoder stack.

        Args:
            y: Target input tensor of shape (B, T, D).
            context: Encoder output tensor of shape (B, S, D).
            self_mask: Optional mask for target sequence.
            cross_mask: Optional mask for source sequence.

        Returns:
            Decoded sequence of shape (B, T, D).
        """

        for layer in self.layers:
            y = layer(
                y,
                context,
                self_mask=self_mask,
                cross_mask=cross_mask,
            )
        return y


class TransformerModel(nn.Module):
    """
    The complete Transformer architecture combining an encoder
    and a decoder. It maps a sequence of source tokens to a sequence
    of target tokens by first encoding the source into hidden
    representations and then decoding them autoregressively.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        max_length: int,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_model: int = 512,
        dim_k: int = 64,
        dim_v: int = 64,
        dim_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_vocab_size: Size of the source vocabulary.
            output_vocab_size: Size of the target vocabulary.
            max_length: Maximum sequence length supported.
            n_layers: Number of encoder and decoder layers.
            n_heads: Number of attention heads per layer.
            dim_model: Dimension of embeddings and model hidden states.
            dim_k: Dimension of key/query vectors.
            dim_v: Dimension of value vectors.
            dim_ff: Dimension of feed-forward hidden layer.
            dropout: Dropout probability.
        """

        super().__init__()

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim=dim_model)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_dim=dim_model)
        self.positional_encoding = PositionalEncoding(
            max_len=max_length, dim_model=dim_model, dropout=dropout
        )

        self.multi_layer_encoder = EncoderMultiLayer(
            n_layers=n_layers,
            n_heads=n_heads,
            dim_model=dim_model,
            dim_k=dim_k,
            dim_v=dim_v,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        self.multi_layer_decoder = DecoderMultiLayer(
            n_layers=n_layers,
            n_heads=n_heads,
            dim_model=dim_model,
            dim_k=dim_k,
            dim_v=dim_v,
            dim_ff=dim_ff,
            dropout=dropout,
        )

        self.linear = nn.Linear(dim_model, output_vocab_size)

    def forward(
        self,
        x: torch.LongTensor,  # (B, S) — source token IDs
        y: torch.LongTensor,  # (B, T) — target input token IDs
        src_mask: Optional[
            torch.Tensor
        ] = None,  # (B, S, S) - padding mask for source sequence batch
        tgt_mask: Optional[
            torch.Tensor
        ] = None,  # (B, T, T) - padding mask for target sequence batch
        crs_mask: Optional[torch.Tensor] = None,  # (B, T, S) - cross attention mask
    ) -> torch.Tensor:
        """
        Forward pass for the full Transformer model.

        Args:
            x: Source tokens of shape (batch_size, seq_len).
            y: Target tokens of shape (batch_size, seq_len).
            src_mask: Optional mask for source (padding).
            tgt_mask: Optional mask for target (causal + padding).
            crs_mask: Optional mask for encoder-decoder attention.
        Returns:
            Logits of shape (B, T, output_vocab_size).
        """

        # === Embeddings ===
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        y = self.output_embedding(y)
        y = self.positional_encoding(y)

        # === Encoder ===
        x = self.multi_layer_encoder(x, mask=src_mask)  # (B, S, D)

        # === Decoder ===
        y = self.multi_layer_decoder(
            y,  # decoder input embeddings
            x,  # encoder output
            self_mask=tgt_mask,
            cross_mask=crs_mask,
        )

        # === Output projection ===
        return self.linear(y)  # (B, T, vocab_size)
