"""
Architectural Notes:
--------------------
- The code implements a basic Vision Transformer (ViT) variant, with modular separation for embeddings, attention, MLP, encoder layers, and the overall transformer.
- No explicit support for masking, variable input sizes, or advanced attention mechanisms (e.g., relative position, windowed attention).
- TODO: Consider abstracting attention and MLP blocks for easier experimentation with alternative architectures.
- TODO: The model has square images and patch sizes; generalization to non-square or variable-sized images would require refactoring.
- TODO: No checkpointing or memory optimization; may be a concern for large-scale training.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """
    Configuration container for SiglipVision models.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of hidden representations.
    intermediate_size : int
        Dimensionality of intermediate MLP layer.
    num_hidden_layers : int
        Number of transformer encoder layers.
    num_attention_heads : int
        Number of attention heads.
    num_channels : int
        Number of input image channels.
    image_size : int
        Input image size (assumed square).
    patch_size : int
        Patch size (assumed square).
    layer_norm_eps : float
        Epsilon for layer normalization.
    attention_dropout : float
        Dropout probability for attention weights.
    num_image_tokens : Optional[int]
        Number of image tokens; computed if not provided.
    **kwargs
        Forwards unused config keys for compatibility.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        # If num_image_tokens is not provided, compute as (image_size // patch_size) ** 2
        self.num_image_tokens = (self.image_size // self.patch_size) ** 2 if num_image_tokens is None else num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """
    Image-to-patch embedding and positional encoding for vision transformer.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.

    Notes
    -----
    - Assumes input images are square and divisible by patch_size.
    - Position IDs are precomputed and registered as a buffer.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Converts images to patch embeddings and adds positional encodings.

        Parameters
        ----------
        pixel_values : torch.FloatTensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Patch embeddings with positional encodings, shape (batch_size, num_patches, embed_dim).

        Notes
        -----
        - Assumes input dimensions are compatible with patch size.
        """
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed self-attention as per 'Attention Is All You Need'.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.

    Notes
    -----
    - Assumes input hidden size is divisible by num_attention_heads.
    - No masking or causal attention.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes multi-head self-attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Output tensor of shape (batch_size, seq_len, embed_dim) and attention weights.
        """
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        # I'm keeping this check to catch shape mismatches early, especially if config or input is misaligned.
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # This check guards against silent shape errors after matmul, which can be subtle if head_dim or num_heads is wrong.
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    Feed-forward MLP block for transformer encoder.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies two-layer MLP with GELU activation.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    Single transformer encoder layer: self-attention + MLP + residuals.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.

    Notes
    -----
    - LayerNorm is applied before attention and MLP (pre-norm).
    - No dropout on residual connections.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies self-attention and MLP with pre-norm and residuals.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    Stacked transformer encoder layers.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.

    Notes
    -----
    - No intermediate outputs or hooks for layer-wise analysis.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies all encoder layers sequentially.

        Parameters
        ----------
        inputs_embeds : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    Vision Transformer backbone: embeddings, encoder, and final normalization.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Processes input images through embedding, encoder, and final normalization.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    Top-level vision model wrapper for SiglipVisionTransformer.

    Parameters
    ----------
    config : SiglipVisionConfig
        Model configuration.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Processes input images through the vision transformer.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        return self.vision_model(pixel_values=pixel_values)
