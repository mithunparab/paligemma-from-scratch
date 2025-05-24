"""
Architectural notes:
- LoRA integration is tightly coupled to nn.Linear; consider a more generic adapter pattern if future modules require LoRA.
- The SiglipVisionConfig and SiglipVisionModel are not robust; if their code changes, PaliGemmaConfig and PaliGemmaForConditionalGeneration will break.
- KVCache is stateful and not thread-safe; if used in multi-threaded or distributed settings, race conditions may arise.
- TODO: Consider separating model definition from training/inference utilities for clearer modularity.
"""

import torch
from torch import nn
from typing import Optional, Tuple, List, Dict, Any
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class LoraConfig:
    """
    Configuration for LoRA adapters.

    Parameters
    ----------
    r : int
        Rank of the LoRA decomposition.
    lora_alpha : int
        Scaling factor for LoRA.
    lora_dropout : float
        Dropout probability for LoRA branch.
    target_modules : List[str]
        Names of modules to apply LoRA to.
    """
    def __init__(self, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0,
                 target_modules: List[str] = None):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules if target_modules is not None else []

class LoraLinear(nn.Module):
    """
    LoRA-augmented linear layer.

    Parameters
    ----------
    linear_layer : nn.Linear
        The base linear layer to augment.
    config : LoraConfig
        LoRA configuration.

    Notes
    -----
    The base linear weights are frozen. Only LoRA parameters are trainable.
    """
    def __init__(self, linear_layer: nn.Linear, config: LoraConfig):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = config.r
        self.lora_alpha = config.lora_alpha
        
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        self.scaling = self.lora_alpha / self.rank

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after base linear and LoRA branch.
        """
        original_output = self.linear(x)
        lora_branch_output = self.lora_b(self.lora_a(self.lora_dropout(x)))
        return original_output + lora_branch_output * self.scaling

    @property
    def weight(self):
        return self.linear.weight

    @property
    def device(self):
        return self.linear.weight.device

    @property
    def dtype(self):
        return self.linear.weight.dtype

def _replace_module_with_lora(module: nn.Module, name: str, child_module: nn.Module, lora_config: LoraConfig):
    """
    Replace a child module with a LoRA-augmented version if it matches target_modules.

    Parameters
    ----------
    module : nn.Module
        Parent module.
    name : str
        Name of the child module.
    child_module : nn.Module
        The child module to potentially replace.
    lora_config : LoraConfig
        LoRA configuration.
    """
    if isinstance(child_module, nn.Linear) and name in lora_config.target_modules:
        lora_layer = LoraLinear(child_module, lora_config)
        setattr(module, name, lora_layer)

def add_lora_adapters(model: nn.Module, lora_config: LoraConfig):
    """
    Recursively add LoRA adapters to target modules in the model.

    Parameters
    ----------
    model : nn.Module
        Model to modify in-place.
    lora_config : LoraConfig
        LoRA configuration.
    """
    for layer_name, layer_module in model.named_modules():
        attr_name = layer_name.split(".")[-1]
        if isinstance(layer_module, nn.Linear) and attr_name in lora_config.target_modules:
            parent_name = ".".join(layer_name.split(".")[:-1])
            if not parent_name:
                parent_module = model
            else:
                parent_module = model.get_submodule(parent_name)
            if not isinstance(getattr(parent_module, attr_name), LoraLinear):
                lora_layer = LoraLinear(layer_module, lora_config)
                setattr(parent_module, attr_name, lora_layer)

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freeze all parameters except LoRA adapters (and optionally biases).

    Parameters
    ----------
    model : nn.Module
        Model to modify in-place.
    bias : str
        If 'lora_only', only LoRA biases are trainable. If 'all', all biases are trainable.
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'lora_only':
        for n, p in model.named_parameters():
            if 'lora_' in n and 'bias' in n:
                p.requires_grad = True
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True

def print_trainable_parameters(model: nn.Module):
    """
    Print the number and percentage of trainable parameters in the model.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class KVCache:
    """
    Key-value cache for transformer attention.

    Notes
    -----
    Not thread-safe. Used for incremental decoding/generation.
    """
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        """
        Returns the number of cached items per layer.

        Returns
        -------
        int
            Number of cached items.
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache for a given layer.

        Parameters
        ----------
        key_states : torch.Tensor
            New key states.
        value_states : torch.Tensor
            New value states.
        layer_idx : int
            Layer index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated key and value tensors for the layer.
        """
        device = key_states.device
        dtype = key_states.dtype
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states.to(device, dtype=dtype))
            self.value_cache.append(value_states.to(device, dtype=dtype))
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx].to(device, dtype=dtype), key_states.to(device, dtype=dtype)], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx].to(device, dtype=dtype), value_states.to(device, dtype=dtype)], dim=-2)
        return self.key_cache[layer_idx].to(device, dtype=dtype), self.value_cache[layer_idx].to(device, dtype=dtype)

class GemmaConfig:
    """
    Configuration for Gemma transformer model.

    Parameters
    ----------
    vocab_size : int
    hidden_size : int
    intermediate_size : int
    num_hidden_layers : int
    num_attention_heads : int
    num_key_value_heads : int
    head_dim : int
    max_position_embeddings : int
    rms_norm_eps : float
    rope_theta : float
    attention_bias : bool
    attention_dropout : float
    pad_token_id : int
    use_cache : bool
    gradient_checkpointing : bool
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing

class PaliGemmaConfig:
    """
    Configuration for PaliGemma multimodal model.

    Parameters
    ----------
    vision_config : dict or SiglipVisionConfig
    text_config : dict or GemmaConfig
    ignore_index : int
    image_token_index : int
    vocab_size : int
    projection_dim : int
    hidden_size : int
    pad_token_id : int
    use_cache : bool
    gradient_checkpointing : bool

    Notes
    -----
    vision_config and text_config can be dicts or config objects.
    """
    def __init__(
        self,
        vision_config: Optional[Any] = None,
        text_config: Optional[Any] = None,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int = 0,
        use_cache: bool = True, 
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size

        # I’m keeping this block: vision_config must be initialized before text_config, as text_config depends on vision_config for num_image_tokens.
        if not isinstance(vision_config, dict):
            vision_config_dict = vision_config.__dict__ if vision_config else {}
        else:
            vision_config_dict = vision_config
        self.vision_config = SiglipVisionConfig(**vision_config_dict)

        if not isinstance(text_config, dict):
            text_config_dict = text_config.__dict__ if text_config else {}
        else:
            text_config_dict = text_config
        text_config_dict.pop('gradient_checkpointing', None)
        text_config_dict.pop('use_cache', None)
        self.text_config = GemmaConfig(**text_config_dict, pad_token_id=pad_token_id, 
                                       gradient_checkpointing=gradient_checkpointing,
                                       use_cache=use_cache)
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = self.text_config.pad_token_id
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm for Gemma.

    Parameters
    ----------
    dim : int
        Feature dimension.
    eps : float
        Epsilon for numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary positional embedding for attention.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    max_position_embeddings : int
    base : float
    device : torch.device
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (for device/dtype).
        position_ids : torch.Tensor
            Position indices.
        seq_len : int, optional
            Sequence length.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Cosine and sine embeddings.
        """
        self.inv_freq = self.inv_freq.to(x.device)
        position_ids_on_device = position_ids.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids_on_device.shape[0], -1, 1)
        position_ids_expanded = position_ids_on_device[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype, device=x.device), sin.to(dtype=x.dtype, device=x.device)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate last dimension by half for rotary embedding.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key.

    Parameters
    ----------
    q : torch.Tensor
    k : torch.Tensor
    cos : torch.Tensor
    sin : torch.Tensor
    unsqueeze_dim : int

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaMLP(nn.Module):
    """
    MLP block for Gemma transformer.

    Parameters
    ----------
    config : GemmaConfig
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for multi-head attention.

    Parameters
    ----------
    hidden_states : torch.Tensor
    n_rep : int

    Returns
    -------
    torch.Tensor
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    """
    Multi-head self-attention for Gemma.

    Parameters
    ----------
    config : GemmaConfig
    layer_idx : int, optional
    """
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.use_cache = config.use_cache

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        _gradient_checkpointing_no_kv_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
        attention_mask : torch.Tensor, optional
        position_ids : torch.LongTensor, optional
        kv_cache : KVCache, optional
        _gradient_checkpointing_no_kv_cache : bool

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Output and attention weights.
        """
        target_device = self.q_proj.linear.weight.device
        target_dtype = self.q_proj.linear.weight.dtype

        hidden_states = hidden_states.to(target_device, dtype=target_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device, dtype=target_dtype)
        if position_ids is not None:
            position_ids = position_ids.to(target_device)

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # I’m keeping this: kv_cache is only updated if not in checkpointing mode, to avoid double-append.
        if kv_cache is not None and self.use_cache and not _gradient_checkpointing_no_kv_cache:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.dtype != attn_weights.dtype:
                 attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    """
    Single transformer decoder layer for Gemma.

    Parameters
    ----------
    config : GemmaConfig
    layer_idx : int
    """
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        _gradient_checkpointing: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for decoder layer.

        Parameters
        ----------
        hidden_states : torch.Tensor
        attention_mask : torch.Tensor, optional
        position_ids : torch.LongTensor, optional
        kv_cache : KVCache, optional
        _gradient_checkpointing : bool

        Returns
        -------
        torch.Tensor
        """
        def create_custom_forward(module):
            def custom_forward(*inputs):
                hidden_states_cp, attention_mask_cp, position_ids_cp = inputs[:3]
                target_device = module.self_attn.q_proj.linear.weight.device
                target_dtype = module.self_attn.q_proj.linear.weight.dtype

                residual = hidden_states_cp
                hidden_states_norm = module.input_layernorm(hidden_states_cp.to(target_device, dtype=target_dtype))

                attn_output, _ = module.self_attn(
                    hidden_states=hidden_states_norm,
                    attention_mask=attention_mask_cp.to(target_device, dtype=target_dtype) if attention_mask_cp is not None else attention_mask_cp,
                    position_ids=position_ids_cp.to(target_device) if position_ids_cp is not None else position_ids_cp,
                    kv_cache=None,
                    _gradient_checkpointing_no_kv_cache=True,
                )
                hidden_states_attn = residual + attn_output

                residual = hidden_states_attn
                hidden_states_norm_mlp = module.post_attention_layernorm(hidden_states_attn.to(target_device, dtype=target_dtype))
                mlp_output = module.mlp(hidden_states_norm_mlp)
                final_hidden_states = residual + mlp_output
                return final_hidden_states
            return custom_forward

        # I’m keeping this: checkpointing disables kv_cache to avoid state mutation during recomputation.
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self), 
                hidden_states, 
                attention_mask, 
                position_ids, 
                use_reentrant=False 
            )
            attn_weights = None
        else:
            target_device = self.self_attn.q_proj.linear.weight.device
            target_dtype = self.self_attn.q_proj.linear.weight.dtype

            residual = hidden_states
            hidden_states_norm = self.input_layernorm(hidden_states.to(target_device, dtype=target_dtype))

            attn_output, attn_weights = self.self_attn(
                hidden_states=hidden_states_norm,
                attention_mask=attention_mask.to(target_device, dtype=target_dtype) if attention_mask is not None else attention_mask,
                position_ids=position_ids.to(target_device) if position_ids is not None else position_ids,
                kv_cache=kv_cache if self.use_cache else None,
                _gradient_checkpointing_no_kv_cache=False,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states_norm = self.post_attention_layernorm(hidden_states.to(target_device, dtype=target_dtype))
            mlp_output = self.mlp(hidden_states_norm)
            hidden_states = residual + mlp_output

        return hidden_states

class GemmaModel(nn.Module):
    """
    Gemma transformer language model backbone.

    Parameters
    ----------
    config : GemmaConfig
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = config.gradient_checkpointing 
        self.use_cache = config.use_cache

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Returns the input embedding layer.

        Returns
        -------
        nn.Embedding
        """
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the language model.

        Parameters
        ----------
        inputs_embeds : torch.FloatTensor, optional
        attention_mask : torch.Tensor, optional
        position_ids : torch.LongTensor, optional
        kv_cache : KVCache, optional

        Returns
        -------
        torch.FloatTensor
        """
        target_device = self.embed_tokens.weight.device
        target_dtype = self.embed_tokens.weight.dtype
        hidden_states = inputs_embeds.to(target_device, dtype=target_dtype)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask.to(target_device, dtype=target_dtype) if attention_mask is not None else attention_mask,
                position_ids=position_ids.to(target_device) if position_ids is not None else position_ids,
                kv_cache=kv_cache,
                _gradient_checkpointing=self.gradient_checkpointing,
            )
        hidden_states = self.norm(hidden_states.to(target_device, dtype=target_dtype))
        return hidden_states

class GemmaForCausalLM(nn.Module):
    """
    Gemma model for causal language modeling.

    Parameters
    ----------
    config : GemmaConfig
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens
    
    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.model.embed_tokens = new_embeddings

    def tie_weights(self):
        """
        Tie lm_head weights to input embeddings if dimensions match.
        """
        if self.config.hidden_size == self.model.embed_tokens.embedding_dim:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for causal LM.

        Parameters
        ----------
        inputs_embeds : torch.FloatTensor, optional
        attention_mask : torch.Tensor, optional
        position_ids : torch.LongTensor, optional
        kv_cache : KVCache, optional
        labels : torch.LongTensor, optional

        Returns
        -------
        Dict[str, Any]
            Contains logits, loss (if labels), and kv_cache (if used).
        """
        target_device = self.lm_head.weight.device
        target_dtype = self.lm_head.weight.dtype

        hidden_states = self.model(
            inputs_embeds=inputs_embeds.to(target_device, dtype=target_dtype),
            attention_mask=attention_mask.to(target_device, dtype=target_dtype) if attention_mask is not None else attention_mask,
            position_ids=position_ids.to(target_device) if position_ids is not None else position_ids,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = loss.to(target_device)

        return_data = {"logits": logits}
        if loss is not None:
            return_data["loss"] = loss
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    """
    Projects vision features to language model space.

    Parameters
    ----------
    config : PaliGemmaConfig
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features.

        Parameters
        ----------
        image_features : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.linear(image_features)

class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Multimodal conditional generation model combining vision and language.

    Parameters
    ----------
    config : PaliGemmaConfig
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config) 
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _prepare_combined_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        kv_cache_len: int = 0
    ) -> torch.Tensor:
        """
        Construct combined causal and padding mask for attention.

        Parameters
        ----------
        attention_mask : torch.Tensor
            Boolean mask (True for tokens).
        input_shape : tuple
        dtype : torch.dtype
        device : torch.device
        kv_cache_len : int

        Returns
        -------
        torch.Tensor
        """
        batch_size, seq_len = input_shape
        kv_len = seq_len + kv_cache_len

        causal_mask_q_part = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
            diagonal=1
        )

        if kv_cache_len > 0:
            past_key_mask = torch.zeros((seq_len, kv_cache_len), device=device, dtype=dtype)
            full_causal_mask_for_q = torch.cat([past_key_mask, causal_mask_q_part], dim=1)
        else:
            full_causal_mask_for_q = causal_mask_q_part
        
        expanded_causal_mask = full_causal_mask_for_q.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, kv_len)

        padding_effect_mask = attention_mask.to(device, dtype=dtype)
        padding_effect = (1.0 - padding_effect_mask).unsqueeze(1).unsqueeze(2) * torch.finfo(dtype).min
        
        if kv_cache_len > 0:
             past_padding_effect = torch.zeros((batch_size, 1, 1, kv_cache_len), device=device, dtype=dtype)
             padding_effect = torch.cat([past_padding_effect, padding_effect], dim=-1)
        
        final_mask = expanded_causal_mask + padding_effect
        return final_mask

    def _prepare_position_ids(
        self,
        attention_mask: torch.Tensor,
        device: torch.device,
        kv_cache_len: int = 0
    ) -> torch.LongTensor:
        """
        Compute position ids, handling cache offset.

        Parameters
        ----------
        attention_mask : torch.Tensor
        device : torch.device
        kv_cache_len : int

        Returns
        -------
        torch.LongTensor
        """
        batch_size, seq_len = attention_mask.shape
        attention_mask = attention_mask.to(device)

        if kv_cache_len > 0:
            position_ids = torch.arange(kv_cache_len, kv_cache_len + seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == False), 1).to(device)

        return position_ids

    def _merge_input_ids_with_image_features(
        self,
        image_features: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Merge image features into input embeddings at image token positions.

        Parameters
        ----------
        image_features : torch.Tensor, optional
        inputs_embeds : torch.Tensor
        input_ids : torch.Tensor
        kv_cache : KVCache, optional

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]
            (merged embeddings, attention mask, position ids)
        """
        batch_size, sequence_length = input_ids.shape
        embed_dim = inputs_embeds.shape[-1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        final_embedding = inputs_embeds.clone() 

        if image_features is not None:
            image_mask = (input_ids == self.config.image_token_index).to(device)
            num_image_tokens_expected = self.config.text_config.num_image_tokens
            # I’m keeping this: strict check on image token count to catch misalignment between tokenizer and vision config.
            for i in range(batch_size):
                current_image_mask = image_mask[i]
                num_image_locations = current_image_mask.sum().item()
                if num_image_locations > 0:
                    if num_image_locations != num_image_tokens_expected:
                        raise ValueError(f"Batch {i}: Expected {num_image_tokens_expected} image tokens but found {num_image_locations}. "
                                         f"Check image_seq_length and model config.")
                    final_embedding[i, current_image_mask, :] = image_features[i].to(dtype=dtype, device=device)
        
        current_kv_cache_len = kv_cache.num_items() if kv_cache is not None else 0
        model_attention_mask = self._prepare_combined_attention_mask(
            attention_mask=(input_ids != self.config.pad_token_id).to(device),
            input_shape=input_ids.shape,
            dtype=dtype,
            device=device,
            kv_cache_len=current_kv_cache_len
        )
        
        position_ids = self._prepare_position_ids(
            attention_mask=(input_ids != self.config.pad_token_id).to(device),
            device=device,
            kv_cache_len=current_kv_cache_len
        )

        return final_embedding, model_attention_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for multimodal conditional generation.

        Parameters
        ----------
        input_ids : torch.LongTensor
        pixel_values : torch.FloatTensor, optional
        attention_mask : torch.Tensor, optional
        labels : torch.LongTensor, optional
        kv_cache : KVCache, optional

        Returns
        -------
        Dict[str, Any]
        """
        model_device = self.language_model.lm_head.weight.device
        model_dtype = self.language_model.lm_head.weight.dtype

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        processed_image_features = None
        if pixel_values is not None:
            image_vision_output = self.vision_tower(pixel_values)
            processed_image_features = self.multi_modal_projector(image_vision_output)
            
        merged_inputs_embeds, model_attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features=processed_image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            kv_cache=kv_cache
        )
        
        outputs = self.language_model(
            inputs_embeds=merged_inputs_embeds,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache if self.config.use_cache else None,
            labels=labels,
        )
        return outputs