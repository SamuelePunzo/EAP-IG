from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


_EAP_PREPARED_FLAG = "_eap_prepared_for_compatibility"
_EAP_COMPAT_FLAG = "_eap_enabled_bridge_compatibility"
_MISSING = object()


@runtime_checkable
class EAPModelProtocol(Protocol):
    cfg: Any
    tokenizer: Any

    def to_tokens(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def hooks(self, *args: Any, **kwargs: Any) -> Any: ...

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class EAPModelAdapter:
    """A thin delegating wrapper over HookedTransformer-like models."""

    def __init__(self, model: EAPModelProtocol):
        self.model = model

    @property
    def cfg(self) -> Any:
        return self.model.cfg

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def to_tokens(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model.to_tokens(*args, **kwargs)

    def hooks(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.hooks(*args, **kwargs)

    def run_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.run_with_hooks(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.zero_grad(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def cfg_get(cfg: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(cfg, dict):
        if name in cfg:
            return cfg[name]
    elif hasattr(cfg, name):
        return getattr(cfg, name)
    else:
        try:
            return cfg[name]
        except (KeyError, TypeError):
            pass

    if default is _MISSING:
        raise AttributeError(f"Config is missing required field {name!r}")
    return default


def cfg_set(cfg: Any, name: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def is_bridge_like(model: Any) -> bool:
    return (
        hasattr(model, "enable_compatibility_mode")
        and hasattr(model, "run_with_hooks")
        and hasattr(model, "hook_aliases")
    )


def _bridge_compatibility_enabled(model: Any) -> bool:
    return bool(
        getattr(model, _EAP_COMPAT_FLAG, False)
        or getattr(model, "compatibility_mode_enabled", False)
        or getattr(model, "_compatibility_mode_enabled", False)
    )


def _call_if_present(model: Any, method_name: str, *args: Any, **kwargs: Any) -> bool:
    method = getattr(model, method_name, None)
    if method is None:
        return False
    method(*args, **kwargs)
    return True


def _configure_bridge_hooks(model: Any) -> None:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and cfg_get(cfg, "use_attn_in", False):
        _call_if_present(model, "set_use_attn_in", False)
        if cfg_get(cfg, "use_attn_in", False):
            cfg_set(cfg, "use_attn_in", False)

    if not _call_if_present(model, "set_use_attn_result", True) and cfg is not None:
        cfg_set(cfg, "use_attn_result", True)

    if not _call_if_present(model, "set_use_split_qkv_input", True) and cfg is not None:
        cfg_set(cfg, "use_split_qkv_input", True)

    if not _call_if_present(model, "set_use_hook_mlp_in", True) and cfg is not None:
        cfg_set(cfg, "use_hook_mlp_in", True)


def prepare_model_for_eap(
    model: EAPModelProtocol | EAPModelAdapter,
    *,
    auto_enable_bridge_compat: bool = True,
    compatibility_mode_kwargs: Optional[dict[str, Any]] = None,
) -> EAPModelAdapter:
    """Prepare a HookedTransformer-like model for EAP and return a delegating adapter."""

    if isinstance(model, EAPModelAdapter):
        original_model = model.model
        adapter = model
    else:
        original_model = model
        adapter = EAPModelAdapter(model)

    if is_bridge_like(original_model):
        if auto_enable_bridge_compat and not _bridge_compatibility_enabled(original_model):
            original_model.enable_compatibility_mode(**(compatibility_mode_kwargs or {}))
            setattr(original_model, _EAP_COMPAT_FLAG, True)

        if not getattr(original_model, _EAP_PREPARED_FLAG, False):
            _configure_bridge_hooks(original_model)
            setattr(original_model, _EAP_PREPARED_FLAG, True)

    return adapter


def get_model_device(model: Any) -> torch.device | str:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        device = cfg_get(cfg, "device", None)
        if device is not None:
            return device

    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            pass

    return torch.device("cpu")


def get_model_dtype(model: Any) -> torch.dtype:
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        dtype = cfg_get(cfg, "dtype", None)
        if dtype is not None:
            return dtype
    return torch.float32


def validate_model_for_eap(model: Any) -> None:
    cfg = model.cfg

    required_flags = [
        ("use_attn_result", "Model must be configured to use attention result"),
        ("use_split_qkv_input", "Model must be configured to use split qkv inputs"),
        ("use_hook_mlp_in", "Model must be configured to use hook MLP in"),
    ]
    missing = [
        f"{message} (model.cfg.{flag})"
        for flag, message in required_flags
        if hasattr(cfg, flag) and not cfg_get(cfg, flag)
    ]

    n_heads = cfg_get(cfg, "n_heads", None)
    n_key_value_heads = cfg_get(cfg, "n_key_value_heads", None)
    if n_key_value_heads is not None and n_key_value_heads != n_heads:
        if not cfg_get(cfg, "ungroup_grouped_query_attention", False):
            missing.append(
                "Model must ungroup grouped query attention "
                "(model.cfg.ungroup_grouped_query_attention)"
            )

    if missing:
        raise AssertionError("; ".join(missing))
