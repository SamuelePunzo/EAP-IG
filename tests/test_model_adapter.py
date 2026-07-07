from types import SimpleNamespace

import pytest
import torch

from eap.model_adapter import EAPModelAdapter, prepare_model_for_eap, validate_model_for_eap


class HookContext:
    def __enter__(self):
        return "entered"

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeBridge:
    def __init__(self, *, n_key_value_heads=None, ungroup_grouped_query_attention=False):
        self.cfg = SimpleNamespace(
            device="cpu",
            dtype=torch.float32,
            use_attn_in=True,
            use_attn_result=False,
            use_split_qkv_input=False,
            use_hook_mlp_in=False,
            n_heads=2,
            n_key_value_heads=n_key_value_heads,
            ungroup_grouped_query_attention=ungroup_grouped_query_attention,
        )
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.hook_aliases = {}
        self.compatibility_calls = 0
        self.attn_in_calls = []
        self.attn_result_calls = []
        self.split_qkv_calls = []
        self.zero_grad_calls = 0

    def enable_compatibility_mode(self, **kwargs):
        self.compatibility_calls += 1
        self.compatibility_kwargs = kwargs

    def set_use_attn_in(self, value):
        self.attn_in_calls.append(value)
        self.cfg.use_attn_in = value

    def set_use_attn_result(self, value):
        self.attn_result_calls.append(value)
        self.cfg.use_attn_result = value

    def set_use_split_qkv_input(self, value):
        self.split_qkv_calls.append(value)
        self.cfg.use_split_qkv_input = value

    def run_with_hooks(self, *args, **kwargs):
        return args, kwargs

    def hooks(self, *args, **kwargs):
        return HookContext()

    def to_tokens(self, inputs, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def zero_grad(self):
        self.zero_grad_calls += 1

    def __call__(self, tokens, **kwargs):
        return tokens + 1


class CompatibleFakeBridge(FakeBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hook_mlp_in_calls = []

    def set_use_hook_mlp_in(self, value):
        self.hook_mlp_in_calls.append(value)
        self.cfg.use_hook_mlp_in = value


def test_prepare_bridge_enables_compatibility_and_required_hooks_once():
    bridge = CompatibleFakeBridge()

    adapter = prepare_model_for_eap(
        bridge,
        compatibility_mode_kwargs={"no_processing": True},
    )
    prepare_model_for_eap(adapter)

    assert isinstance(adapter, EAPModelAdapter)
    assert bridge.compatibility_calls == 1
    assert bridge.compatibility_kwargs == {"no_processing": True}
    assert bridge.attn_in_calls == [False]
    assert bridge.attn_result_calls == [True]
    assert bridge.split_qkv_calls == [True]
    assert bridge.hook_mlp_in_calls == [True]
    assert bridge.cfg.use_attn_in is False
    assert bridge.cfg.use_attn_result is True
    assert bridge.cfg.use_split_qkv_input is True
    assert bridge.cfg.use_hook_mlp_in is True


def test_adapter_delegates_model_surface():
    bridge = FakeBridge()
    adapter = prepare_model_for_eap(bridge)

    assert adapter.to_tokens(["hello"]).tolist() == [[1, 2, 3]]
    with adapter.hooks() as value:
        assert value == "entered"
    assert adapter(torch.tensor([1])).tolist() == [2]
    adapter.zero_grad()
    assert bridge.zero_grad_calls == 1


def test_validate_accepts_prepared_bridge_by_default():
    bridge = CompatibleFakeBridge()
    adapter = prepare_model_for_eap(bridge)

    validate_model_for_eap(adapter)


def test_validate_rejects_bridge_without_legacy_hook_capability_by_default():
    bridge = FakeBridge()
    adapter = prepare_model_for_eap(bridge)

    with pytest.raises(AssertionError, match="legacy-compatible EAP hook semantics"):
        validate_model_for_eap(adapter)


def test_prepare_bridge_auto_ungroups_grouped_query_attention():
    bridge = CompatibleFakeBridge(n_key_value_heads=1, ungroup_grouped_query_attention=False)

    adapter = prepare_model_for_eap(bridge)

    assert isinstance(adapter, EAPModelAdapter)
    assert bridge.cfg.ungroup_grouped_query_attention is True
    validate_model_for_eap(adapter)


class FakeUnsupportedAttentionBridge(CompatibleFakeBridge):
    def set_use_attn_result(self, value):
        raise NotImplementedError("use_attn_result: unsupported attention fork")

    def set_use_split_qkv_input(self, value):
        raise NotImplementedError("use_split_qkv_input: unsupported attention fork")


def test_validate_rejects_bridge_without_required_attention_fork():
    bridge = FakeUnsupportedAttentionBridge()

    adapter = prepare_model_for_eap(bridge)

    with pytest.raises(NotImplementedError, match="hook surface required for EAP"):
        validate_model_for_eap(adapter)
