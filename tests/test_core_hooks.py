import torch

from nndbg.core.cache import ActivationCache
from nndbg.core.collect import collect_activations
from nndbg.core.hooks import HookManager
from nndbg.core.registry import LayerRegistry


def test_registry_discovers_named_modules(tiny_mlp):
    registry = LayerRegistry(tiny_mlp)
    assert set(registry.names()) == {"fc1", "act", "fc2"}
    assert registry["fc1"] is tiny_mlp.fc1
    assert registry.find(r"^fc") == ["fc1", "fc2"]
    assert registry.of_type(torch.nn.Linear) == ["fc1", "fc2"]


def test_registry_missing_layer_raises_helpful_error(tiny_mlp):
    registry = LayerRegistry(tiny_mlp)
    try:
        registry["does_not_exist"]
    except KeyError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_hook_manager_capture_records_correct_shapes(tiny_mlp):
    hooks = HookManager(tiny_mlp, LayerRegistry(tiny_mlp))
    x = torch.randn(2, 4)
    with hooks.capture(["fc1", "fc2"]) as cache:
        tiny_mlp(x)
    assert cache["fc1"].shape == (2, 16)
    assert cache["fc2"].shape == (2, 3)
    assert list(cache.layers()) == ["fc1", "fc2"]


def test_hook_manager_capture_detaches_by_default(tiny_mlp):
    hooks = HookManager(tiny_mlp, LayerRegistry(tiny_mlp))
    with hooks.capture(["fc1"], detach=True) as cache:
        tiny_mlp(torch.randn(1, 4))
    assert not cache["fc1"].requires_grad


def test_hook_manager_capture_retains_graph_when_not_detached(tiny_mlp):
    hooks = HookManager(tiny_mlp, LayerRegistry(tiny_mlp))
    x = torch.randn(1, 4)
    with hooks.capture(["fc1"], detach=False) as cache:
        out = tiny_mlp(x)
    assert cache["fc1"].requires_grad
    out.sum().backward()  # graph through fc1 is intact


def test_hook_manager_patch_replaces_layer_output(tiny_mlp):
    hooks = HookManager(tiny_mlp, LayerRegistry(tiny_mlp))
    x = torch.randn(1, 4)
    with hooks.capture(["fc1"]) as cache:
        tiny_mlp(x)
    zeroed = torch.zeros_like(cache["fc1"])
    with hooks.capture(["fc2"]) as patched_cache, hooks.patch({"fc1": zeroed}):
        tiny_mlp(x)
    # fc2's input came from a zeroed fc1 output, post-ReLU(0) == 0, so fc2's
    # output should equal its bias term exactly.
    expected = tiny_mlp.fc2.bias.detach()
    assert torch.allclose(patched_cache["fc2"].squeeze(0), expected, atol=1e-6)


def test_activation_cache_save_and_load(tmp_path):
    cache = ActivationCache()
    cache["a"] = torch.randn(3, 3)
    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = ActivationCache.load(path)
    assert torch.allclose(loaded["a"], cache["a"])


def test_collect_activations_pools_and_stacks(tiny_mlp):
    hooks = HookManager(tiny_mlp, LayerRegistry(tiny_mlp))
    inputs = [torch.randn(4) for _ in range(5)]
    out = collect_activations(tiny_mlp, hooks, inputs, ["fc1"], pooling="mean")
    assert out["fc1"].shape == (5, 16)
