"""Catalog-driven image pipeline and profile dispatch tests."""

import sys
from types import SimpleNamespace

import pytest


class _FakePipeline:
    last_model = None
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.last_model = model
        cls.last_kwargs = kwargs
        return cls()

    def to(self, device):
        self.device = device
        return self

    def enable_vae_slicing(self):
        pass

    def enable_vae_tiling(self):
        pass


class _AdapterPipeline(_FakePipeline):
    def load_lora_weights(self, path, **kwargs):
        self.adapter_call = (path, kwargs)

    def set_adapters(self, name, adapter_weights):
        self.set_adapter_call = (name, adapter_weights)


class _FakeScheduler:
    @classmethod
    def from_config(cls, config):
        return {"scheduler": dict(config)}


@pytest.fixture
def isolated_diffusion_runtime(monkeypatch):
    import torch

    fake_diffusers = SimpleNamespace(
        __version__="0.40.0",
        Flux2KleinPipeline=type("Flux2KleinPipeline", (_FakePipeline,), {}),
        SanaSprintPipeline=type("SanaSprintPipeline", (_FakePipeline,), {}),
        QwenImagePipeline=type("QwenImagePipeline", (_AdapterPipeline,), {}),
        FlowMatchEulerDiscreteScheduler=_FakeScheduler,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr("hftool.utils.deps.check_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr("hftool.core.device.configure_rocm_env", lambda: None)
    monkeypatch.setattr("hftool.core.device.detect_device", lambda: "cpu")
    monkeypatch.setattr(
        "hftool.core.device.get_multi_gpu_kwargs",
        lambda **kwargs: {
            "use_multi_gpu": False,
            "message": "",
            "device_map": None,
            "max_memory": None,
        },
    )
    return fake_diffusers, torch


@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("flux2-klein-4b", "Flux2KleinPipeline"),
        ("sana-sprint-1.6b", "SanaSprintPipeline"),
    ],
)
def test_catalog_dispatches_exact_pipeline_class(
    isolated_diffusion_runtime,
    model_name,
    expected_class,
):
    from hftool.core.executor import build_model_runtime_kwargs
    from hftool.core.models import get_model_info
    from hftool.tasks.diffusion_utils import load_catalog_pipeline

    info = get_model_info("text-to-image", model_name)
    load_kwargs, _ = build_model_runtime_kwargs(info)
    pipeline = load_catalog_pipeline(
        "/models/base",
        device="cpu",
        requested_dtype=None,
        load_kwargs=load_kwargs,
    )

    assert type(pipeline).__name__ == expected_class
    assert type(pipeline).last_model == "/models/base"
    assert type(pipeline).last_kwargs["torch_dtype"] is isolated_diffusion_runtime[1].float32


def test_lightning_profile_loads_only_exact_adapter_and_scheduler(
    isolated_diffusion_runtime,
):
    from hftool.core.executor import build_model_runtime_kwargs
    from hftool.core.models import get_model_info
    from hftool.tasks.diffusion_utils import load_catalog_pipeline

    info = get_model_info("text-to-image", "qwen-image-2512-lightning")
    load_kwargs, inference_kwargs = build_model_runtime_kwargs(
        info,
        "/models/lightning",
    )
    pipeline = load_catalog_pipeline(
        "/models/qwen",
        device="cpu",
        requested_dtype=None,
        load_kwargs=load_kwargs,
    )

    assert pipeline.adapter_call == (
        "/models/lightning",
        {
            "weight_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            "adapter_name": "hftool_profile",
        },
    )
    assert pipeline.set_adapter_call == ("hftool_profile", 1.0)
    assert type(pipeline).last_kwargs["scheduler"]["scheduler"]["shift"] == 1.0
    assert inference_kwargs["num_inference_steps"] == 4
    assert inference_kwargs["true_cfg_scale"] == 1.0


def test_runtime_kwargs_normalize_downloaded_adapter_file_to_parent():
    from hftool.core.executor import build_model_runtime_kwargs
    from hftool.core.models import get_model_info

    info = get_model_info("text-to-image", "qwen-image-2512-lightning")
    adapter_file = f"/models/lightning/{info.adapter.weight_name}"
    load_kwargs, _ = build_model_runtime_kwargs(info, adapter_file)
    assert load_kwargs["_adapter_path"] == "/models/lightning"


def test_image_tasks_use_t2i_dependency_extra(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hftool.utils.deps.check_dependencies",
        lambda dependencies, extra=None: calls.append((dependencies, extra)),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(__version__="0", DiffusionPipeline=_FakePipeline),
    )
    monkeypatch.setattr("hftool.core.device.configure_rocm_env", lambda: None)
    monkeypatch.setattr("hftool.core.device.detect_device", lambda: "cpu")
    monkeypatch.setattr(
        "hftool.core.device.get_multi_gpu_kwargs",
        lambda **kwargs: {"use_multi_gpu": False, "message": ""},
    )

    from hftool.tasks.diffusion_utils import load_catalog_pipeline

    load_catalog_pipeline(
        "/models/custom",
        device="cpu",
        requested_dtype="float32",
        load_kwargs={},
    )
    assert calls == [(["diffusers", "torch", "accelerate"], "with_t2i")]


def test_benchmark_suite_has_first_plus_four_warm_cases():
    from hftool.core.benchmark import IMAGE_BENCHMARK_SUITE

    assert len(IMAGE_BENCHMARK_SUITE) == 5
    assert len({case["seed"] for case in IMAGE_BENCHMARK_SUITE}) == 5
    assert {case["name"] for case in IMAGE_BENCHMARK_SUITE} >= {
        "typography",
        "human_detail",
        "wide_composition",
    }
