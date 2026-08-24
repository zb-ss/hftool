"""Regression coverage for the packaged, versioned model catalog."""

from importlib import resources

import pytest


def test_packaged_catalog_contains_verified_image_profiles():
    from hftool.core.models import get_model_info

    flux = get_model_info("text-to-image", "flux2-klein-4b")
    sana = get_model_info("text-to-image", "sana-sprint-1.6b")
    lightning = get_model_info("text-to-image", "qwen-image-2512-lightning")

    assert flux.pipeline_class == "Flux2KleinPipeline"
    assert flux.license == "Apache-2.0"
    assert sana.pipeline_class == "SanaSprintPipeline"
    assert sana.inference_defaults["num_inference_steps"] == 2
    assert lightning.adapter is not None
    assert lightning.adapter.repo_id == "lightx2v/Qwen-Image-2512-Lightning"
    assert lightning.adapter.weight_name == (
        "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
    )


def test_packaged_catalog_is_an_installable_resource():
    catalog = resources.files("hftool.catalog").joinpath("models-v1.toml")
    assert catalog.is_file()
    assert "version = 1" in catalog.read_text(encoding="utf-8")


def test_historical_flux2_alias_keeps_9b_semantics():
    from hftool.core.models import get_model_info, get_model_key

    assert get_model_key("image-to-image", "flux2-klein") == "flux2-klein-9b"
    assert get_model_info("image-to-image", "flux2-klein").repo_id.endswith(
        "FLUX.2-klein-9B"
    )
    assert get_model_info("image-to-image", "flux2-klein").gated is True
    assert get_model_info("image-to-image", "flux2-klein").commercial_use is False


@pytest.mark.parametrize(
    "body, error_match",
    [
        ("[tasks.x.models.y]\nrepo_id='a/b'\n", "missing.*catalog"),
        (
            """
[catalog]
version = 1
[tasks.x.models.y]
repo_id = "a/b"
name = "Y"
model_type = "diffusers"
size_gb = 1
description = ""
status = "retired"
""",
            "unknown status",
        ),
        (
            """
[catalog]
version = 1
[tasks.x.models.one]
repo_id = "a/one"
name = "One"
model_type = "diffusers"
size_gb = 1
description = ""
aliases = ["same"]
[tasks.x.models.two]
repo_id = "a/two"
name = "Two"
model_type = "diffusers"
size_gb = 1
description = ""
aliases = ["same"]
""",
            "alias.*conflicts",
        ),
    ],
)
def test_malformed_catalogs_fail_with_actionable_errors(tmp_path, body, error_match):
    from hftool.core.models import CatalogError, read_model_catalog

    path = tmp_path / "models.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(CatalogError, match=error_match):
        read_model_catalog(path)


def test_runtime_gpu_policy_is_catalog_driven():
    from hftool.core.models import get_catalog_runtime_config

    policy = get_catalog_runtime_config("gpu_selection")
    assert policy == {"safety_reserve_gb": 2.0, "display_penalty_gb": 4.0}


def test_curated_profiles_pin_base_and_adapter_revisions():
    from hftool.core.models import get_model_info

    profile = get_model_info("text-to-image", "qwen-image-2512-lightning")
    assert len(profile.revision or "") == 40
    assert profile.adapter is not None
    assert len(profile.adapter.revision or "") == 40


def test_download_status_rejects_a_stale_pinned_revision(tmp_path, monkeypatch):
    from hftool.core.download import get_download_status, get_model_path

    monkeypatch.setenv("HFTOOL_MODELS_DIR", str(tmp_path))
    model_path = get_model_path("example/model")
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")
    (model_path / ".hftool-revision").write_text("old\n", encoding="utf-8")

    assert get_download_status("example/model", "new") == "partial"
    assert get_download_status("example/model", "old") == "downloaded"
