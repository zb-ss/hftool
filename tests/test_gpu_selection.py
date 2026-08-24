"""Pure tests for live-memory selection and Docker identity mapping."""

from unittest.mock import patch


def _gpu(index, free, *, display=False, physical=None):
    from hftool.core.device import GPUInfo

    return GPUInfo(
        index=index,
        name=f"GPU {index}",
        vram_gb=24.0,
        free_vram_gb=free,
        pci_bus=f"0000:0{index}:00.0",
        has_display=display,
        render_device=f"/dev/dri/renderD{128 + index}",
        is_rocm=True,
        physical_index=physical,
    )


def test_auto_selection_uses_live_headroom_not_fixed_index():
    from hftool.core.device import select_compute_gpu

    selection = select_compute_gpu(
        gpus=[_gpu(0, 8.0), _gpu(1, 21.0)],
        safety_reserve_gb=2.0,
        display_penalty_gb=4.0,
    )
    assert selection.visible_index == 1


def test_adequate_display_gpu_beats_inadequate_compute_gpu():
    from hftool.core.device import select_compute_gpu

    selection = select_compute_gpu(
        required_vram_gb=13.0,
        gpus=[_gpu(0, 20.0, display=True), _gpu(1, 14.0)],
        safety_reserve_gb=2.0,
        display_penalty_gb=4.0,
    )
    assert selection.visible_index == 0
    assert selection.adequate is True


def test_inadequate_selection_explains_offload_requirement():
    from hftool.core.device import select_compute_gpu

    selection = select_compute_gpu(
        required_vram_gb=20.0,
        gpus=[_gpu(0, 18.0), _gpu(1, 17.0)],
        safety_reserve_gb=2.0,
        display_penalty_gb=4.0,
    )
    assert selection.adequate is False
    assert "CPU offload" in selection.reason


def test_no_gpu_path_is_actionable():
    from hftool.core.device import select_compute_gpu

    selection = select_compute_gpu(
        required_vram_gb=13.0,
        gpus=[],
        safety_reserve_gb=2.0,
        display_penalty_gb=4.0,
    )
    assert selection.gpu is None
    assert selection.adequate is False
    assert "No compatible GPU" in selection.format_message()


def test_visible_to_physical_mapping_prefers_hftool_identity(monkeypatch):
    from hftool.core.device import get_visible_physical_indices

    monkeypatch.setenv("HFTOOL_PHYSICAL_GPU_INDICES", "3,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    assert get_visible_physical_indices(2) == [3, 1]


def test_rocm_docker_mapping_keeps_rocr_physical_and_hip_renumbered(tmp_path):
    from hftool.utils.docker import GPUPlatform, HardwareInfo, get_docker_run_command

    hardware = HardwareInfo(
        platform=GPUPlatform.ROCM,
        gpu_name="AMD GPU",
        gpu_available=True,
        docker_available=True,
        docker_compose_available=True,
        image_available=True,
        recommended_image="hftool:rocm",
    )
    with (
        patch("hftool.utils.docker._fix_hf_cache_permissions"),
        patch(
            "hftool.utils.docker.get_render_devices_for_gpus",
            return_value=["/dev/dri/renderD129"],
        ),
    ):
        command = get_docker_run_command(
            hardware,
            ["tui", "--native"],
            workdir=str(tmp_path),
            gpu_indices=[1],
            mount_home=False,
            tty=False,
        )

    assert "ROCR_VISIBLE_DEVICES=1" in command
    assert "HIP_VISIBLE_DEVICES=0" in command
    assert "HFTOOL_PHYSICAL_GPU_INDICES=1" in command
    assert "ROCR_VISIBLE_DEVICES=0" not in command


def test_host_auto_selection_accounts_for_display_penalty():
    from hftool.utils.docker import GPUInfo, select_amd_gpu

    selected = select_amd_gpu(
        [
            GPUInfo(0, "display", "r0", "c0", 24.0, 22.0, True),
            GPUInfo(1, "compute", "r1", "c1", 24.0, 20.0, False),
        ],
        display_penalty_gb=4.0,
    )
    assert selected is not None
    assert selected.index == 1


def test_multi_gpu_is_only_enabled_when_explicit(tmp_path):
    from hftool.utils.docker import GPUPlatform, HardwareInfo, get_docker_run_command

    hardware = HardwareInfo(
        platform=GPUPlatform.ROCM,
        gpu_name="AMD GPU",
        gpu_available=True,
        docker_available=True,
        docker_compose_available=True,
        image_available=True,
        recommended_image="hftool:rocm",
    )
    with patch("hftool.utils.docker._fix_hf_cache_permissions"):
        default_command = get_docker_run_command(
            hardware,
            ["tui", "--native"],
            workdir=str(tmp_path),
            mount_home=False,
            tty=False,
        )
        multi_command = get_docker_run_command(
            hardware,
            ["tui", "--native"],
            workdir=str(tmp_path),
            mount_home=False,
            tty=False,
            multi_gpu=True,
        )

    assert "HFTOOL_MULTI_GPU=1" not in default_command
    assert "HFTOOL_MULTI_GPU=1" in multi_command
