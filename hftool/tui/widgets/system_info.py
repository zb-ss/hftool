"""SystemInfo widget — device, VRAM, GPU, Docker status panel."""

import os

from textual.widgets import Static


class SystemInfo(Static):
    """Displays system information: device, GPU, VRAM, Docker status."""

    DEFAULT_CSS = """
    SystemInfo {
        border: round $accent 40%;
        border-title-color: $text-muted;
        border-title-style: bold;
        padding: 1 2;
        height: auto;
    }
    SystemInfo:focus-within {
        border: round $accent 100%;
    }
    """

    def on_mount(self) -> None:
        self.border_title = "System"
        self.update(self._build_info())

    def _build_info(self) -> str:
        lines = []
        in_docker = os.environ.get("HFTOOL_IN_DOCKER")

        # Device detection uses global live free VRAM, not process-local
        # allocator counters, so gaming/desktop pressure is visible.
        try:
            import torch
            if torch.cuda.is_available():
                from hftool.core.device import get_all_gpus, select_compute_gpu

                gpus = get_all_gpus()
                device_type = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
                lines.append(f"[cyan]Device:[/]   {device_type}")
                for gpu in gpus:
                    free = gpu.free_vram_gb if gpu.free_vram_gb is not None else gpu.vram_gb
                    display = " · display" if gpu.has_display else ""
                    lines.append(
                        f"[cyan]GPU {gpu.resolved_physical_index}:[/]   {gpu.name} · "
                        f"cuda:{gpu.index} · {free:.1f}/{gpu.vram_gb:.1f} GB free{display}"
                    )
                selection = select_compute_gpu(gpus=gpus)
                if selection.gpu is not None:
                    lines.append(
                        f"[cyan]Auto:[/]     physical {selection.gpu.resolved_physical_index} "
                        f"→ cuda:{selection.gpu.index}"
                    )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lines.append("[cyan]Device:[/]   MPS (Apple Silicon)")
            else:
                lines.append("[cyan]Device:[/]   CPU")

            lines.append(f"[cyan]PyTorch:[/]  {torch.__version__}")
        except ImportError:
            lines.append("[cyan]Device:[/]   [yellow]PyTorch not installed[/]")

        # Docker context
        if in_docker:
            lines.append("")
            lines.append("[cyan]Docker:[/]   [green]Running in container[/]")
            mapping = os.environ.get("HFTOOL_PHYSICAL_GPU_INDICES", "")
            if mapping:
                lines.append(f"[cyan]Mapping:[/]  physical {mapping} → visible 0…")
            rocm_ver = os.environ.get("ROCM_VERSION", "")
            if rocm_ver:
                lines.append(f"[cyan]ROCm:[/]     {rocm_ver}")
        else:
            lines.append("")
            try:
                from hftool.utils.docker import detect_hardware
                hw = detect_hardware()
                docker_status = "[green]Available[/]" if hw.docker_available else "[dim]Not installed[/]"
                lines.append(f"[cyan]Docker:[/]   {docker_status}")
                if hw.docker_available and hw.recommended_image:
                    img_status = "[green]Ready[/]" if hw.image_available else "[yellow]Not built[/]"
                    lines.append(f"[cyan]Image:[/]    {img_status}")
            except Exception:
                lines.append("[cyan]Docker:[/]   [dim]N/A[/]")

        return "\n".join(lines)
