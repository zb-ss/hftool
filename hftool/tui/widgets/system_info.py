"""SystemInfo widget — device, VRAM, GPU, Docker status panel."""

import os

from textual.app import ComposeResult
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

        # Device detection — try torch first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                device_type = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
                lines.append(f"[cyan]Device:[/]   {device_type}")
                lines.append(f"[cyan]GPU:[/]      {gpu_name}")
                if gpu_count > 1:
                    lines.append(f"[cyan]GPUs:[/]     {gpu_count}")

                # VRAM
                try:
                    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                    free = (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / (1024**3)
                    lines.append(f"[cyan]VRAM:[/]     {total:.1f} GB total, {free:.1f} GB free")
                except Exception:
                    pass
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lines.append(f"[cyan]Device:[/]   MPS (Apple Silicon)")
            else:
                lines.append(f"[cyan]Device:[/]   CPU")

            lines.append(f"[cyan]PyTorch:[/]  {torch.__version__}")
        except ImportError:
            lines.append(f"[cyan]Device:[/]   [yellow]PyTorch not installed[/]")

        # Docker context
        if in_docker:
            lines.append("")
            lines.append(f"[cyan]Docker:[/]   [green]Running in container[/]")
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
                lines.append(f"[cyan]Docker:[/]   [dim]N/A[/]")

        return "\n".join(lines)
