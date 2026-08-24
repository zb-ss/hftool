"""Benchmarking utilities for hftool.

Measures model load time, inference time, and VRAM usage with standardized test prompts.
Results are cached in ~/.hftool/benchmarks.json for reference.
"""

import json
import time
import os
import statistics
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import click


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    task: str
    model: str
    repo_id: str
    timestamp: str  # ISO format
    device: str
    dtype: Optional[str]
    
    # Timings (in seconds)
    load_time: float
    inference_time: float
    total_time: float
    
    # VRAM usage (in GB)
    vram_peak: Optional[float]
    vram_allocated: Optional[float]
    
    # Test parameters
    test_prompt: str
    test_params: Dict[str, Any]
    
    # Status
    success: bool
    error: Optional[str] = None
    revision: Optional[str] = None
    profile: Optional[str] = None
    pipeline_class: Optional[str] = None
    catalog_status: Optional[str] = None
    first_generation_time: Optional[float] = None
    warm_inference_times: Optional[List[float]] = None
    gpu_free_before: Optional[float] = None
    gpu_free_after: Optional[float] = None
    cpu_offload: bool = False


IMAGE_BENCHMARK_SUITE = (
    {
        "name": "general_landscape",
        "prompt": "A cinematic mountain lake at sunrise, natural light, detailed reflections",
        "seed": 101,
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "typography",
        "prompt": "Editorial poster with the exact title 'OPEN HORIZONS', clean modern typography",
        "seed": 202,
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "human_detail",
        "prompt": "Natural-light portrait of an adult ceramic artist at work, realistic hands and tools",
        "seed": 303,
        "width": 1024,
        "height": 1024,
    },
    {
        "name": "wide_composition",
        "prompt": "Wide architectural interior with layered depth, balanced composition, afternoon light",
        "seed": 404,
        "width": 1344,
        "height": 768,
    },
    {
        "name": "tall_cover",
        "prompt": "Vertical science-fiction book cover, one astronaut, bold negative space, no text",
        "seed": 505,
        "width": 768,
        "height": 1344,
    },
)


# Standard test prompts for each task
STANDARD_TEST_PROMPTS = {
    "text-to-image": "A colorful sunset over mountains",
    "image-to-image": {"prompt": "Make it vibrant", "image": "test.png"},
    "text-to-video": "A bird flying across the sky",
    "text-to-speech": "Hello, this is a test.",
    "automatic-speech-recognition": "test_audio.wav",
    "text-generation": "Once upon a time",
    "text-classification": "This is a great product!",
    "question-answering": {"question": "What is AI?", "context": "AI stands for Artificial Intelligence."},
    "summarization": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "translation": "Hello, how are you?",
}

# Standard test parameters for each task
STANDARD_TEST_PARAMS = {
    "text-to-image": {"width": 512, "height": 512, "num_inference_steps": 10},
    "image-to-image": {"num_inference_steps": 10},
    "text-to-video": {"num_frames": 16, "height": 256, "width": 256, "num_inference_steps": 10},
    "text-to-speech": {},
    "automatic-speech-recognition": {},
    "text-generation": {"max_new_tokens": 50},
}


def get_benchmarks_file() -> Path:
    """Get path to benchmarks cache file.

    Returns:
        Path to benchmarks.json
    """
    # Use HFTOOL_CONFIG env var if set (e.g., in Docker), otherwise use home directory
    config_dir = os.environ.get("HFTOOL_CONFIG")
    if config_dir:
        return Path(config_dir) / "benchmarks.json"
    return Path.home() / ".hftool" / "benchmarks.json"


def load_benchmarks() -> List[BenchmarkResult]:
    """Load cached benchmark results.
    
    Returns:
        List of benchmark results
    """
    benchmarks_file = get_benchmarks_file()
    
    if not benchmarks_file.exists():
        return []
    
    try:
        with open(benchmarks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = []
        for entry in data.get("benchmarks", []):
            result = BenchmarkResult(**entry)
            results.append(result)
        
        return results
    except Exception as e:
        click.echo(f"Warning: Failed to load benchmarks: {e}", err=True)
        return []


def save_benchmark(result: BenchmarkResult) -> None:
    """Save a benchmark result to cache.
    
    Args:
        result: Benchmark result to save
    """
    benchmarks_file = get_benchmarks_file()
    benchmarks_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    results = load_benchmarks()
    
    # Add new result
    results.append(result)
    
    # Keep only last 100 results per model
    # Group by model
    by_model: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        key = f"{r.task}:{r.repo_id}"
        if key not in by_model:
            by_model[key] = []
        by_model[key].append(r)
    
    # Keep last 100 per model
    pruned_results = []
    for model_results in by_model.values():
        # Sort by timestamp (newest first)
        sorted_results = sorted(model_results, key=lambda x: x.timestamp, reverse=True)
        pruned_results.extend(sorted_results[:100])
    
    # Save
    data = {
        "version": "1.0",
        "benchmarks": [asdict(r) for r in pruned_results],
    }
    
    try:
        with open(benchmarks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        click.echo(f"Warning: Failed to save benchmarks: {e}", err=True)


def measure_vram() -> Optional[tuple]:
    """Measure current VRAM usage.
    
    Returns:
        Tuple of (peak_gb, allocated_gb) or None if not available
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        # Get VRAM stats
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB

        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        return (peak, allocated)
    except Exception:
        return None


def _free_vram_gb() -> Optional[float]:
    """Return system-wide free memory for the current CUDA/HIP device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / (1024**3)
    except Exception:
        return None


def _run_image_benchmark(
    *,
    task_handler,
    model_to_load: str,
    model_info,
    model: str,
    repo_id: str,
    resolved_task: str,
    timestamp: str,
    device: str,
    dtype: Optional[str],
    load_kwargs: Dict[str, Any],
    verbose: bool,
) -> BenchmarkResult:
    """Measure exact cold load, first generation, and four warm suite cases."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    gpu_free_before = _free_vram_gb()
    load_started = time.perf_counter()
    task_handler._pipeline = task_handler.load_pipeline(model_to_load, **load_kwargs)
    load_time = time.perf_counter() - load_started

    timings: List[float] = []
    recorded_cases = []
    catalog_defaults = dict(model_info.inference_defaults) if model_info else {}
    for index, case in enumerate(IMAGE_BENCHMARK_SUITE):
        params = dict(catalog_defaults)
        params.update(
            {
                "seed": case["seed"],
                "width": case["width"],
                "height": case["height"],
            }
        )
        started = time.perf_counter()
        task_handler.run_inference(task_handler._pipeline, case["prompt"], **params)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        recorded_cases.append({**case, "elapsed_s": elapsed, "settings": params})
        if verbose:
            label = "first" if index == 0 else f"warm {index}"
            click.echo(f"  {label}: {case['name']} {elapsed:.2f}s")

    first_generation = timings[0]
    warm_times = timings[1:]
    median_warm = statistics.median(warm_times)
    vram = measure_vram()
    gpu_free_after = _free_vram_gb()
    total_time = load_time + sum(timings)
    return BenchmarkResult(
        task=resolved_task,
        model=model,
        repo_id=repo_id,
        timestamp=timestamp,
        device=device,
        dtype=dtype or (model_info.dtype if model_info else None),
        load_time=load_time,
        inference_time=median_warm,
        total_time=total_time,
        vram_peak=vram[0] if vram else None,
        vram_allocated=vram[1] if vram else None,
        test_prompt="hftool-image-suite-v1",
        test_params={
            "suite_version": 1,
            "catalog_defaults": catalog_defaults,
            "cases": recorded_cases,
            "warm_statistic": "median of 4 runs",
        },
        success=True,
        revision=model_info.revision if model_info else None,
        profile=model_info.profile if model_info else None,
        pipeline_class=model_info.pipeline_class if model_info else None,
        catalog_status=model_info.status_label if model_info else "custom",
        first_generation_time=first_generation,
        warm_inference_times=warm_times,
        gpu_free_before=gpu_free_before,
        gpu_free_after=gpu_free_after,
        cpu_offload=os.environ.get("HFTOOL_CPU_OFFLOAD", "").lower()
        in {"1", "2", "true", "yes"},
    )


def run_benchmark(
    task: str,
    model: str,
    device: str = "auto",
    dtype: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a benchmark for a model.
    
    Args:
        task: Task name
        model: Model name or repo_id
        device: Device to use
        dtype: Data type
        custom_prompt: Custom test prompt (uses standard if None)
        custom_params: Custom test parameters (uses standard if None)
        verbose: Whether to show progress
    
    Returns:
        BenchmarkResult
    """
    from datetime import datetime
    from hftool.core.models import get_model_info, find_model_by_repo_id
    from hftool.core.registry import TASK_ALIASES, get_task_config
    
    # Resolve task alias
    resolved_task = TASK_ALIASES.get(task, task)
    
    # Get model info
    model_info = None
    try:
        model_info = get_model_info(resolved_task, model)
        repo_id = model_info.repo_id
    except ValueError:
        # Try to find by repo_id
        found = find_model_by_repo_id(model)
        if found:
            _, _, model_info = found
            repo_id = model_info.repo_id
        else:
            # Assume it's a custom repo_id
            repo_id = model
            model_info = None
    
    # Get test prompt and params
    test_prompt = custom_prompt if custom_prompt is not None else STANDARD_TEST_PROMPTS.get(resolved_task, "test")
    test_params = custom_params if custom_params is not None else STANDARD_TEST_PARAMS.get(resolved_task, {})
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    if verbose:
        click.echo(f"Benchmarking {model} for {resolved_task}...")
    
    try:
        # Import task handler
        from hftool.core.download import ensure_model_available
        
        # Ensure model is downloaded
        if not os.path.exists(model):
            if verbose:
                click.echo("  Ensuring model is downloaded...")
            
            model_path = ensure_model_available(
                repo_id=repo_id,
                size_gb=model_info.size_gb if model_info else 5.0,
                task_name=resolved_task,
                model_name=model,
                gated=getattr(model_info, 'gated', False) if model_info else False,
                revision=model_info.revision if model_info else None,
                ignore_patterns=model_info.ignore_patterns if model_info else None,
            )
            model_to_load = str(model_path)
        else:
            model_to_load = model

        adapter_path = None
        if model_info and model_info.adapter:
            from hftool.core.download import ensure_model_file_available

            adapter_path = ensure_model_file_available(
                repo_id=model_info.adapter.repo_id,
                filename=model_info.adapter.weight_name,
                size_gb=model_info.adapter.size_gb,
                task_name=resolved_task,
                model_name=model,
                revision=model_info.adapter.revision,
            )
        
        # Get task config
        task_config = get_task_config(resolved_task)
        
        # Create a temporary output file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_output = tmp.name
        
        try:
            # Run inference (this includes model loading)
            # We measure load + inference together since models are loaded lazily
            if isinstance(test_prompt, str):
                input_data = test_prompt
            else:
                input_data = json.dumps(test_prompt)
            
            if verbose:
                click.echo("  Running inference...")
            
            # Start timing for the full execution (load + inference)
            exec_start = time.time()
            
            # Create task handler
            if resolved_task == "text-to-image":
                from hftool.tasks.text_to_image import create_task
                task_handler = create_task(device=device, dtype=dtype)
            elif resolved_task == "image-to-image":
                from hftool.tasks.image_to_image import create_task
                task_handler = create_task(device=device, dtype=dtype)
            elif resolved_task in ("text-to-video", "image-to-video"):
                from hftool.tasks.text_to_video import create_task
                mode = task_config.config.get("mode", "t2v")
                task_handler = create_task(device=device, dtype=dtype, mode=mode)
            elif resolved_task == "text-to-speech":
                from hftool.tasks.text_to_speech import create_task
                task_handler = create_task(device=device, dtype=dtype)
            elif resolved_task == "automatic-speech-recognition":
                from hftool.tasks.speech_to_text import create_task
                task_handler = create_task(device=device, dtype=dtype)
            else:
                from hftool.tasks.transformers_generic import create_task
                task_handler = create_task(task_name=resolved_task, device=device, dtype=dtype)

            if resolved_task == "text-to-image":
                from hftool.core.executor import build_model_runtime_kwargs

                load_kwargs, _ = (
                    build_model_runtime_kwargs(
                        model_info,
                        str(adapter_path) if adapter_path else None,
                    )
                    if model_info
                    else ({}, {})
                )
                return _run_image_benchmark(
                    task_handler=task_handler,
                    model_to_load=model_to_load,
                    model_info=model_info,
                    model=model,
                    repo_id=repo_id,
                    resolved_task=resolved_task,
                    timestamp=timestamp,
                    device=device,
                    dtype=dtype,
                    load_kwargs=load_kwargs,
                    verbose=verbose,
                )
            
            # Execute (this loads model and runs inference)
            task_handler.execute(
                model=model_to_load,
                input_data=input_data,
                output_path=tmp_output,
                **test_params
            )
            
            total_time = time.time() - exec_start
            # Estimate load vs inference (roughly 1/3 load, 2/3 inference for most models)
            load_time = total_time * 0.3
            inference_time = total_time * 0.7
            
            if verbose:
                click.echo(f"  Total time: {total_time:.2f}s")
                click.echo(f"    (Estimated) Load: {load_time:.2f}s, Inference: {inference_time:.2f}s")
            
            # Measure VRAM
            vram_after = measure_vram()
            vram_peak = None
            vram_allocated = None
            
            if vram_after:
                vram_peak = vram_after[0]
                vram_allocated = vram_after[1]
                
                if verbose and vram_peak:
                    click.echo(f"  VRAM peak: {vram_peak:.2f} GB")
                    click.echo(f"  VRAM allocated: {vram_allocated:.2f} GB")
            
            # Clean up
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
            
            return BenchmarkResult(
                task=resolved_task,
                model=model,
                repo_id=repo_id,
                timestamp=timestamp,
                device=device,
                dtype=dtype,
                load_time=load_time,
                inference_time=inference_time,
                total_time=total_time,
                vram_peak=vram_peak,
                vram_allocated=vram_allocated,
                test_prompt=str(test_prompt),
                test_params=test_params,
                success=True,
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_output):
                try:
                    os.unlink(tmp_output)
                except Exception:
                    pass
    
    except Exception as e:
        if verbose:
            click.echo(f"  Benchmark failed: {e}", err=True)
        
        return BenchmarkResult(
            task=resolved_task,
            model=model,
            repo_id=repo_id,
            timestamp=timestamp,
            device=device,
            dtype=dtype,
            load_time=0.0,
            inference_time=0.0,
            total_time=0.0,
            vram_peak=None,
            vram_allocated=None,
            test_prompt=str(test_prompt),
            test_params=test_params,
            success=False,
            error=str(e),
        )


def get_benchmark_history(task: Optional[str] = None, model: Optional[str] = None) -> List[BenchmarkResult]:
    """Get benchmark history with optional filtering.
    
    Args:
        task: Filter by task (None = all)
        model: Filter by model repo_id (None = all)
    
    Returns:
        List of matching benchmark results
    """
    results = load_benchmarks()
    
    if task:
        from hftool.core.registry import TASK_ALIASES
        resolved_task = TASK_ALIASES.get(task, task)
        results = [r for r in results if r.task == resolved_task]
    
    if model:
        results = [r for r in results if model.lower() in r.repo_id.lower() or model.lower() in r.model.lower()]
    
    return results
