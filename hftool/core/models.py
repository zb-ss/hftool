"""Validated loader for hftool's packaged, versioned model catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


CATALOG_VERSION = 1
_VALID_STATUS = frozenset({"standard", "recommended", "legacy", "experimental"})
_CLASS_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class CatalogError(ValueError):
    """Raised when the packaged model catalog is malformed."""


class ModelType(Enum):
    """Type of model or pipeline."""

    DIFFUSERS = "diffusers"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"


@dataclass
class AdapterInfo:
    """A model profile adapter loaded on top of a base model."""

    repo_id: str
    weight_name: str
    size_gb: float
    scale: float = 1.0
    revision: Optional[str] = None
    scheduler_class: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a downloadable model or model profile."""

    repo_id: str
    name: str
    model_type: ModelType
    size_gb: float
    is_default: bool = False
    description: str = ""
    revision: Optional[str] = None
    ignore_patterns: List[str] = field(default_factory=list)
    pip_dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    gated: bool = False
    aliases: List[str] = field(default_factory=list)
    status: str = "standard"
    license: Optional[str] = None
    commercial_use: Optional[bool] = None
    min_vram_gb: Optional[float] = None
    recommended_vram_gb: Optional[float] = None
    dtype: Optional[str] = None
    pipeline_class: Optional[str] = None
    profile: Optional[str] = None
    use_case: Optional[str] = None
    adapter: Optional[AdapterInfo] = None
    inference_defaults: Dict[str, Any] = field(default_factory=dict)
    load_defaults: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep the historical metadata mapping compatible with callers."""
        if not self.inference_defaults and self.metadata:
            excluded = {"pipeline_class", "subfolder", "variant", "torch_dtype", "use_safetensors"}
            self.inference_defaults = {
                key: value for key, value in self.metadata.items() if key not in excluded
            }
        if not self.load_defaults and self.metadata:
            load_keys = {"subfolder", "variant", "torch_dtype", "use_safetensors"}
            self.load_defaults = {
                key: value for key, value in self.metadata.items() if key in load_keys
            }
        merged_metadata = {**self.load_defaults, **self.inference_defaults}
        if self.pipeline_class:
            merged_metadata["pipeline_class"] = self.pipeline_class
        self.metadata = merged_metadata

    @property
    def short_name(self) -> str:
        """Return the final path component of the base repository ID."""
        return self.repo_id.split("/")[-1]

    @property
    def size_str(self) -> str:
        """Return the approximate complete download size."""
        total_size = self.size_gb + (self.adapter.size_gb if self.adapter else 0.0)
        if total_size >= 1:
            return f"{total_size:.1f} GB"
        return f"{int(total_size * 1024)} MB"

    @property
    def is_recommended(self) -> bool:
        """Whether this model is recommended for at least one use case."""
        return self.status == "recommended"

    @property
    def is_legacy(self) -> bool:
        """Whether this model is retained for backward compatibility."""
        return self.status == "legacy"

    @property
    def status_label(self) -> str:
        """Return a compact user-facing status label."""
        if self.is_default:
            return "default"
        return self.status


@dataclass(frozen=True)
class ModelCatalog:
    """Validated catalog data and runtime defaults."""

    version: int
    verified: Optional[str]
    models: Dict[str, Dict[str, ModelInfo]]
    runtime: Dict[str, Any]


def _require_string(
    entry: Mapping[str, Any],
    field_name: str,
    location: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CatalogError(f"{location}: '{field_name}' must be a non-empty string")
    return value


def _optional_number(
    entry: Mapping[str, Any],
    field_name: str,
    location: str,
) -> Optional[float]:
    value = entry.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise CatalogError(f"{location}: '{field_name}' must be a non-negative number")
    return float(value)


def _string_list(entry: Mapping[str, Any], field_name: str, location: str) -> List[str]:
    value = entry.get(field_name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CatalogError(f"{location}: '{field_name}' must be an array of strings")
    return list(value)


def _optional_string(
    entry: Mapping[str, Any],
    field_name: str,
    location: str,
) -> Optional[str]:
    value = entry.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{location}: '{field_name}' must be a non-empty string")
    return value


def _parse_adapter(raw: Any, location: str) -> Optional[AdapterInfo]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CatalogError(f"{location}: 'adapter' must be a table")

    adapter_location = f"{location}.adapter"
    scheduler_raw = raw.get("scheduler", {})
    if not isinstance(scheduler_raw, Mapping):
        raise CatalogError(f"{adapter_location}: 'scheduler' must be a table")

    scheduler_config = dict(scheduler_raw)
    scheduler_class = scheduler_config.pop("class_name", None)
    if scheduler_class is not None:
        if not isinstance(scheduler_class, str) or not _CLASS_NAME_RE.fullmatch(scheduler_class):
            raise CatalogError(
                f"{adapter_location}: scheduler class_name must be a simple Python class name"
            )

    size_gb = _optional_number(raw, "size_gb", adapter_location)
    if size_gb is None or size_gb <= 0:
        raise CatalogError(f"{adapter_location}: 'size_gb' must be greater than zero")

    scale = raw.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise CatalogError(f"{adapter_location}: 'scale' must be numeric")

    return AdapterInfo(
        repo_id=_require_string(raw, "repo_id", adapter_location),
        weight_name=_require_string(raw, "weight_name", adapter_location),
        size_gb=size_gb,
        scale=float(scale),
        revision=_optional_string(raw, "revision", adapter_location),
        scheduler_class=scheduler_class,
        scheduler_config=scheduler_config,
    )


def _parse_model(raw: Any, location: str) -> ModelInfo:
    if not isinstance(raw, Mapping):
        raise CatalogError(f"{location}: model entry must be a table")

    model_type_value = _require_string(raw, "model_type", location)
    try:
        model_type = ModelType(model_type_value)
    except ValueError as error:
        valid = ", ".join(item.value for item in ModelType)
        raise CatalogError(
            f"{location}: unknown model_type '{model_type_value}' (expected {valid})"
        ) from error

    size_gb = _optional_number(raw, "size_gb", location)
    if size_gb is None or size_gb <= 0:
        raise CatalogError(f"{location}: 'size_gb' must be greater than zero")

    status = raw.get("status", "standard")
    if status not in _VALID_STATUS:
        raise CatalogError(
            f"{location}: unknown status '{status}' (expected {', '.join(sorted(_VALID_STATUS))})"
        )

    pipeline_class = raw.get("pipeline_class")
    if pipeline_class is not None:
        if not isinstance(pipeline_class, str) or not _CLASS_NAME_RE.fullmatch(pipeline_class):
            raise CatalogError(f"{location}: pipeline_class must be a simple Python class name")

    inference = raw.get("inference", {})
    load = raw.get("load", {})
    if not isinstance(inference, Mapping):
        raise CatalogError(f"{location}: 'inference' must be a table")
    if not isinstance(load, Mapping):
        raise CatalogError(f"{location}: 'load' must be a table")

    commercial_use = raw.get("commercial_use")
    if commercial_use is not None and not isinstance(commercial_use, bool):
        raise CatalogError(f"{location}: 'commercial_use' must be true or false")

    is_default = raw.get("default", False)
    gated = raw.get("gated", False)
    if not isinstance(is_default, bool):
        raise CatalogError(f"{location}: 'default' must be true or false")
    if not isinstance(gated, bool):
        raise CatalogError(f"{location}: 'gated' must be true or false")

    min_vram_gb = _optional_number(raw, "min_vram_gb", location)
    recommended_vram_gb = _optional_number(raw, "recommended_vram_gb", location)
    if (
        min_vram_gb is not None
        and recommended_vram_gb is not None
        and recommended_vram_gb < min_vram_gb
    ):
        raise CatalogError(
            f"{location}: recommended_vram_gb must be at least min_vram_gb"
        )

    dtype = _optional_string(raw, "dtype", location)
    if dtype is not None and dtype not in {"bfloat16", "float16", "float32"}:
        raise CatalogError(f"{location}: unsupported dtype '{dtype}'")

    profile = _optional_string(raw, "profile", location)
    adapter = _parse_adapter(raw.get("adapter"), location)
    if profile == "adapter" and adapter is None:
        raise CatalogError(f"{location}: adapter profile is missing [adapter] data")
    if adapter is not None and profile != "adapter":
        raise CatalogError(f"{location}: adapter data requires profile = 'adapter'")

    aliases = _string_list(raw, "aliases", location)
    normalized_aliases = [alias.lower() for alias in aliases]
    if len(set(normalized_aliases)) != len(normalized_aliases):
        raise CatalogError(f"{location}: aliases must be unique ignoring case")

    return ModelInfo(
        repo_id=_require_string(raw, "repo_id", location),
        name=_require_string(raw, "name", location),
        model_type=model_type,
        size_gb=size_gb,
        is_default=is_default,
        description=_require_string(raw, "description", location, allow_empty=True),
        revision=_optional_string(raw, "revision", location),
        ignore_patterns=_string_list(raw, "ignore_patterns", location),
        pip_dependencies=_string_list(raw, "dependencies", location),
        gated=gated,
        aliases=aliases,
        status=status,
        license=_optional_string(raw, "license", location),
        commercial_use=commercial_use,
        min_vram_gb=min_vram_gb,
        recommended_vram_gb=recommended_vram_gb,
        dtype=dtype,
        pipeline_class=pipeline_class,
        profile=profile,
        use_case=_optional_string(raw, "use_case", location),
        adapter=adapter,
        inference_defaults=dict(inference),
        load_defaults=dict(load),
    )


def _load_catalog_document(catalog_path: Optional[Path] = None) -> Mapping[str, Any]:
    try:
        if catalog_path is not None:
            with Path(catalog_path).open("rb") as catalog_file:
                return tomllib.load(catalog_file)

        catalog_resource = resources.files("hftool.catalog").joinpath("models-v1.toml")
        with catalog_resource.open("rb") as catalog_file:
            return tomllib.load(catalog_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        source = str(catalog_path) if catalog_path else "packaged models-v1.toml"
        raise CatalogError(f"Unable to load model catalog from {source}: {error}") from error


def read_model_catalog(catalog_path: Optional[Path] = None) -> ModelCatalog:
    """Load and validate a local packaged catalog or a test fixture."""
    document = _load_catalog_document(catalog_path)
    catalog_meta = document.get("catalog")
    if not isinstance(catalog_meta, Mapping):
        raise CatalogError("catalog: missing [catalog] table")

    version = catalog_meta.get("version")
    if version != CATALOG_VERSION:
        raise CatalogError(
            f"catalog: unsupported version {version!r}; expected {CATALOG_VERSION}"
        )

    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise CatalogError("catalog: [tasks] must contain at least one task")

    registry: Dict[str, Dict[str, ModelInfo]] = {}
    for task_name, task_entry in raw_tasks.items():
        location = f"tasks.{task_name}"
        if not isinstance(task_entry, Mapping):
            raise CatalogError(f"{location}: task entry must be a table")
        raw_models = task_entry.get("models")
        if not isinstance(raw_models, Mapping) or not raw_models:
            raise CatalogError(f"{location}: models must contain at least one entry")

        models: Dict[str, ModelInfo] = {}
        aliases: Dict[str, str] = {}
        normalized_model_keys = {str(key).lower() for key in raw_models}
        default_count = 0
        for model_key, raw_model in raw_models.items():
            model_location = f"{location}.models.{model_key}"
            info = _parse_model(raw_model, model_location)
            if info.is_default:
                default_count += 1
            for alias in info.aliases:
                normalized_alias = alias.lower()
                if normalized_alias in normalized_model_keys or normalized_alias in aliases:
                    raise CatalogError(
                        f"{model_location}: alias '{alias}' conflicts with another model or alias"
                    )
                aliases[normalized_alias] = model_key
            models[model_key] = info

        if default_count > 1:
            raise CatalogError(f"{location}: only one model may be the default")
        registry[task_name] = models

    runtime = document.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise CatalogError("catalog: [runtime] must be a table")

    return ModelCatalog(
        version=version,
        verified=catalog_meta.get("verified"),
        models=registry,
        runtime=dict(runtime),
    )


_CATALOG = read_model_catalog()
MODEL_REGISTRY: Dict[str, Dict[str, ModelInfo]] = _CATALOG.models


def load_model_catalog(catalog_path: Optional[Path] = None) -> Dict[str, Dict[str, ModelInfo]]:
    """Load and validate a catalog, returning its task-to-model mapping."""
    return read_model_catalog(catalog_path).models


def get_catalog_runtime_config(section: str) -> Dict[str, Any]:
    """Return a copy of one packaged runtime-default section."""
    value = _CATALOG.runtime.get(section, {})
    return dict(value) if isinstance(value, Mapping) else {}


def get_models_for_task(task: str) -> Dict[str, ModelInfo]:
    """Return all canonical catalog models for a task."""
    from hftool.core.registry import TASK_ALIASES

    resolved_task = TASK_ALIASES.get(task, task)
    if resolved_task not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY)
        raise ValueError(f"Unknown task: '{task}'. Available: {', '.join(available)}")
    return MODEL_REGISTRY[resolved_task]


def _find_alias(models: Mapping[str, ModelInfo], requested: str) -> Optional[ModelInfo]:
    requested_lower = requested.lower()
    for info in models.values():
        if any(alias.lower() == requested_lower for alias in info.aliases):
            return info
    return None


def get_model_info(task: str, model: str) -> ModelInfo:
    """Resolve a canonical key, legacy alias, repo ID, or repo short name."""
    models = get_models_for_task(task)
    if model in models:
        return models[model]

    alias_match = _find_alias(models, model)
    if alias_match is not None:
        return alias_match

    model_lower = model.lower()
    for info in models.values():
        if info.repo_id.lower() == model_lower or info.short_name.lower() == model_lower:
            return info

    available = list(models)
    raise ValueError(
        f"Unknown model '{model}' for task '{task}'. Available: {', '.join(available)}"
    )


def get_model_key(task: str, model: str) -> str:
    """Return the canonical catalog key for any supported model reference."""
    models = get_models_for_task(task)
    info = get_model_info(task, model)
    for key, candidate in models.items():
        if candidate is info:
            return key
    raise ValueError(f"Unable to resolve canonical key for '{model}'")


def get_default_model_info(task: str) -> ModelInfo:
    """Return the configured default, or the first entry for legacy tasks."""
    models = get_models_for_task(task)
    for info in models.values():
        if info.is_default:
            return info
    if models:
        return next(iter(models.values()))
    raise ValueError(f"No models configured for task '{task}'")


def find_model_by_repo_id(repo_id: str) -> Optional[tuple]:
    """Find the first catalog model using a repository ID."""
    repo_lower = repo_id.lower()
    for task, models in MODEL_REGISTRY.items():
        for short_name, info in models.items():
            if info.repo_id.lower() == repo_lower:
                return (task, short_name, info)
    return None
