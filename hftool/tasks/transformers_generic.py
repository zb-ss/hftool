"""Generic transformers pipeline task handler.

Fallback handler for tasks that use standard transformers pipelines.
"""

from typing import Any, Dict, Optional
from hftool.tasks.base import BaseTask


class TransformersGenericTask(BaseTask):
    """Handler for generic transformers pipeline tasks.
    
    Supports any task that works with the transformers pipeline() function:
    - text-generation
    - text-classification
    - question-answering
    - summarization
    - translation
    - image-classification
    - object-detection
    - image-to-text
    - etc.
    """
    
    def __init__(
        self,
        task_name: str,
        device: str = "auto",
        dtype: Optional[str] = None
    ):
        super().__init__(device, dtype)
        self.task_name = task_name
        self._model_name: Optional[str] = None
    
    def load_pipeline(self, model: str, **kwargs) -> Any:
        """Load a transformers pipeline.
        
        Args:
            model: HuggingFace model name or local path
            **kwargs: Additional arguments for pipeline()
        
        Returns:
            Loaded transformers pipeline
        """
        from hftool.utils.deps import check_dependencies
        check_dependencies(["transformers", "torch"])
        
        from hftool.core.device import detect_device, get_optimal_dtype
        from transformers import pipeline
        import torch
        
        self._model_name = model
        
        # Determine device
        device = self.device if self.device != "auto" else detect_device()
        
        # Determine dtype
        if self.dtype:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.dtype)
        else:
            dtype = None  # Let pipeline choose
        
        # Load pipeline
        pipe_kwargs = {
            "model": model,
            "device": device if device != "mps" else None,
            **kwargs
        }
        
        if dtype:
            pipe_kwargs["torch_dtype"] = dtype
        
        pipe = pipeline(self.task_name, **pipe_kwargs)
        
        return pipe
    
    def run_inference(self, pipeline: Any, input_data: Any, **kwargs) -> Any:
        """Run inference using the pipeline.
        
        Args:
            pipeline: Loaded transformers pipeline
            input_data: Input data (text, image path, etc.)
            **kwargs: Additional inference arguments
        
        Returns:
            Pipeline output (varies by task)
        """
        result = pipeline(input_data, **kwargs)
        return result
    
    def save_output(self, result: Any, output_path: str, **kwargs) -> str:
        """Save the output to a file.
        
        Args:
            result: Pipeline output
            output_path: Path to save output
            **kwargs: Additional save arguments
        
        Returns:
            Path to saved file
        """
        import json
        
        # Determine how to save based on result type
        if isinstance(result, str):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
        elif isinstance(result, (list, dict)):
            # Try to extract text if available
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    # Common patterns
                    if "generated_text" in first:
                        text = "\n".join(r.get("generated_text", "") for r in result)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        return output_path
                    elif "summary_text" in first:
                        text = "\n".join(r.get("summary_text", "") for r in result)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        return output_path
                    elif "translation_text" in first:
                        text = "\n".join(r.get("translation_text", "") for r in result)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        return output_path
                    elif "answer" in first:
                        text = first.get("answer", "")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        return output_path
            
            # Fallback: save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        else:
            # Fallback: convert to string
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(str(result))
        
        return output_path


def create_task(
    task_name: str,
    device: str = "auto",
    dtype: Optional[str] = None
) -> TransformersGenericTask:
    """Factory function to create a TransformersGenericTask.
    
    Args:
        task_name: Name of the transformers pipeline task
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        dtype: Data type ("bfloat16", "float16", "float32")
    
    Returns:
        Configured TransformersGenericTask instance
    """
    return TransformersGenericTask(task_name=task_name, device=device, dtype=dtype)
