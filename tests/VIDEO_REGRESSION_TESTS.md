# Video Output Regression Tests

## Overview

This document describes the regression tests added to prevent video generation from producing empty files due to improper handling of numpy arrays with batch dimensions.

## Bug Fixed

Video generation was outputting empty files because numpy arrays with batch dimension `(1, N, H, W, C)` weren't being properly unwrapped before saving.

## Test Files Created/Modified

### 1. `/home/zashboy/projects/hftool/tests/test_io.py`

Added `TestVideoOutputBugFix` class with 18 comprehensive tests.

#### Tests for `_convert_frame_to_pil()`:
- `test_convert_frame_to_pil_normal_3d_frame` - Converts 3D arrays (H, W, C)
- `test_convert_frame_to_pil_uint8_frame` - Handles uint8 pixel values
- `test_convert_frame_to_pil_channels_first` - Converts channels-first format (C, H, W)
- `test_convert_frame_to_pil_single_batch_dimension` - Unwraps single frame with batch dim (1, H, W, C)
- `test_convert_frame_to_pil_rejects_5d_array` - Raises ValueError for 5D arrays
- `test_convert_frame_to_pil_rejects_large_4d_array` - Raises ValueError for 4D arrays with many frames
- `test_convert_frame_to_pil_handles_pil_image` - Pass-through for PIL Images
- `test_convert_frame_to_pil_converts_rgba_to_rgb` - Converts RGBA to RGB
- `test_convert_frame_to_pil_handles_torch_tensor` - Converts torch tensors to numpy

#### Tests for `save_video()`:
- `test_save_video_with_5d_numpy_array` - Unwraps 5D numpy array (1, N, H, W, C)
- `test_save_video_with_4d_numpy_array` - Converts 4D numpy array (N, H, W, C)
- `test_save_video_with_list_of_pil_images` - Backward compatibility with PIL image lists
- `test_save_video_with_torch_tensor` - Converts and unwraps torch tensors
- `test_save_video_raises_on_empty_frames` - Raises ValueError for empty frame lists
- `test_save_video_raises_on_none_frames` - Raises ValueError for None frames
- `test_save_video_respects_fps_parameter` - Uses custom FPS parameter
- `test_save_video_respects_codec_parameter` - Uses custom codec parameter
- `test_save_video_raises_on_ffmpeg_failure` - Raises RuntimeError when ffmpeg fails

### 2. `/home/zashboy/projects/hftool/tests/test_text_to_video.py`

New test file with 12 comprehensive tests for the text-to-video task handler.

#### Tests for `run_inference()` frame unwrapping:
- `test_run_inference_unwraps_5d_numpy_array` - Unwraps 5D batch dimension
- `test_run_inference_unwraps_4d_numpy_array` - Handles 4D arrays
- `test_run_inference_converts_torch_tensor` - Converts torch tensors
- `test_run_inference_handles_nested_list` - Unwraps nested list [[frame1, frame2]]
- `test_run_inference_handles_images_attribute` - Falls back to result.images
- `test_run_inference_handles_direct_array_return` - Handles direct array returns
- `test_run_inference_preserves_list_of_pil_images` - Backward compatibility
- `test_run_inference_passes_kwargs_to_pipeline` - Passes inference parameters
- `test_run_inference_converts_seed_to_generator` - Converts seed to generator
- `test_run_inference_uses_model_defaults` - Merges model-specific defaults

#### Integration tests for `save_output()`:
- `test_save_output_calls_save_video` - Calls save_video with correct parameters
- `test_save_output_uses_default_fps` - Uses default FPS of 24

## Test Coverage

The tests verify:

1. **Array Shape Handling**: 5D, 4D, and 3D numpy arrays
2. **Data Type Conversion**: float32, float64, uint8
3. **Format Conversion**: Channels-first to channels-last
4. **Framework Support**: Both numpy and torch tensors
5. **Backward Compatibility**: List of PIL Images still works
6. **Error Handling**: Proper errors for empty/invalid inputs
7. **Parameter Passing**: FPS, codec, and other parameters respected
8. **Integration**: End-to-end flow from inference to save

## Running the Tests

```bash
# Run all video regression tests
python3 -m pytest tests/test_io.py::TestVideoOutputBugFix tests/test_text_to_video.py -v

# Run only output handler tests
python3 -m pytest tests/test_io.py::TestVideoOutputBugFix -v

# Run only task handler tests
python3 -m pytest tests/test_text_to_video.py -v
```

## Key Features

- **Fast**: All tests use small 8x8 pixel frames and mock ffmpeg calls
- **Isolated**: No GPU/models required, no external dependencies
- **Comprehensive**: Cover all code paths related to the bug fix
- **Maintainable**: Clear test names and documentation
- **Reliable**: Use pytest's importorskip for optional dependencies

## Test Results

All 30 tests pass:
- 18 tests in `test_io.py::TestVideoOutputBugFix`
- 12 tests in `test_text_to_video.py`

Total runtime: ~0.8 seconds
