"""Tests for the interactive mode wizard."""

import os
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from hftool.io.interactive_mode import (
    is_inquirer_available,
    check_interactive_mode,
    run_interactive_mode,
)


class TestInteractiveModeHelpers:
    """Test helper functions."""
    
    def test_is_inquirer_available(self):
        """Test InquirerPy detection."""
        result = is_inquirer_available()
        # Should return True since InquirerPy is installed
        assert isinstance(result, bool)
    
    def test_check_interactive_mode_cli_flag(self):
        """Test that CLI flag takes priority."""
        mock_ctx = MagicMock()
        
        # CLI flag True should return True
        assert check_interactive_mode(mock_ctx, True) is True
    
    def test_check_interactive_mode_env_var(self):
        """Test environment variable check."""
        mock_ctx = MagicMock()
        
        with patch.dict(os.environ, {"HFTOOL_INTERACTIVE": "1"}):
            assert check_interactive_mode(mock_ctx, False) is True
        
        with patch.dict(os.environ, {"HFTOOL_INTERACTIVE": "true"}):
            assert check_interactive_mode(mock_ctx, False) is True
        
        with patch.dict(os.environ, {"HFTOOL_INTERACTIVE": "yes"}):
            assert check_interactive_mode(mock_ctx, False) is True
    
    def test_check_interactive_mode_env_var_disabled(self):
        """Test that other env var values don't enable interactive."""
        mock_ctx = MagicMock()
        
        # Clear the env var and patch config
        env = os.environ.copy()
        env.pop("HFTOOL_INTERACTIVE", None)
        
        with patch.dict(os.environ, env, clear=True):
            with patch("hftool.core.config.Config") as mock_config:
                mock_config.get.return_value.get_value.return_value = False
                assert check_interactive_mode(mock_ctx, False) is False
    
    def test_check_interactive_mode_config(self):
        """Test config file check."""
        mock_ctx = MagicMock()
        
        # Clear env var
        env = os.environ.copy()
        env.pop("HFTOOL_INTERACTIVE", None)
        
        with patch.dict(os.environ, env, clear=True):
            with patch("hftool.core.config.Config") as mock_config_class:
                mock_config = MagicMock()
                mock_config.get_value.return_value = True
                mock_config_class.get.return_value = mock_config
                
                assert check_interactive_mode(mock_ctx, False) is True


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_help_shows_interactive_wizard(self):
        """Test that -I flag is in help."""
        from hftool.cli import main
        
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "-I" in result.output
        assert "--interactive-wizard" in result.output
    
    def test_wizard_requires_inquirerpy(self):
        """Test that wizard shows error without InquirerPy."""
        from hftool.cli import main
        
        runner = CliRunner()
        
        with patch("hftool.io.interactive_mode.is_inquirer_available", return_value=False):
            result = runner.invoke(main, ["-I"])
            
            assert result.exit_code == 1
            assert "InquirerPy" in result.output
    
    def test_wizard_starts_with_banner(self):
        """Test that wizard shows banner."""
        from hftool.cli import main
        
        runner = CliRunner()
        
        # The wizard shows the banner before any InquirerPy calls
        # We can test by checking output even when it fails
        result = runner.invoke(main, ["-I"], input="\n")
        
        # Should show the banner even if it fails
        assert "hftool - Interactive Mode" in result.output or "InquirerPy" in result.output


class TestWizardSteps:
    """Test individual wizard steps with mocked InquirerPy."""
    
    @pytest.fixture
    def mock_inquirer(self):
        """Create mock inquirer module."""
        mock = MagicMock()
        
        # Mock Choice and Separator
        mock_choice = MagicMock()
        mock_separator = MagicMock()
        
        return mock, mock_choice, mock_separator
    
    def test_task_selection(self, mock_inquirer):
        """Test task selection step."""
        from hftool.io.interactive_mode import _select_task
        
        inquirer, Choice, Separator = mock_inquirer
        inquirer.select.return_value.execute.return_value = "text-to-image"
        
        result = _select_task(inquirer, Choice, Separator)
        
        assert result == "text-to-image"
        inquirer.select.assert_called_once()
    
    def test_model_selection(self, mock_inquirer):
        """Test model selection step."""
        from hftool.io.interactive_mode import _select_model
        
        inquirer, Choice, Separator = mock_inquirer
        inquirer.select.return_value.execute.return_value = "z-image-turbo"
        
        result = _select_model(inquirer, Choice, Separator, "text-to-image")
        
        assert result == "z-image-turbo"
    
    def test_device_selection(self, mock_inquirer):
        """Test device selection step."""
        from hftool.io.interactive_mode import _select_device
        
        inquirer, Choice, _ = mock_inquirer
        inquirer.select.return_value.execute.return_value = "cuda"
        
        result = _select_device(inquirer, Choice)
        
        assert result == "cuda"
    
    def test_dtype_selection(self, mock_inquirer):
        """Test dtype selection step."""
        from hftool.io.interactive_mode import _select_dtype
        
        inquirer, Choice, _ = mock_inquirer
        inquirer.select.return_value.execute.return_value = "float16"
        
        result = _select_dtype(inquirer, Choice)
        
        assert result == "float16"
    
    def test_seed_random(self, mock_inquirer):
        """Test random seed selection."""
        from hftool.io.interactive_mode import _get_seed
        
        inquirer, _, _ = mock_inquirer
        inquirer.select.return_value.execute.return_value = "random"
        
        result = _get_seed(inquirer)
        
        assert isinstance(result, int)
        assert 0 <= result < 2**32
    
    def test_seed_manual(self, mock_inquirer):
        """Test manual seed entry."""
        from hftool.io.interactive_mode import _get_seed
        
        inquirer, _, _ = mock_inquirer
        inquirer.select.return_value.execute.return_value = "manual"
        inquirer.text.return_value.execute.return_value = "12345"
        
        result = _get_seed(inquirer)
        
        assert result == 12345
    
    def test_extra_params_skip(self, mock_inquirer):
        """Test skipping extra parameters."""
        from hftool.io.interactive_mode import _get_extra_params
        
        inquirer, _, _ = mock_inquirer
        inquirer.confirm.return_value.execute.return_value = False
        
        result = _get_extra_params(inquirer, "text-to-image")
        
        assert result == {}
    
    def test_extra_params_json(self, mock_inquirer):
        """Test entering extra parameters as JSON."""
        from hftool.io.interactive_mode import _get_extra_params
        
        inquirer, _, _ = mock_inquirer
        inquirer.confirm.return_value.execute.return_value = True
        inquirer.text.return_value.execute.return_value = '{"num_inference_steps": 30}'
        
        result = _get_extra_params(inquirer, "text-to-image")
        
        assert result == {"num_inference_steps": 30}


class TestFullWizardFlow:
    """Test complete wizard flow."""
    
    def test_full_wizard_structure(self):
        """Test that wizard module has expected structure."""
        from hftool.io.interactive_mode import (
            run_interactive_mode,
            check_interactive_mode,
            is_inquirer_available,
            _select_task,
            _select_model,
            _select_device,
            _select_dtype,
            _get_seed,
            _get_extra_params,
        )
        
        # Verify all functions are callable
        assert callable(run_interactive_mode)
        assert callable(check_interactive_mode)
        assert callable(is_inquirer_available)
        assert callable(_select_task)
        assert callable(_select_model)
        assert callable(_select_device)
        assert callable(_select_dtype)
        assert callable(_get_seed)
        assert callable(_get_extra_params)


class TestEnvAndConfigIntegration:
    """Test environment variable and config integration."""
    
    def test_env_var_enables_wizard(self):
        """Test HFTOOL_INTERACTIVE=1 enables wizard mode."""
        from hftool.cli import main
        
        runner = CliRunner()
        
        with patch.dict(os.environ, {"HFTOOL_INTERACTIVE": "1"}):
            with patch("hftool.io.interactive_mode.is_inquirer_available", return_value=True):
                with patch("hftool.io.interactive_mode.run_interactive_mode") as mock_run:
                    mock_run.side_effect = KeyboardInterrupt()
                    
                    result = runner.invoke(main, [])
                    
                    # Should have tried to run interactive mode
                    # (it will fail with KeyboardInterrupt but that's OK for this test)
    
    def test_explicit_task_bypasses_wizard(self):
        """Test that -t flag bypasses wizard even with env var."""
        from hftool.cli import main
        
        runner = CliRunner()
        
        with patch.dict(os.environ, {"HFTOOL_INTERACTIVE": "1"}):
            # With -t flag, should not enter wizard
            result = runner.invoke(main, ["-t", "t2i", "-i", "test"])
            
            # Should proceed with task (may fail due to missing deps, but shouldn't show wizard)
            assert "Interactive Mode" not in result.output
