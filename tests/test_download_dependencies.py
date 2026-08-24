"""Tests for dependency auto-install behavior in download helpers."""

from types import SimpleNamespace


def test_install_pip_dependencies_uses_active_python_when_not_pipx_env(monkeypatch):
    """pipx should be skipped if current runtime is not pipx-managed hftool."""
    from hftool.core import download

    commands = []

    def fake_run(cmd, capture_output=True, text=True):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download, "check_dependency_satisfied", lambda dep: False)
    monkeypatch.setattr(download, "_running_in_pipx_hftool_env", lambda: False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pipx" if name == "pipx" else None)
    monkeypatch.setattr("subprocess.run", fake_run)

    ok = download.install_pip_dependencies(["kokoro>=0.8.0"], use_pipx=True)

    assert ok is True
    assert commands, "Expected at least one install command"
    assert all(cmd[0] != "pipx" for cmd in commands)


def test_install_pip_dependencies_returns_false_when_pip_fails(monkeypatch):
    """Dependency installation should fail loudly when pip install fails."""
    from hftool.core import download

    def fake_run(cmd, capture_output=True, text=True):
        return SimpleNamespace(returncode=1, stdout="", stderr="install failed")

    monkeypatch.setattr(download, "check_dependency_satisfied", lambda dep: False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.run", fake_run)

    ok = download.install_pip_dependencies(["kokoro>=0.8.0"], use_pipx=True)

    assert ok is False


def test_download_model_with_progress_raises_if_dependency_install_fails(monkeypatch):
    """Model download should stop if required pip dependencies failed to install."""
    from hftool.core import download

    monkeypatch.setattr(download, "install_pip_dependencies", lambda deps: False)

    try:
        download.download_model_with_progress(
            repo_id="hexgrad/Kokoro-82M",
            size_gb=0.3,
            pip_dependencies=["kokoro>=0.8.0"],
        )
        assert False, "Expected RuntimeError when dependency install fails"
    except RuntimeError as exc:
        assert "Failed to install required dependencies" in str(exc)


def test_install_pip_dependencies_retries_with_break_system_packages(monkeypatch):
    """PEP 668 environments should retry pip with --break-system-packages."""
    from hftool.core import download

    commands = []

    def fake_run(cmd, capture_output=True, text=True):
        commands.append(cmd)
        if "--break-system-packages" in cmd:
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="externally-managed-environment")

    monkeypatch.setattr(download, "check_dependency_satisfied", lambda dep: False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.run", fake_run)

    ok = download.install_pip_dependencies(["kokoro>=0.8.0"], use_pipx=False)

    assert ok is True
    assert len(commands) == 2
    assert "--break-system-packages" in commands[1]
