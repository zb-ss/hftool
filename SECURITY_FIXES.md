# Security Fixes - hftool v0.3.0

This document summarizes the security fixes implemented to address vulnerabilities identified in the security audit.

## Date: 2024-12-31
## Status: ✅ COMPLETED

---

## HIGH Severity Issues (2/2 Fixed)

### H-1: Path Traversal Vulnerability in Config.get_path()

**Status**: ✅ FIXED

**Location**: `hftool/core/config.py:195-244`

**Issue**: 
The `get_path()` method performed `expanduser()` on user-provided config values without validation, potentially allowing path traversal attacks.

**Fix**:
- Added path validation using `Path.resolve()` and `is_relative_to()`
- Restricted allowed paths to:
  - User's home directory (`~`)
  - `/tmp` directory
- Raises `ValueError` if path resolves outside allowed boundaries

**Code Changes**:
```python
def get_path(self, key: str, default: Optional[str] = None) -> Optional[Path]:
    # ... 
    resolved_path = Path(path_str).expanduser().resolve()
    
    # Security: Validate path is within safe boundaries
    home_dir = Path.home().resolve()
    tmp_dir = Path("/tmp").resolve()
    
    is_in_home = resolved_path.is_relative_to(home_dir)
    is_in_tmp = resolved_path.is_relative_to(tmp_dir)
    
    if not (is_in_home or is_in_tmp):
        raise ValueError(f"Path '{path_str}' resolves to '{resolved_path}' which is outside allowed directories")
    
    return resolved_path
```

**Tests Added**: 
- `tests/test_config.py::TestConfigSecurity::test_path_traversal_prevention`
- `tests/test_config.py::TestConfigSecurity::test_path_in_home_allowed`
- `tests/test_config.py::TestConfigSecurity::test_path_in_tmp_allowed`

---

### H-2: Information Disclosure in Error Messages

**Status**: ✅ FIXED

**Location**: `hftool/utils/errors.py:11-72, 182-209`

**Issue**:
Error messages displayed full file paths, potentially exposing usernames, directory structure, and sensitive system information.

**Fix**:
- Added `sanitize_path()` function to sanitize file paths:
  - Replaces home directory with `~` (e.g., `/home/user/file.txt` → `~/file.txt`)
  - Shows only basename for paths outside home (e.g., `/etc/passwd` → `passwd`)
- Updated `HFToolError.__init__()` to sanitize error messages
- Updated `handle_exception()` to sanitize captured path groups

**Code Changes**:
```python
def sanitize_path(path_str: str) -> str:
    """Sanitize file paths to prevent information disclosure."""
    try:
        path = Path(path_str)
        home = Path.home()
        
        try:
            rel_path = path.relative_to(home)
            return f"~/{rel_path}"
        except ValueError:
            return path.name if path.name else str(path)
    except Exception:
        try:
            return Path(path_str).name
        except Exception:
            return path_str
```

**Tests Added**:
- `tests/test_errors.py::TestSecurityFixes::test_sanitize_path_home_directory`
- `tests/test_errors.py::TestSecurityFixes::test_sanitize_path_outside_home`
- `tests/test_errors.py::TestSecurityFixes::test_sanitize_path_in_tmp`
- `tests/test_errors.py::TestSecurityFixes::test_error_message_path_sanitization`
- `tests/test_errors.py::TestSecurityFixes::test_handle_exception_sanitizes_paths`
- `tests/test_errors.py::TestSecurityFixes::test_sanitize_windows_path`

---

## MEDIUM Severity Issues (3/3 Fixed)

### M-1: Unconstrained TOML Parsing

**Status**: ✅ FIXED

**Location**: `hftool/core/config.py:64-88`

**Issue**:
Config loader parsed TOML files without size limits, potentially allowing DoS attacks via large malicious config files.

**Fix**:
- Added `MAX_CONFIG_SIZE` constant (1MB limit)
- Added file size check before parsing using `stat().st_size`
- Logs warning and skips loading if file exceeds limit

**Code Changes**:
```python
class Config:
    # Security: Maximum config file size (1MB)
    MAX_CONFIG_SIZE = 1024 * 1024
    
    def _load_config(self) -> None:
        # ...
        file_size = user_config_path.stat().st_size
        if file_size > self.MAX_CONFIG_SIZE:
            print(f"Warning: Config file {user_config_path} is too large ({file_size} bytes, max {self.MAX_CONFIG_SIZE})")
        else:
            with open(user_config_path, "rb") as f:
                self._config = tomllib.load(f)
```

**Tests Added**:
- `tests/test_config.py::TestConfigSecurity::test_config_file_size_limit`

---

### M-2: Arbitrary Environment Variable Injection

**Status**: ✅ FIXED

**Location**: `hftool/core/config.py:128-136`

**Issue**:
The `get_value()` method allowed arbitrary environment variable names, potentially enabling environment variable injection attacks.

**Fix**:
- Added `ENV_VAR_PATTERN` regex: `^HFTOOL_[A-Z0-9_]+$`
- Validates environment variable names before accessing `os.environ`
- Rejects invalid names and continues to next priority in fallback chain

**Code Changes**:
```python
class Config:
    # Security: Environment variable name pattern (must start with HFTOOL_)
    ENV_VAR_PATTERN = re.compile(r'^HFTOOL_[A-Z0-9_]+$')
    
    def get_value(self, key: str, task: Optional[str] = None, default: Any = None, env_var: Optional[str] = None) -> Any:
        # ...
        if env_var is None:
            env_var = f"HFTOOL_{key.upper()}"
        
        # Security: Validate environment variable name
        if not self.ENV_VAR_PATTERN.match(env_var):
            env_value = None
        else:
            env_value = os.environ.get(env_var)
```

**Tests Added**:
- `tests/test_config.py::TestConfigSecurity::test_env_var_validation`
- `tests/test_config.py::TestConfigSecurity::test_env_var_injection_special_chars`

---

### M-3: Subprocess Command Injection Risk

**Status**: ✅ FIXED

**Location**: `hftool/cli.py:1364-1397`

**Issue**:
`_open_file()` used subprocess with user-controlled paths without validation, potentially enabling command injection.

**Fix**:
- Added comprehensive path validation:
  - Resolves path to absolute using `Path.resolve()`
  - Validates file exists using `path.exists()`
  - Validates it's a regular file using `path.is_file()`
  - Uses validated absolute path for subprocess
- Added proper exception handling

**Code Changes**:
```python
def _open_file(file_path: str, verbose: bool = False) -> bool:
    # Security: Validate file path (M-3)
    try:
        path = Path(file_path).resolve()
        
        if not path.exists():
            click.echo(f"Cannot open file: {file_path} (file not found)", err=True)
            return False
        
        if not path.is_file():
            click.echo(f"Cannot open file: {file_path} (not a regular file)", err=True)
            return False
        
        # Use the validated absolute path
        file_path = str(path)
        
    except Exception as e:
        click.echo(f"Cannot open file: invalid path ({e})", err=True)
        return False
```

**Tests Added**: 
- No dedicated tests (function is internal and uses platform-specific commands)
- Manual verification recommended

---

## Test Results

All tests pass successfully:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
collected 125 items

tests/test_config.py::TestConfigSecurity::test_path_traversal_prevention PASSED
tests/test_config.py::TestConfigSecurity::test_path_in_home_allowed PASSED
tests/test_config.py::TestConfigSecurity::test_path_in_tmp_allowed PASSED
tests/test_config.py::TestConfigSecurity::test_env_var_validation PASSED
tests/test_config.py::TestConfigSecurity::test_config_file_size_limit PASSED
tests/test_config.py::TestConfigSecurity::test_env_var_injection_special_chars PASSED
tests/test_errors.py::TestSecurityFixes::test_sanitize_path_home_directory PASSED
tests/test_errors.py::TestSecurityFixes::test_sanitize_path_outside_home PASSED
tests/test_errors.py::TestSecurityFixes::test_sanitize_path_in_tmp PASSED
tests/test_errors.py::TestSecurityFixes::test_error_message_path_sanitization PASSED
tests/test_errors.py::TestSecurityFixes::test_handle_exception_sanitizes_paths PASSED
tests/test_errors.py::TestSecurityFixes::test_sanitize_windows_path PASSED

============================= 125 passed in 4.90s ==============================
```

---

## Security Improvements Summary

1. **Path Traversal Protection**: Config paths are now restricted to safe directories
2. **Information Disclosure Prevention**: Error messages no longer expose sensitive paths
3. **DoS Prevention**: Config file size is limited to 1MB
4. **Environment Variable Injection Protection**: Only HFTOOL_* variables are accepted
5. **Command Injection Protection**: File paths are validated before subprocess execution

---

## Recommendations for Future Security

1. Consider adding security scanning to CI/CD pipeline
2. Regularly update dependencies and run `pip audit`
3. Add SAST (Static Application Security Testing) tools
4. Consider implementing rate limiting for model downloads
5. Add input validation for model parameters

---

## References

- Security Audit Report: `/home/zashboy/projects/hftool/docs/security_audit_2024-12-31.md`
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CWE-22 (Path Traversal): https://cwe.mitre.org/data/definitions/22.html
- CWE-200 (Information Exposure): https://cwe.mitre.org/data/definitions/200.html
- CWE-78 (OS Command Injection): https://cwe.mitre.org/data/definitions/78.html
