# hftool Documentation

Welcome to the hftool documentation. This folder contains detailed guides for configuration and setup.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Configuration Guide](configuration.md) | Config file format, locations, and examples |
| [Environment Variables](environment.md) | Complete environment variable reference |
| [GPU Setup Guide](gpu-setup.md) | AMD ROCm, NVIDIA CUDA, Apple MPS, and CPU setup |

## Quick Links

- **[.env.example](../.env.example)** - Copy this to `.env` for your settings
- **[CLAUDE.md](../CLAUDE.md)** - Developer guide for contributing

## Getting Started

1. **Check your system:** `hftool doctor`
2. **Copy example config:** `cp .env.example .env`
3. **Edit your settings:** Customize `.env` or `~/.hftool/config.toml`
4. **Generate something:** `hftool -t t2i -i "A cat" -o cat.png`

## Common Tasks

### Set up HuggingFace token (for gated models)

```bash
# Get your token from: https://huggingface.co/settings/tokens
echo "HF_TOKEN=hf_xxxxxxxxxxxx" >> ~/.bashrc
source ~/.bashrc
```

### Configure default output directory

```bash
mkdir -p ~/.hftool
cat > ~/.hftool/config.toml << 'EOF'
[defaults]
output_dir = "~/AI/outputs"
auto_download = true
EOF
```

### Set up Docker for AMD ROCm

```bash
hftool docker setup
hftool docker build
hftool docker run -- -t t2i -i "A cat" -o ~/Pictures/cat.png
```

## Need Help?

- Run `hftool --help` for CLI usage
- Run `hftool doctor` for system diagnostics
- See [GitHub Issues](https://github.com/zb-ss/hftool/issues) for bug reports
