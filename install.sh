#!/usr/bin/env bash
#
# hftool Docker Installer
#
# One-liner install:
#   curl -fsSL https://raw.githubusercontent.com/zb-ss/hftool/master/install.sh | bash
#
# Or with options:
#   curl -fsSL https://raw.githubusercontent.com/zb-ss/hftool/master/install.sh | bash -s -- --platform rocm
#
# Options:
#   --platform <rocm|cuda|cpu>  Force specific platform (default: auto-detect)
#   --no-build                  Skip building, just install wrapper
#   --install-dir <path>        Installation directory (default: ~/.local/bin)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
PLATFORM=""
NO_BUILD=false
INSTALL_DIR="${HOME}/.local/bin"
HFTOOL_VERSION="0.6.0"
REPO_URL="https://github.com/zb-ss/hftool"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "hftool Docker Installer"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --platform <rocm|cuda|cpu>  Force specific platform (default: auto-detect)"
            echo "  --no-build                  Skip building, just install wrapper"
            echo "  --install-dir <path>        Installation directory (default: ~/.local/bin)"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "  _     __ _              _ "
echo " | |__ / _| |_ ___   ___ | |"
echo " | '_ \\ |_| __/ _ \\ / _ \\| |"
echo " | | | |  _| || (_) | (_) | |"
echo " |_| |_|_|  \\__\\___/ \\___/|_|"
echo -e "${NC}"
echo -e "${BLUE}HuggingFace CLI - Docker Installer${NC}"
echo ""

# Check for Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed.${NC}"
        echo ""
        echo "Please install Docker first:"
        echo "  https://docs.docker.com/get-docker/"
        echo ""
        echo "For Ubuntu/Debian:"
        echo "  curl -fsSL https://get.docker.com | sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  # Log out and back in"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running.${NC}"
        echo ""
        echo "Start Docker with:"
        echo "  sudo systemctl start docker"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Docker is installed and running"
}

# Detect GPU platform
detect_platform() {
    if [[ -n "$PLATFORM" ]]; then
        echo -e "${GREEN}✓${NC} Using specified platform: $PLATFORM"
        return
    fi

    echo -e "${BLUE}Detecting hardware...${NC}"

    # Check for AMD ROCm
    if [[ -e /dev/kfd ]]; then
        PLATFORM="rocm"
        # Try to get GPU name
        if command -v rocm-smi &> /dev/null; then
            GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card series" | cut -d: -f2 | xargs || echo "AMD GPU")
        else
            GPU_NAME="AMD GPU (ROCm device detected)"
        fi
        echo -e "${GREEN}✓${NC} Detected: ${GPU_NAME} (ROCm)"
        return
    fi

    # Check for NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            PLATFORM="cuda"
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
            echo -e "${GREEN}✓${NC} Detected: ${GPU_NAME} (CUDA)"
            return
        fi
    fi

    # CPU fallback
    PLATFORM="cpu"
    echo -e "${YELLOW}!${NC} No GPU detected, using CPU mode"
}

# Build Docker image
build_image() {
    if [[ "$NO_BUILD" == true ]]; then
        echo -e "${YELLOW}Skipping build (--no-build specified)${NC}"
        return
    fi

    IMAGE_NAME="hftool:${PLATFORM}"

    # Check if image already exists
    if docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo -e "${GREEN}✓${NC} Image ${IMAGE_NAME} already exists"
        read -p "Rebuild? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    echo ""
    echo -e "${BLUE}Building Docker image: ${IMAGE_NAME}${NC}"
    echo "This may take 10-15 minutes on first run..."
    echo ""

    # Create temp directory for Dockerfile
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # Download Dockerfile and project files
    echo "Downloading project files..."

    if command -v git &> /dev/null; then
        git clone --depth 1 "$REPO_URL" "$TEMP_DIR/hftool" 2>/dev/null || {
            echo -e "${RED}Failed to clone repository${NC}"
            exit 1
        }
        cd "$TEMP_DIR/hftool"
    else
        # Fallback: download as zip
        curl -fsSL "${REPO_URL}/archive/main.zip" -o "$TEMP_DIR/hftool.zip"
        unzip -q "$TEMP_DIR/hftool.zip" -d "$TEMP_DIR"
        cd "$TEMP_DIR/hftool-main"
    fi

    # Build the image
    docker build \
        -f "docker/Dockerfile.${PLATFORM}" \
        -t "$IMAGE_NAME" \
        --build-arg "HFTOOL_VERSION=${HFTOOL_VERSION}" \
        . || {
        echo -e "${RED}Failed to build Docker image${NC}"
        exit 1
    }

    echo ""
    echo -e "${GREEN}✓${NC} Successfully built ${IMAGE_NAME}"
}

# Create wrapper script
create_wrapper() {
    mkdir -p "$INSTALL_DIR"

    WRAPPER_PATH="${INSTALL_DIR}/hftool"

    echo -e "${BLUE}Creating wrapper script: ${WRAPPER_PATH}${NC}"

    cat > "$WRAPPER_PATH" << 'WRAPPER_EOF'
#!/usr/bin/env bash
#
# hftool Docker wrapper
# Generated by install.sh
#

PLATFORM="__PLATFORM__"
IMAGE="hftool:${PLATFORM}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed" >&2
    exit 1
fi

# Build docker run command based on platform
DOCKER_ARGS=(
    "run" "--rm" "-it"
    "--shm-size" "16g"
    "-v" "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    "-v" "${HFTOOL_CONFIG:-$HOME/.hftool}:/root/.hftool"
    "-v" "$(pwd):/workspace"
    "-w" "/workspace"
    "-e" "HFTOOL_AUTO_DOWNLOAD=1"
)

# GPU-specific flags
case "$PLATFORM" in
    rocm)
        DOCKER_ARGS+=(
            "--device=/dev/kfd"
            "--device=/dev/dri"
            "--security-opt" "seccomp=unconfined"
            "--group-add" "video"
            "--group-add" "render"
        )
        # Pass through AMD-specific env vars
        [[ -n "$HSA_OVERRIDE_GFX_VERSION" ]] && DOCKER_ARGS+=("-e" "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION")
        ;;
    cuda)
        DOCKER_ARGS+=("--gpus" "all")
        [[ -n "$NVIDIA_VISIBLE_DEVICES" ]] && DOCKER_ARGS+=("-e" "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES")
        ;;
esac

# Pass through common env vars
[[ -n "$HF_TOKEN" ]] && DOCKER_ARGS+=("-e" "HF_TOKEN=$HF_TOKEN")
[[ -n "$HFTOOL_DEBUG" ]] && DOCKER_ARGS+=("-e" "HFTOOL_DEBUG=$HFTOOL_DEBUG")

# Custom models directory
if [[ -n "$HFTOOL_MODELS_DIR" ]]; then
    DOCKER_ARGS+=("-v" "$HFTOOL_MODELS_DIR:/models" "-e" "HFTOOL_MODELS_DIR=/models")
fi

# Log file support
if [[ -n "$HFTOOL_LOG_FILE" ]]; then
    LOG_DIR=$(dirname "$HFTOOL_LOG_FILE")
    LOG_NAME=$(basename "$HFTOOL_LOG_FILE")
    [[ -n "$LOG_DIR" ]] && DOCKER_ARGS+=("-v" "$LOG_DIR:/var/log/hftool" "-e" "HFTOOL_LOG_FILE=/var/log/hftool/$LOG_NAME")
fi

# Add image name and pass through all arguments
DOCKER_ARGS+=("$IMAGE")
DOCKER_ARGS+=("$@")

exec docker "${DOCKER_ARGS[@]}"
WRAPPER_EOF

    # Replace placeholder with actual platform
    sed -i "s/__PLATFORM__/${PLATFORM}/g" "$WRAPPER_PATH"

    chmod +x "$WRAPPER_PATH"

    echo -e "${GREEN}✓${NC} Created wrapper script"
}

# Check if install dir is in PATH
check_path() {
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        echo ""
        echo -e "${YELLOW}Note:${NC} ${INSTALL_DIR} is not in your PATH"
        echo ""
        echo "Add it to your shell config:"
        echo ""

        SHELL_NAME=$(basename "$SHELL")
        case "$SHELL_NAME" in
            bash)
                echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
                echo "  source ~/.bashrc"
                ;;
            zsh)
                echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
                echo "  source ~/.zshrc"
                ;;
            fish)
                echo "  fish_add_path ~/.local/bin"
                ;;
            *)
                echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
                ;;
        esac
    fi
}

# Print success message
print_success() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  hftool installed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Quick start:"
    echo ""
    echo "  # Generate an image"
    echo "  hftool -t t2i -i \"A cat in space\" -o cat.png"
    echo ""
    echo "  # Transcribe audio"
    echo "  hftool -t asr -i audio.wav -o transcript.txt"
    echo ""
    echo "  # Interactive mode"
    echo "  hftool -I"
    echo ""
    echo "Environment variables (optional):"
    echo "  HF_TOKEN                    - HuggingFace token for gated models"
    echo "  HFTOOL_MODELS_DIR           - Custom model storage directory"
    echo "  HSA_OVERRIDE_GFX_VERSION    - AMD GPU architecture (e.g., 11.0.0)"
    echo ""
    echo "Documentation: ${REPO_URL}"
    echo ""
}

# Main installation flow
main() {
    check_docker
    detect_platform
    build_image
    create_wrapper
    check_path
    print_success
}

main
