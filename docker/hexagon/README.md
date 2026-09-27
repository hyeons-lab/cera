# Qualcomm Hexagon SDK Docker Environment for macOS

This setup provides an x86_64 Linux environment on macOS (Apple Silicon or Intel) to run Qualcomm Package Manager (QPM3), download/install the Qualcomm Hexagon SDK, and cross-compile Hexagon DSP (`.so` skel) binaries and FastRPC host libraries.

---

## Why Docker on macOS?

The Qualcomm Hexagon toolchain (`hexagon-clang`, `qaic`, `hexagon-sim`) and Qualcomm Package Manager CLI (`qpm-cli`) are only distributed as Linux x86_64 and Windows ELF/PE binaries. They do not run natively on macOS Darwin.

By running an x86_64 container on macOS with Docker (using Rosetta 2 emulation), you get near-native execution speed for compilation without requiring a separate Linux PC.

---

## Fast Path: Pre-built Snapdragon Container (Zero Setup)

If you only need to compile Hexagon DSP / FastRPC code and do not need a custom QPM installation, Qualcomm maintains pre-built multi-arch images on GitHub Container Registry:

```bash
docker run --platform linux/amd64 -it --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash
```

This container comes with the Hexagon SDK, Android NDK, CMake, and cross-compilers pre-installed.

---

## Custom Hexagon SDK Container Setup

Follow these steps to build your own container with Qualcomm Package Manager (QPM3) and the official Hexagon SDK.

### 1. Prerequisites on macOS

Ensure Rosetta emulation is enabled for x86_64 containers:
- **Docker Desktop**: Settings -> General -> Check "Use Rosetta for x86/amd64 emulation on Apple Silicon".
- **Colima**: Start Colima with Rosetta support:
  ```bash
  colima start --arch aarch64 --rosetta
  ```

### 2. Download QPM3 (Qualcomm Package Manager)

1. Open your browser on macOS and navigate to [qpm.qualcomm.com](https://qpm.qualcomm.com/).
2. Sign in with your Qualcomm OneID account.
3. Click the **Tools** tab, search for **Qualcomm Package Manager 3**, and download the Linux `.deb` installer (for example, `QualcommPackageManager3.x.x.x.Linux-x86.deb`).
4. Save or copy the `.deb` file into this directory (`docker/hexagon/`).

### 3. Build the Docker Image

Run from the `docker/hexagon/` folder:

```bash
docker build --platform linux/amd64 -t hexagon-sdk-dev:latest .
```

If you placed the `QualcommPackageManager*.deb` file in the folder, the build automatically installs `qpm-cli`.

### 4. Start the Container

Using Docker Compose:

```bash
docker compose run --rm hexagon-dev
```

Or using standard `docker run`:

```bash
docker run --platform linux/amd64 -it --rm \
  -v "$(pwd)/../..":/workspace \
  -v hexagon-qcom-cache:/opt/qcom \
  -w /workspace \
  hexagon-sdk-dev:latest bash
```

The named volume `hexagon-qcom-cache` ensures your downloaded Hexagon SDK persists across container restarts.

### 5. Install the Hexagon SDK inside the Container

Inside the container shell:

```bash
# 1. Log in with your Qualcomm OneID credentials
qpm-cli --login <your-email@example.com>

# 2. View available Hexagon SDK packages
qpm-cli --product-list

# 3. Activate the license (e.g. Hexagon SDK 6.0 or 5.4)
qpm-cli --license-activate hexagonsdk6.0

# 4. Install the SDK to /opt/qcom/Hexagon_SDK
qpm-cli --install hexagonsdk6.0 --path /opt/qcom/Hexagon_SDK
```

### 6. Verify the Toolchain

Once installed, reload the environment:

```bash
source /opt/qcom/Hexagon_SDK/setup_sdk_env.source
hexagon-clang --version
qaic --version
```
