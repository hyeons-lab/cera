# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
FROM --platform=linux/amd64 ubuntu:${UBUNTU_VERSION}

LABEL maintainer="cera"
LABEL description="Qualcomm Hexagon SDK and FastRPC cross-compilation environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install base build tools, Python, and system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    ninja-build \
    make \
    gcc \
    g++ \
    git \
    curl \
    wget \
    ca-certificates \
    unzip \
    xz-utils \
    tar \
    file \
    sudo \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    # QPM3 GUI and headless runtime dependencies
    bc \
    xdg-utils \
    policykit-1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libxcb1 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    # Hexagon simulator and debugger utilities
    libc6-dev-i386 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install libncurses5 and libtinfo5 compatibility libraries
# Hexagon LLVM tools (hexagon-clang, qaic, hexagon-sim) link against libncurses.so.5 and libtinfo.so.5
RUN if [ "${UBUNTU_VERSION}" = "22.04" ]; then \
        echo "deb http://security.ubuntu.com/ubuntu focal-security main universe" > /etc/apt/sources.list.d/focal-security.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends libncurses5 libtinfo5 && \
        rm /etc/apt/sources.list.d/focal-security.list && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && \
        apt-get install -y --no-install-recommends libncurses5 libtinfo5 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Optional: Install Android NDK (r30 LTS) for FastRPC Android host cross-compilation
ARG INSTALL_NDK=true
ARG NDK_VERSION=r30
ENV ANDROID_NDK_ROOT=/opt/android-ndk
ENV ANDROID_NDK_HOME=/opt/android-ndk
RUN if [ "${INSTALL_NDK}" = "true" ]; then \
        mkdir -p /opt/android-ndk && \
        wget -q --show-progress https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip -O /tmp/ndk.zip && \
        unzip -q /tmp/ndk.zip -d /tmp && \
        mv /tmp/android-ndk-${NDK_VERSION}/* /opt/android-ndk/ && \
        rm -rf /tmp/android-ndk-${NDK_VERSION} /tmp/ndk.zip; \
    fi

# Set up Qualcomm installation directories
ENV QUALCOMM_ROOT=/opt/qcom
ENV HEXAGON_SDK_ROOT=/opt/qcom/Hexagon_SDK
ENV PATH=/opt/qcom/QPM3/bin:/opt/qcom/qpm3/bin:/opt/qcom/Hexagon_SDK/tools/scripts:${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}

RUN mkdir -p /opt/qcom /workspace

# Copy local QPM deb or Hexagon SDK installers if placed in the build context
COPY .dockerkeep* *QualcommPackageManager*.deb* /tmp/installers/
RUN if ls /tmp/installers/*.deb 1> /dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends /tmp/installers/*.deb && \
        rm -rf /tmp/installers /var/lib/apt/lists/*; \
    fi

# Set up entrypoint and workspace
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
