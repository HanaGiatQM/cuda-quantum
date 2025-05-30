# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a self-extractable CUDA-Q archive that can be installed
# on a compatible Linux host system; see also https://makeself.io/.
# A suitable base image can be obtained by building docker/build/assets.Dockerfile.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/installer.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-assets:amd64-cu12-llvm-main
ARG additional_components=none

FROM $base_image AS additional_components_none
RUN echo "No additional components included."
FROM $base_image AS additional_components_assets
COPY assets /assets/
RUN source /cuda-quantum/scripts/configure_build.sh && \
    for folder in `find /assets/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash /cuda-quantum/scripts/migrate_assets.sh -s "$folder" && rm -rf "$folder"; \
    done

# [Installer]
FROM additional_components_${additional_components} AS assets
RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0

## [Content]
RUN source /cuda-quantum/scripts/configure_build.sh && \
    cp /cuda-quantum/scripts/migrate_assets.sh install.sh && \
    chmod a+x install.sh "${CUDAQ_INSTALL_PREFIX}/set_env.sh" && \
    mkdir cuda_quantum_assets && mv install.sh cuda_quantum_assets/install.sh && \
    ## [>CUDAQuantumAssets]
    mkdir -p cuda_quantum_assets/llvm/bin && \
    mkdir -p cuda_quantum_assets/llvm/lib && \
    mkdir -p cuda_quantum_assets/llvm/include && \
    mv "${LLVM_INSTALL_PREFIX}/bin/"clang* cuda_quantum_assets/llvm/bin/ && \
    mv cuda_quantum_assets/llvm/bin/clang-format* "${LLVM_INSTALL_PREFIX}/bin/" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/llc" cuda_quantum_assets/llvm/bin/llc && \
    mv "${LLVM_INSTALL_PREFIX}/bin/lld" cuda_quantum_assets/llvm/bin/lld && \
    mv "${LLVM_INSTALL_PREFIX}/bin/ld.lld" cuda_quantum_assets/llvm/bin/ld.lld && \
    mv "${LLVM_INSTALL_PREFIX}/lib/"* cuda_quantum_assets/llvm/lib/ && \
    mv "${LLVM_INSTALL_PREFIX}/include/"* cuda_quantum_assets/llvm/include/ && \
    mv "${CUTENSOR_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUQUANTUM_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUDAQ_INSTALL_PREFIX}/build_config.xml" cuda_quantum_assets/build_config.xml && \
    mv "${CUDAQ_INSTALL_PREFIX}" cuda_quantum_assets
    ## [<CUDAQuantumAssets]

## [Self-extracting Archive]
ARG deprecation_notice=""
RUN if [ -z "${deprecation_notice}" ] && [ "${CUDA_VERSION#11.}" != "${CUDA_VERSION}" ]; then \
        deprecation_notice="**Note**: Support for CUDA 11 will be removed in future releases. Please update to CUDA 12."; \
    fi && \
    if [ -n "${deprecation_notice}" ]; then \
        notice="deprecation_notice='"${deprecation_notice}"'" && \
        key='expected_key="Enter any key to continue."' && \
        # prompt and wait for confirmation if input and output is a terminal, print to error stream otherwise
        prompt='if [ -t 0 ] && [ -t 1 ]; then while true; do read -p "$deprecation_notice $expected_key" -r choice < /dev/tty; case "$choice" in * ) break;; esac; done; else echo -e "\e[01;31m${deprecation_notice}\e[0m" >&2; fi' && \
        echo "$notice" >> install.sh && echo "$key" >> install.sh && echo "$prompt" >> install.sh; \
    fi
RUN bash /makeself/makeself.sh --gzip --sha256 --license /cuda-quantum/LICENSE \
        /cuda_quantum_assets install_cuda_quantum_cu$(echo ${CUDA_VERSION} | cut -d . -f1).$(uname -m) \
        "CUDA-Q toolkit for heterogeneous quantum-classical workflows" \
        bash install.sh -t /opt/nvidia/cudaq

FROM scratch
COPY --from=assets install_cuda_quantum* . 
COPY --from=assets /cuda-quantum/wheelhouse/* . 
