FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV JULIA_VERSION=1.9.2
WORKDIR /app

# ===============================
# Install system dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    wget git curl sudo build-essential \
    python3 python3-pip \
    tmux aria2 psmisc libglib2.0-0 libxrender1 libsm6 libxext6 && \
    apt-get clean

# ===============================
# Install Julia
# ===============================
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar -xvzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    mv julia-${JULIA_VERSION} /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# ===============================
# Clone the toolbox and setup Julia env
# ===============================
RUN git clone https://github.com/intelligent-control-lab/ModelVerification.jl.git /app/ModelVerification

# Julia environment setup
RUN julia -e 'using Pkg; \
    Pkg.add("CUDA"); \
    Pkg.add("cuDNN"); \
    Pkg.add("ONNX"); \
    import CUDA; CUDA.set_runtime_version!(v"12.2"); \
    Pkg.develop(path="/app/ModelVerification"); \
    Pkg.activate("/app/ModelVerification"); \
    Pkg.develop(path="/app/ModelVerification/onnx_parser/NaiveNASflux"); \
    Pkg.develop(path="/app/ModelVerification/onnx_parser/ONNXNaiveNASflux"); \
    Pkg.add(["LazySets", "PyCall", "CSV", "DataFrames", "CUDA"]); \
    import CUDA; CUDA.set_runtime_version!(v"12.2"); '

# ===============================
# Python packages
# ===============================
RUN pip3 install -r /app/ModelVerification/vnncomp_scripts/NNet/test_requirements.txt

