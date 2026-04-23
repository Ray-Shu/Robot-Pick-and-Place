FROM osrf/ros:noetic-desktop-full
ARG USER=user
ARG DEBIAN_FRONTEND=noninteractive

COPY packages.txt packages.txt

# realsense setup so we can use 405
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [trusted=yes] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list

# install dependencies
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN apt-get update && apt-get install -y \
    $(cat packages.txt) 

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install  --ignore-installed -r requirements.txt

RUN conan config set general.revisions_enabled=1 && \
    conan profile new default --detect > /dev/null && \
    conan profile update settings.compiler.libcxx=libstdc++11 default
RUN rosdep update

# 3D Diffusion
COPY third_party third_party

RUN apt-get update && apt-get install -y \
    libglew-dev \
    patchelf \
    wget

RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install --ignore-installed zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
RUN python3 -m pip install --ignore-installed kaleido plotly

RUN pip install huggingface_hub==0.25.0

# cuda stuff for torch3d
ARG CUDA_MAJOR_VERSION=12
ARG CUDA_VERSION=12.1
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-1
RUN git clone https://github.com/NVIDIA/cub.git /opt/cub

ENV CUB_HOME=/opt/cub
ENV CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
RUN ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"
RUN cd /third_party/pytorch3d_simplified/ && python3 -m pip install .


# Aliases
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "alias die='tmux kill-session'" >> ~/.bashrc