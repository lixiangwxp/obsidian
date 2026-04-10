- **要稳定部署 / 给别人复现**：Dockerfile
- **Docker**：一个「容器化」工具，能把软件、环境、依赖打包成一个独立的「容器」，在任何机器上都能一模一样地运行，解决了「我本地能跑，你那跑不了」的问题。
- **Ubuntu 容器**：一个独立、隔离的 Ubuntu 系统环境，和你电脑的主系统完全分开，用完可以直接删掉，不会污染主机。

![[Pasted image 20260408135854.png]]




```
# 1. 查看CUDA Toolkit版本（容器内的，不是宿主机的）
nvcc -V
# 重点：release 后面的版本，比如 12.1、12.4

# 2. 查看PyTorch版本、CUDA编译版本、GPU状态
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA编译版本:', torch.version.cuda); print('GPU可用:', torch.cuda.is_available())"

# 3. 查看Python版本
python --version

# 4. 查看Ubuntu系统版本
cat /etc/os-release
# 重点：VERSION_CODENAME，比如 jammy（对应Ubuntu 22.04）





export OMP_NUM_THREADS=1
conda activate nanodrone_pip

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

import pytorch3d
print("pytorch3d import ok")
PY


conda install -n base -c conda-forge -y conda-pack
conda pack -n nanodrone_pip -o /root/autodl-tmp/nanodrone_pip.tar.gz



如果你的项目目录是仓库目录，再把代码也打包一份。假设项目在 /root/nanodrone-sysid-benchmark：
cd /root/nanodrone-sysid-benchmark
tar czf /root/nanodrone_project.tar.gz .


git rev-parse HEAD


## **把包从 AutoDL 下载到你本地 4060 机器**

scp -P 50226 root@connect.westc.seetacloud.com:/root/nanodrone_pip.tar.gz .
scp -P 50226 root@connect.westc.seetacloud.com:/root/nanodrone_project.tar.gz .



## **先把你本地 4060 的 Docker GPU 环境装好**

如果你本地 4060 是 **Windows**，就用 **Docker Desktop + WSL2**。
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04 nvidia-smi

docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04 nvidia-smi

先把项目和环境包放到一个目录里，比如：
mkdir nanodrone_docker
cd nanodrone_docker
tar xzf ../nanodrone_project.tar.gz
cp ../nanodrone_pip.tar.gz .


再建一个 .dockerignore，尽量让构建上下文小一点，这是 Docker 官方的构建最佳实践。
.git
wandb
__pycache__
*.pyc
*.pyo
*.pt
*.pth
tf-logs
autodl-tmp
data


然后写 Dockerfile。你现在环境是 Ubuntu 20.04，PyTorch 是 cu124，所以直接对齐到 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04 这个官方镜像标签就行；这个 tag 在 Docker Hub 是存在的。



FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV PATH=/opt/conda_env/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY nanodrone_pip.tar.gz /tmp/

RUN mkdir -p /opt/conda_env && \
    tar -xzf /tmp/nanodrone_pip.tar.gz -C /opt/conda_env && \
    /opt/conda_env/bin/conda-unpack && \
    rm -f /tmp/nanodrone_pip.tar.gz

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]


构建：

docker build -t nanodrone:cu124 .

Docker 构建本质上就是读取 Dockerfile 里的指令来自动组装镜像



## **在 4060 上先做最小验证，再正式跑**

先别一上来跑训练，先验证镜像里的关键包：
docker run --rm -it --gpus all nanodrone:cu124 python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import pytorch3d
print("pytorch3d ok")
PY


如果这一步通过，再进交互式 shell：
docker run --rm -it --gpus all nanodrone:cu124 bash


如果你的数据集不想打进镜像，建议挂载目录而不是 COPY 进去：

docker run --rm -it --gpus all \

  -v /your/local/data:/workspace/data \

  nanodrone:cu124 bash


## **六、你当前记录里，最值得注意的坑**

  

最主要就这三个：

requirements.txt 里这行：
packaging @ file:///home/task_.../work
这行在别的机器上大概率会直接失败，最好删掉，或者改成普通版本号。
pytorch3d 是目前最大的风险点。不是说它现在一定坏，而是**它不是一个“看到 requirements 就能稳定重建”的官方标准组合**。所以我才建议你用 conda-pack 打当前能跑的环境，而不是在 4060 上重新 pip install -r requirements.txt。


OMP_NUM_THREADS 要设成整数，不然每次启动都有这个 libgomp 警告。
  
