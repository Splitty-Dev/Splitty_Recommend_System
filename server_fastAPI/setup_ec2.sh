#!/bin/bash
# EC2 Ubuntu 22.04 LTS 초기 설정 스크립트

echo "=== AWS EC2 추천 시스템 환경 설정 ==="

# 시스템 업데이트
sudo apt-get update -y
sudo apt-get upgrade -y

# Python 3.10+ 설치
sudo apt-get install -y python3 python3-pip python3-venv

# 필수 패키지 설치
sudo apt-get install -y git curl wget htop

# GPU 지원 (선택사항 - GPU 인스턴스인 경우)
if lspci | grep -i nvidia > /dev/null; then
    echo "GPU 감지됨. CUDA 설치 중..."
    
    # CUDA 설치
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-2
    
    # cuDNN 설치 (선택사항)
    echo "CUDA 설치 완료. 재부팅 필요할 수 있습니다."
else
    echo "CPU 전용 환경으로 설정됩니다."
fi

# Python 가상환경 생성
python3 -m venv /home/ubuntu/splitty_env
source /home/ubuntu/splitty_env/bin/activate

# 프로젝트 클론
cd /home/ubuntu
git clone https://github.com/Splitty-Dev/Splitty_Recommend_System.git
cd Splitty_Recommend_System/server_fastAPI

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch GPU 버전 설치 (GPU 인스턴스인 경우)
if lspci | grep -i nvidia > /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== 설정 완료 ==="
echo "가상환경 활성화: source /home/ubuntu/splitty_env/bin/activate"
echo "프로젝트 디렉토리: cd /home/ubuntu/Splitty_Recommend_System/server_fastAPI"