# syntax=docker/dockerfile:1.7-labs
FROM databricksruntime/standard:16.4-LTS

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) 装系统依赖 + venv 组件（确保 ensurepip 存在）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git python3-venv python3.12-venv \
    pkg-config default-libmysqlclient-dev \
    r-base r-base-dev \
    libtirpc-dev \
    libicu-dev libbz2-dev liblzma-dev libpcre2-dev zlib1g-dev \
    libreadline-dev libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) 先拷 requirements 和本地包，利用构建缓存
COPY pyproject.toml .
COPY README.md .
COPY src ./src

COPY requirements.txt .

# 3) 创建 venv 并把它放进 PATH
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4) 安装依赖
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# 5) 再拷其余源码
COPY . .