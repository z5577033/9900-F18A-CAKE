# 从基础镜像开始
FROM databricksruntime/standard:16.4-LTS

# 设置工作目录
WORKDIR /app

# --- 最终更新 ---
# 更新包列表，并同时安装所有系统依赖：
# python3-venv: 用于创建 Python 虚拟环境
# git: 用于从 git 仓库下载代码
# build-essential: C/C++ 编译器等基础构建工具
# pkg-config: 帮助编译器找到其他库
# default-libmysqlclient-dev: MySQL 客户端的开发文件
# r-base-dev: R 语言本身及其开发工具
# libtirpc-dev: rpy2 编译时需要的网络通信库 (新添加！)
RUN apt-get update && apt-get install -y \
    python3-venv \
    git \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    r-base-dev \
    libtirpc-dev
# --- 更新结束 ---

# 创建虚拟环境
RUN python3 -m venv /opt/venv

# 激活虚拟环境
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件
COPY requirements.txt .

# 在虚拟环境里安装依赖包
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目剩余文件
COPY . .