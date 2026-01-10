FROM python:3.11-slim

WORKDIR /app

# LightGBMに必要なシステムライブラリをインストール
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをコピー
COPY . .

# デフォルトコマンド
CMD ["python", "main.py", "--help"]
