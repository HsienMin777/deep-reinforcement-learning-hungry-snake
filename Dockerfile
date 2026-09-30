FROM python:3.12-slim

WORKDIR /app

# 即時輸出 log (不緩衝)，並隱藏 pygame 的歡迎訊息
ENV PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1

# 先裝 CPU 版 torch，避免下載數 GB 的 CUDA 版本
# pygame 的 wheel 已內含 SDL，且容器內以 --no-render 執行，不需要額外的系統套件
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY App/ .

# 容器內沒有顯示器：不開視窗、全速訓練，結束後把權重存回 /app
CMD ["python", "app.py", "--no-render"]
