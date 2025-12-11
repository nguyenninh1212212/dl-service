# 1️⃣ Base image TensorFlow slim (có sẵn TF, numpy, keras)
FROM tensorflow/tensorflow:2.14.0

# 2️⃣ Env
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# 3️⃣ Install system libraries cần thiết cho librosa / ffmpeg / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Set working directory
WORKDIR /app

# 5️⃣ Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# ... (Dòng 1 đến 5 giữ nguyên)

# 6️⃣ Upgrade pip + install packages + handle blinker conflict
# Loại bỏ lệnh riêng cho protobuf>=6.0.0. Để requirements.txt quyết định phiên bản.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir python-dotenv \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt
# Thêm --ignore-installed để vượt qua lỗi blinker

# 7️⃣ Copy Python code vào container
COPY server.py audio_embedding.py audio_embed_pb2*.py ./

# 8️⃣ Expose gRPC port
EXPOSE 50053

# 9️⃣ Command để chạy server
CMD ["python", "-u", "server.py"]
