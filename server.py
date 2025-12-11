import os
import tempfile
import requests
import validators
from concurrent import futures
import grpc
import audio_embed_pb2
import audio_embed_pb2_grpc
from audio_embedding import create_embedding
from pymongo import MongoClient
from functools import lru_cache
from dotenv import load_dotenv
from threading import Thread
from flask import Flask, jsonify # <-- THÊM THƯ VIỆN FLASK

load_dotenv()

# Lấy cổng từ biến môi trường của Render, mặc định 50053 cho local test
RENDER_PORT = os.getenv("PORT", "50053") 
GRPC_ADDRESS = f"0.0.0.0:{RENDER_PORT}"

# ====================================================================
# I. KHỞI TẠO DỊCH VỤ PHỤ TRỢ (MongoDB, Cache, Logic gRPC)
# ====================================================================

# MongoDB   
mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo["ai_music"]
emb_col = db["songembeddings"]

# (Các hàm validate_audio_url, cached_download, AudioEmbedServicer giữ nguyên)
# ... (Phần code này được lược bỏ để giữ ngắn gọn)

# ---- Validate URL ----
def validate_audio_url(url):
    print("Validating URL:", url)
    if not validators.url(url):
        raise ValueError("URL không hợp lệ")

    head = requests.head(url, timeout=5)
    if head.status_code != 200:
        raise ValueError("Không truy cập được URL")

    content_type = head.headers.get("Content-Type", "")
    if not content_type.startswith("audio") or content_type.startswith("video"):
        raise ValueError("URL không phải file audio")

    return True

# ---- Cache: lưu 50 file gần nhất ----
@lru_cache(maxsize=50)
def cached_download(url):
    # streaming download → không load full vào RAM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)

        tmp.flush()
        return tmp.name 

class AudioEmbedServicer(audio_embed_pb2_grpc.AudioEmbedServicer):
    def Embed(self, request, context):
        url = request.audioUrl
        song_id = request.songId
        print(f"Received Embed request for song ID: {song_id}, URL: {url}")
        try:
            validate_audio_url(url)
            file_path = cached_download(url)

            emb = create_embedding(file_path)
            embedding_list = [float(x) for x in emb]

            # Lưu vào MongoDB
            emb_col.insert_one({
                "songId": song_id,
                "embedding": embedding_list,
            })

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return audio_embed_pb2.EmbedResponse()

        return audio_embed_pb2.EmbedResponse(
            songId=song_id,
            embedding=embedding_list
        )


# ====================================================================
# II. DỊCH VỤ HTTP FLASK (HEALTH CHECK)
# ====================================================================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint này được Render sử dụng để kiểm tra cổng có mở không."""
    # Bạn có thể thêm logic kiểm tra DB/gRPC ở đây nếu cần
    return jsonify({"status": "healthy", "service": "gRPC AudioEmbed"}), 200

def run_flask_app():
    """Chạy Flask trên cùng cổng nhưng trên một thread riêng."""
    # Host trên 0.0.0.0 và cổng của Render. debug=False là quan trọng trong production.
    print(f"Flask HTTP health check running on {GRPC_ADDRESS}")
    app.run(host='0.0.0.0', port=int(RENDER_PORT), debug=False)


# ====================================================================
# III. KHỞI TẠO CÁC DỊCH VỤ CÙNG LÚC
# ====================================================================

def serve():
    # 1. Khởi tạo gRPC Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    audio_embed_pb2_grpc.add_AudioEmbedServicer_to_server(
        AudioEmbedServicer(),
        server,
    )

    # Lắng nghe gRPC trên cổng của Render
    server.add_insecure_port(GRPC_ADDRESS)
    print(f"Python AudioEmbed gRPC server running at {GRPC_ADDRESS}")
    server.start()
    
    # 2. Chạy Flask HTTP Server trên một thread riêng biệt
    # Flask và gRPC có thể chia sẻ cùng cổng vì chúng sử dụng các giao thức khác nhau (HTTP/1.1 vs HTTP/2)
    # Tuy nhiên, an toàn nhất là chạy Flask trên một Thread riêng
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True # Cho phép thread chính kết thúc thread này
    flask_thread.start()

    # 3. Giữ thread chính không kết thúc (cho gRPC)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()