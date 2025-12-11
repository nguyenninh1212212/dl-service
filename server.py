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

load_dotenv()

# MongoDB   
mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo["ai_music"]
emb_col = db["songembeddings"]

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
        return tmp.name   # trả về path file tạm

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


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    audio_embed_pb2_grpc.add_AudioEmbedServicer_to_server(
        AudioEmbedServicer(),
        server,
    )

    server.add_insecure_port("0.0.0.0:50053")
    print("Python AudioEmbed gRPC server running at :50053")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
