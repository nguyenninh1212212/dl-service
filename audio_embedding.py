# file: audio_embedding.py
import os
# Tắt thông báo log và cảnh báo của TensorFlow/Keras (CỰC KỲ QUAN TRỌNG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Tùy chọn: Fix lỗi OpenMP trên một số hệ thống
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import librosa
import numpy as np
try:
 from tensorflow import layers, models
except Exception:
 # fall back to standalone Keras if tensorflow is not installed or import fails
 from keras import layers, models
from sklearn.preprocessing import normalize
import argparse
import json

# --- CNN embedding model ---
def create_cnn_model(input_shape=(128,128,1), embedding_dim=128):
 # ... (giữ nguyên định nghĩa model)
 model = models.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.GlobalAveragePooling2D(),
  layers.Dense(embedding_dim, activation='relu')
 ])
 return model

model = create_cnn_model()
# model.load_weights('cnn_weights.h5')  # PHẢI BỎ GHI CHÚ VÀ CUNG CẤP TRỌNG SỐ ĐÃ TRAIN

# --- Audio -> mel-spectrogram ---
def get_melspectrogram(file_path, n_mels=128, max_len=128):
 # ... (giữ nguyên hàm)
 y, sr = librosa.load(file_path, sr=22050)
 S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
 S_dB = librosa.power_to_db(S, ref=np.max)
 if S_dB.shape[1] < max_len:
  S_dB = np.pad(S_dB, ((0,0),(0,max_len - S_dB.shape[1])), mode='constant')
 else:
  S_dB = S_dB[:, :max_len]
 return S_dB

# --- Tạo embedding ---
def create_embedding(file_path):
 # ... (giữ nguyên hàm)
 S_dB = get_melspectrogram(file_path)
 S_dB = S_dB[np.newaxis, ..., np.newaxis] # add batch & channel
 emb = model.predict(S_dB)
 emb = normalize(emb)
 return emb[0]

# --- CLI ---
if __name__ == '__main__':
 # ... (giữ nguyên phần CLI)
 parser = argparse.ArgumentParser()
 parser.add_argument('--file', type=str, required=True, help='Audio file path')
 parser.add_argument('--song_id', type=str, required=True, help='Song ID')
 args = parser.parse_args()

 embedding_vector = create_embedding(args.file)
 # Trả về JSON duy nhất:
 output = {
  "song_id": args.song_id,
  "embedding": embedding_vector.tolist()
 }
 print(json.dumps(output))