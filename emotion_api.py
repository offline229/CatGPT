from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import gradio as gr
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# 1. 加载 YAMNet 预训练模型
local_model_path = "C:\\Users\\羽落\\.cache\\kagglehub\\models\\google\\yamnet\\tensorFlow2\\yamnet\\1"
yamnet_model = hub.load(local_model_path)

# 2. 加载分类模型 & PCA 变换
model = joblib.load("voting_classifier.pkl")
pca = joblib.load("pca_transform.pkl")

# 3. 定义类别映射
label_map = {
    0: "Angry",
    1: "Defence",
    2: "Fighting",
    3: "Happy",
    4: "HuntingMind",
    5: "Mating",
    6: "MotherCall",
    7: "Paining",
    8: "Resting",
    9: "Warning",
}

# 4. 特征提取函数
def extract_features(file_path):
    """提取音频特征"""
    audio, sr = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    # **保持和训练时一致的 2048 维特征**
    feature = np.concatenate([tf.reduce_mean(embeddings, axis=0).numpy(), tf.reduce_max(embeddings, axis=0).numpy()])
    feature = feature.reshape(1, -1)  # 确保形状为 (1, n)
    feature = pca.transform(feature)  # 转换为 256 维
    return feature

# 5. 预测函数
def predict_emotion(audio_file_path):
    """使用模型进行预测"""
    try:
        feature = extract_features(audio_file_path)
        predicted_label_idx = model.predict(feature)[0]
        probabilities = model.predict_proba(feature)[0]
        max_prob = max(probabilities)
        predicted_label_text = label_map.get(predicted_label_idx, "Unknown")
        return predicted_label_text, round(max_prob, 4)
    except Exception as e:
        return "Error", 0.0

# FastAPI应用
app = FastAPI()

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    """处理从云平台接收到的音频文件并返回情绪预测"""
    try:
        # 将音频文件保存为临时文件
        audio_path = f"temp_{audio_file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        # 预测情绪
        emotion, probability = predict_emotion(audio_path)

        return {"emotion": emotion, "probability": probability}
    except Exception as e:
        return {"error": str(e)}

# 启动 FastAPI 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
