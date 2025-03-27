import os
import base64
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import uvicorn
import paho.mqtt.client as mqtt
from fastapi import FastAPI, UploadFile, File

# 初始化 FastAPI
app = FastAPI()

# 加载 YAMNet 预训练模型
local_model_path = "/root/.cache/kagglehub/models/google/yamnet/tensorFlow2/yamnet/1"
yamnet_model = hub.load(local_model_path)

# 加载情绪分类模型 & PCA
model = joblib.load("voting_classifier.pkl")
pca = joblib.load("pca_transform.pkl")

# 类别映射
label_map = {
    0: "Angry", 1: "Defence", 2: "Fighting", 3: "Happy", 4: "HuntingMind",
    5: "Mating", 6: "MotherCall", 7: "Paining", 8: "Resting", 9: "Warning"
}

# **MQTT 物联网参数**
MQTT_BROKER = "你的物联网平台MQTT服务器地址"
MQTT_PORT = 1883
MQTT_USERNAME = "你的MQTT用户名"
MQTT_PASSWORD = "你的MQTT密码"
MQTT_TOPIC_SUB = "device/audio"  # 设备上传音频的主题
MQTT_TOPIC_PUB = "device/result"  # 返回情绪识别结果的主题

# **MQTT 客户端**
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


def extract_features(audio_data):
    """提取音频特征"""
    waveform = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)

    # 提取 2048 维特征
    feature = np.concatenate([
        tf.reduce_mean(embeddings, axis=0).numpy(),
        tf.reduce_max(embeddings, axis=0).numpy()
    ])

    feature = feature.reshape(1, -1)  # 确保形状正确
    feature = pca.transform(feature)  # 降维到 256 维
    return feature


def predict_emotion(audio_data):
    """使用模型预测情绪"""
    feature = extract_features(audio_data)
    predicted_label_idx = model.predict(feature)[0]
    predicted_label_text = label_map.get(predicted_label_idx, "Unknown")
    return predicted_label_text


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """接受上传的音频文件，进行情绪分析"""
    audio_data, _ = librosa.load(file.file, sr=16000)
    emotion = predict_emotion(audio_data)
    return {"emotion": emotion}


def on_mqtt_message(client, userdata, msg):
    """处理 MQTT 消息（接收音频并预测情绪）"""
    try:
        audio_base64 = msg.payload.decode("utf-8")  # 获取 Base64 编码的音频
        audio_bytes = base64.b64decode(audio_base64)  # 解码成二进制数据
        
        # 保存临时音频文件
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 加载音频并预测
        audio_data, _ = librosa.load(audio_path, sr=16000)
        emotion = predict_emotion(audio_data)

        # 发送预测结果
        mqtt_client.publish(MQTT_TOPIC_PUB, emotion)
        print(f"✅ 预测结果已发送: {emotion}")

    except Exception as e:
        print(f"❌ 处理 MQTT 消息时出错: {e}")


# **连接 MQTT**
def start_mqtt():
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.subscribe(MQTT_TOPIC_SUB)
    mqtt_client.loop_start()
    print("✅ MQTT 监听中...")


# 启动 API & MQTT
if __name__ == "__main__":
    start_mqtt()
    uvicorn.run(app, host="0.0.0.0", port=8000)
