import gradio as gr
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import moviepy.editor as mp
import requests
import json
import os
import time

# API 相关配置
CHAT_URL = "https://api.coze.cn/v3/chat"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"
HEADERS = {
    "Authorization": "Bearer pat_vXDMjQzI98QKO9gUUN5nEU8YHZeu4v7heWMmjH4LFXzjpnA1o5Bmwvr2JCFVqond",
    "Content-Type": "application/json"
}

# 发送对话请求，获取 chat_id 和 conversation_id
def start_chat():
    data = {
        "bot_id": "7486346843451293708",
        "user_id": "123123***",
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {
                "role": "user",
                "content": "早上好",
                "content_type": "text"
            }
        ]
    }

    response = requests.post(CHAT_URL, headers=HEADERS, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        if response_data.get("code") == 0:
            chat_id = response_data["data"].get("id")
            conversation_id = response_data["data"].get("conversation_id")
            return chat_id, conversation_id
    return None, None

# 查询对话消息
def fetch_chat_messages(chat_id, conversation_id):
    params = {"chat_id": chat_id, "conversation_id": conversation_id}
    time.sleep(1)  # 确保消息处理完成

    response = requests.get(MESSAGE_LIST_URL, headers=HEADERS, params=params)
    if response.status_code == 200:
        response_data = response.json()
        if response_data.get("code") == 0:
            messages = response_data.get("data", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    return msg.get("content", "未获取到消息")
    return "未获取到聊天内容"

# 获取并打印首页聊天记录
def print_initial_chat():
    chat_id, conversation_id = start_chat()
    if chat_id and conversation_id:
        assistant_message = fetch_chat_messages(chat_id, conversation_id)
        print(f"assistant: {assistant_message}")

# **打印首页对话消息**
print_initial_chat()

# 加载 YAMNet 预训练模型
local_model_path = "C:\\Users\\羽落\\.cache\\kagglehub\\models\\google\\yamnet\\tensorFlow2\\yamnet\\1"
yamnet_model = hub.load(local_model_path)

# 加载分类模型 & PCA 变换
model = joblib.load("voting_classifier.pkl")
pca = joblib.load("pca_transform.pkl")

# 定义类别映射
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

# 发送对话请求
def send_chat_message(emotion):
    message = f"我现在很{emotion}"
    data = {
        "bot_id": "7486346843451293708",
        "user_id": "123123***",
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {"role": "user", "content": message, "content_type": "text"}
        ]
    }
    response = requests.post(CHAT_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        response_data = response.json()
        if response_data.get("code") == 0:
            chat_id = response_data["data"].get("id")
            conversation_id = response_data["data"].get("conversation_id")
            return fetch_chat_messages(chat_id, conversation_id)
    return "API 调用失败"

# 提取音频特征
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    feature = np.concatenate([
        tf.reduce_mean(embeddings, axis=0).numpy(),
        tf.reduce_max(embeddings, axis=0).numpy()
    ])
    feature = feature.reshape(1, -1)
    return pca.transform(feature)

# MP4 转 MP3
def convert_mp4_to_mp3(mp4_path, output_path="temp_audio.mp3"):
    clip = mp.VideoFileClip(mp4_path).subclip(0, 4)
    clip.audio.write_audiofile(output_path, codec='mp3')
    return output_path

# 预测函数
def predict_emotion(audio_file_path):
    if audio_file_path.endswith(".mp4"):
        mp3_file_path = convert_mp4_to_mp3(audio_file_path)
    else:
        mp3_file_path = audio_file_path
    
    feature = extract_features(mp3_file_path)
    predicted_label_idx = model.predict(feature)[0]
    predicted_label_text = label_map.get(predicted_label_idx, "Unknown")
    api_response = send_chat_message(predicted_label_text)
    
    return predicted_label_text, api_response

# Gradio 接口
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.File(file_count="single", type="filepath", label="上传猫叫音频文件"),
    outputs=[
        gr.Textbox(label="预测情绪"),
        gr.Textbox(label="API 响应")
    ],
    title="🐱 猫叫情绪分类",
    description="📢 上传猫叫音频文件，系统将返回预测的情绪类别，并与聊天机器人互动。",
    theme="compact"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
