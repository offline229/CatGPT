import gradio as gr
import time
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import moviepy.editor as mp
import requests
import json
import os


# API 相关配置
CHAT_URL = "https://api.coze.cn/v3/chat"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"
HEADERS = {
    "Authorization": "Bearer pat_vXDMjQzI98QKO9gUUN5nEU8YHZeu4v7heWMmjH4LFXzjpnA1o5Bmwvr2JCFVqond",
    "Content-Type": "application/json"
}

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
    message = f"作为一只情绪目前是{emotion}的猫，给我符合你情绪的回应。"
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
            conversation_id = response_data.get("data", {}).get("conversation_id")
            chat_id = response_data.get("data", {}).get("id")
            if conversation_id and chat_id:
                return fetch_chat_messages(conversation_id, chat_id)
    return "API 调用失败"

# 查询对话消息
def fetch_chat_messages(conversation_id, chat_id):
    params = {"conversation_id": conversation_id, "chat_id": chat_id}

    # 第一次请求，等待3秒
    time.sleep(3)

    response = requests.get(MESSAGE_LIST_URL, headers=HEADERS, params=params)

    if response.status_code == 200:
        response_data = response.json()
        print("查询消息返回:", json.dumps(response_data, indent=2, ensure_ascii=False))

        # 解析 API 返回的消息数据
        if response_data.get("code") == 0:
            messages = response_data.get("data", [])  # 这里修正，data 直接是列表

            print("\n对话消息记录：")
            # 打印并返回对话内容
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "（无内容）")
                print(f"{role}: {content}")

            # 过滤掉类型不是 "answer" 的消息
            filtered_messages = [
                msg["content"] for msg in messages 
                if "content" in msg and msg.get("type") == "answer"
            ]

            # 返回处理后的消息内容
            return "\n".join(filtered_messages) if filtered_messages else "没有找到类型为 'answer' 的消息"
        else:
            print(f"查询失败，错误信息: {response_data.get('msg')}")
    else:
        print(f"请求失败，HTTP 状态码: {response.status_code}")
    
    return "未获取到聊天内容"

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
