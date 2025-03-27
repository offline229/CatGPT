import gradio as gr
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import moviepy.editor as mp

# 1. 加载 YAMNet 预训练模型
print("🚀 加载 YAMNet 预训练模型...")
local_model_path = "C:\\Users\\羽落\\.cache\\kagglehub\\models\\google\\yamnet\\tensorFlow2\\yamnet\\1"
yamnet_model = hub.load(local_model_path)
print("✅ YAMNet 加载完成！")

# 2. 加载分类模型 & PCA 变换
print("🚀 加载猫叫分类模型 & PCA...")
model = joblib.load("voting_classifier.pkl")  # 加载已训练好的分类器
pca = joblib.load("pca_transform.pkl")  # 加载 PCA 变换
print("✅ 模型 & PCA 加载成功！")

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
    feature = np.concatenate([ 
        tf.reduce_mean(embeddings, axis=0).numpy(),
        tf.reduce_max(embeddings, axis=0).numpy()
    ])

    feature = feature.reshape(1, -1)  # 确保形状为 (1, n)
    
    # **应用 PCA 降维**
    feature = pca.transform(feature)  # 转换为 256 维
    return feature

# 5. MP4 转 MP3 函数 (提取前 4 秒的音频)
def convert_mp4_to_mp3(mp4_path, output_path="temp_audio.mp3"):
    """将 MP4 转换为 MP3 并截取前 4 秒"""
    try:
        clip = mp.VideoFileClip(mp4_path)
        duration = clip.duration  # 获取视频的总时长

        # 如果视频时长小于 4 秒，则截取整个视频的音频
        end_time = min(4, duration)

        # 截取音频
        clip = clip.subclip(0, end_time)
        clip.audio.write_audiofile(output_path, codec='mp3')
        return output_path
    except Exception as e:
        print(f"❌ 错误发生：{e}")
        return None

# 6. 预测函数
def predict_emotion(audio_file_path):
    """使用模型进行预测"""
    try:
        # 检查文件类型并转换 MP4 为 MP3
        if audio_file_path.endswith(".mp4"):
            print("📽️ 检测到 MP4 文件，开始转换为 MP3...")
            mp3_file_path = convert_mp4_to_mp3(audio_file_path)
            if mp3_file_path is None:
                return "Error: 音频转换失败", 0.0
        else:
            mp3_file_path = audio_file_path
        
        feature = extract_features(mp3_file_path)
        predicted_label_idx = model.predict(feature)[0]  # 预测的类别索引
        probabilities = model.predict_proba(feature)[0]  # 预测的概率
        max_prob = max(probabilities)  # 最高概率值

        # 获取类别名称
        predicted_label_text = label_map.get(predicted_label_idx, "Unknown")

        return predicted_label_text, round(max_prob, 4)  # ✅ 直接返回两个值
    except Exception as e:
        print(f"❌ 错误发生：{e}")
        return "Error", 0.0  # ✅ 确保始终返回两个值

# 7. Gradio 接口
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.File(file_count="single", type="filepath", label="上传猫叫音频文件 (支持 .mp3, .wav, .mp4)"),
    outputs=[
        gr.Textbox(label="预测情绪"),
        gr.Number(label="预测概率")
    ],
    title="🐱 猫叫情绪分类",
    description="📢 上传猫叫音频文件（wav, mp3 或 mp4），系统将返回预测的情绪类别和概率。",
    theme="compact"
)

# 8. 启动 Web 界面
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
