import gradio as gr
import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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

# 5. 预测函数
def predict_emotion(audio_file_path):
    """使用模型进行预测"""
    try:
        feature = extract_features(audio_file_path)
        predicted_label_idx = model.predict(feature)[0]  # 预测的类别索引
        probabilities = model.predict_proba(feature)[0]  # 预测的概率
        max_prob = max(probabilities)  # 最高概率值

        # 获取类别名称
        predicted_label_text = label_map.get(predicted_label_idx, "Unknown")

        return predicted_label_text, round(max_prob, 4)  # ✅ 直接返回两个值
    except Exception as e:
        return "Error", 0.0  # ✅ 确保始终返回两个值


# 6. 测试（可选）
# if __name__ == "__main__":
#    test_audio = "./CatSound/Resting/cat_coll0118.mp3"  # 替换成你的测试音频文件
#    result = predict_emotion(test_audio)
#    print("🎯 预测结果:", result)
    
    
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="上传猫叫音频文件"),
    outputs=[
        gr.Textbox(label="预测情绪"),
        gr.Number(label="预测概率")
    ],
    title="🐱 猫叫情绪分类",
    description="📢 上传猫叫音频文件（wav 或 mp3），系统将返回预测的情绪类别和概率。",
    theme="compact"
)

# 7️⃣ **启动 Web 界面**
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

