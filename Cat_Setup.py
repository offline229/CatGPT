import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import joblib
import io
import json

# 加载预训练模型（本例中使用 YAMNet，从本地路径加载）
local_yamnet_path = './yamnet'
yamnet_model = hub.load(local_yamnet_path)
print("YAMNet 模型加载完成。")

# 加载保存的 VotingClassifier 模型
voting_clf = joblib.load("./voting_classifier.pkl")
print("VotingClassifier 模型加载完成。")

# 假设你的类别映射（在训练时使用过），例如：
label_map = {0: "Angry", 1: "Defence", 2: "Fighting", 3: "Happy", 4: "HuntingMind", 5: "Mating", 6: "MotherCall", 7: "Paining", 8: "Resting", 9: "Warning"}
# 为了后续部署，最好将该字典也保存成 json 文件，部署时加载：
# with open("label_map.json", "w") as f:
#    json.dump(label_map, f)

def preprocess_audio(file_bytes, target_sr=16000, target_duration=5.0):
    """
    从字节流加载音频，调整采样率和时长（若需要补零或截断）
    """
    audio_stream = io.BytesIO(file_bytes)
    waveform, sr = librosa.load(audio_stream, sr=target_sr)
    desired_length = int(target_sr * target_duration)
    if len(waveform) < desired_length:
        waveform = np.pad(waveform, (0, desired_length - len(waveform)))
    else:
        waveform = waveform[:desired_length]
    return waveform

def extract_features_from_yamnet(waveform):
    """
    利用 YAMNet 提取音频特征：
      - 不扩展 batch 维度，因为模型要求输入一维 Tensor
      - 模型输出 embeddings 形状为 (num_frames, 1024)
      - 对所有帧取平均得到 (1024,) 的固定向量
    """
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
    embedding_mean = tf.reduce_mean(embeddings, axis=0)
    embedding_mean = tf.reshape(embedding_mean, (1024,))
    return embedding_mean.numpy().reshape(1, -1)  # 输出 shape=(1,1024)

def custom_predict_proba(voting_clf, X):
    """
    对于 hard voting 的 VotingClassifier，逐个调用其子模型的 predict_proba，
    如果不支持 predict_proba，则将其预测结果转换为 one-hot 编码，
    然后计算平均概率作为最终预测概率。
    """
    probas = []
    # 使用 named_estimators_.items() 来遍历 (name, estimator) 对
    for name, clf in voting_clf.named_estimators_.items():
        try:
            p = clf.predict_proba(X)
        except AttributeError:
            # 不支持 predict_proba 时，使用 predict 转换为 one-hot 编码
            preds = clf.predict(X)
            p = np.zeros((X.shape[0], len(voting_clf.classes_)))
            for i, pred in enumerate(preds):
                class_idx = list(voting_clf.classes_).index(pred)
                p[i, class_idx] = 1.0
        probas.append(p)
    # 计算所有子模型概率的平均值
    avg_probas = np.mean(probas, axis=0)
    return avg_probas


def predict_emotion(file_bytes):
    """
    预测函数：
      1. 从上传的音频文件字节中加载并预处理音频；
      2. 利用预训练的 YAMNet 模型提取特征（1024 维）；
      3. 用已训练的 VotingClassifier（hard voting）预测类别；
      4. 通过自定义的 predict_proba 计算平均概率，
         并返回概率最高的类别及其概率。
    """
    waveform = preprocess_audio(file_bytes, target_sr=16000, target_duration=5.0)
    features = extract_features_from_yamnet(waveform)
    # 用自定义函数计算预测概率
    probas = custom_predict_proba(voting_clf, features)[0]
    pred_idx = int(np.argmax(probas))
    return {
        "predicted_label": label_map.get(pred_idx, "Unknown"),
        "probability": float(probas[pred_idx])
    }

# 测试预测函数
# with open('./Cat_Test_Sound_01.wav', 'rb') as f:
#     file_bytes = f.read()

# result = predict_emotion(file_bytes)
# print("预测结果：", result)

import gradio as gr

def predict_emotion_ui(audio_file_path):
    try:
        # audio_file_path 是文件路径，打开并读取文件字节
        with open(audio_file_path, "rb") as f:
            file_bytes = f.read()
        result = predict_emotion(file_bytes)
        return result["predicted_label"], result["probability"]
    except Exception as e:
        return "Error", str(e)

iface = gr.Interface(
    fn=predict_emotion_ui,
    inputs=gr.Audio(type="filepath", label="上传猫叫音频文件"),
    outputs=[
        gr.Textbox(label="预测情绪"),
        gr.Number(label="预测概率")
    ],
    title="猫叫情绪分类",
    description="上传猫叫音频文件（wav 或 mp3），系统将返回预测的情绪类别和概率。"
)

iface.launch()


