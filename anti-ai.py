import cv2
import numpy as np
from PIL import Image
from stegano import lsb
import torch
import torchattacks
import torchvision.transforms as transforms

def add_adversarial_noise(image_path, eps=0.02):
    """添加对抗性噪声，使AI无法识别"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    attack = torchattacks.FGSM(None, eps=eps)
    perturbed_image = attack(image_tensor)
    return transforms.ToPILImage()(perturbed_image.squeeze())

def add_dct_distortion(image_path):
    """对图片进行DCT变换，扰乱AI的解析"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct[0:10, 0:10] += np.random.uniform(-10, 10, (10, 10))
    gray_encrypted = cv2.idct(dct)
    image[:, :, 0] = gray_encrypted  # 只在蓝色通道加扰动
    return image

def add_noise_watermark(image_path, intensity=50):
    """增加随机噪声水印，使AI难以解析"""
    image = cv2.imread(image_path)
    noise = np.random.randint(0, intensity, image.shape, dtype=np.uint8)
    encrypted_image = cv2.add(image, noise)
    return encrypted_image

def hide_secret_message(image_path, message="AI detection failed!"):
    """使用隐写术在图片中嵌入密文"""
    return lsb.hide(image_path, message)

def encrypt_image(image_path, output_path="encrypted_image.png"):
    """综合加密处理"""
    # 1. 对抗性噪声
    img = add_adversarial_noise(image_path)
    img.save("temp_adv.png")
    
    # 2. DCT 扰动
    img_dct = add_dct_distortion("temp_adv.png")
    cv2.imwrite("temp_dct.png", img_dct)
    
    # 3. 噪声水印
    img_noised = add_noise_watermark("temp_dct.png")
    cv2.imwrite("temp_noised.png", img_noised)
    
    # 4. 隐写术
    final_image = hide_secret_message("temp_noised.png")
    final_image.save(output_path)
    
    print(f"加密完成！保存为 {output_path}")

# 使用示例
encrypt_image("your_image.png", "protected_image.png")