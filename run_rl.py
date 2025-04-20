import base64
import io
import re
import uuid
import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 标准化
])

# 清洗 Base64 字符串
def clean_base64(base64_string):
    """
    清洗 Base64 字符串，移除前缀、换行符、空格，并确保 padding 正确。
    
    Args:
        base64_string (str): 输入的 Base64 字符串
    
    Returns:
        str: 清洗后的 Base64 字符串
    """
    try:
        # 移除常见前缀
        base64_string = re.sub(r'^data:image/(jpeg|gif);base64,', '', base64_string)
        # 移除换行符、空格、制表符
        base64_string = base64_string.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
        # 确保长度是 4 的倍数，填充 '='
        padding = len(base64_string) % 4
        if padding:
            base64_string += '=' * (4 - padding)
        # 验证 Base64 格式
        base64.b64decode(base64_string, validate=True)
        return base64_string
    except Exception as e:
        logging.error(f"Invalid Base64 string after cleaning: {e}")
        raise ValueError(f"Cannot clean Base64 string: {e}")

# 加载模型
def load_model(model_path, device):
    try:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# 预测单帧
def predict_frame(frame, model, transform, device):
    model.eval()
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frame)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][1].item()  # 正类置信度
        _, pred = torch.max(output, 1)
    return pred.item() == 1, confidence

# 预测 Base64 编码的图像
def predict_image_base64(base64_string, identifier, model, transform, device):
    try:
        # 清洗 Base64 字符串
        cleaned_base64 = clean_base64(base64_string)
        # 解码 Base64
        image_data = base64.b64decode(cleaned_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # 检查是否为 GIF
        if image.format.lower() == 'gif':
            max_confidence = 0.0
            is_positive = False
            for frame in ImageSequence.Iterator(image):
                frame = frame.convert('RGB')
                frame_positive, confidence = predict_frame(frame, model, transform, device)
                max_confidence = max(max_confidence, confidence)
                if frame_positive:
                    is_positive = True
            image.close()
            return is_positive, max_confidence
        else:
            image = image.convert('RGB')
            result = predict_frame(image, model, transform, device)
            image.close()
            return result
    except Exception as e:
        logging.error(f"Error predicting image {identifier}: {e}")
        return False, 0.0

# 微调模型
def fine_tune_model(model, mistakes, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    dataset = [(img, label) for img, label in mistakes]
    if not dataset:
        logging.info("No mistakes to fine-tune on")
        return False

    logging.info(f"Fine-tuning on {len(dataset)} mistake samples")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for epoch in range(3):  # 少量 epoch 微调
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Fine-tune Epoch [{epoch+1}/3], Loss: {running_loss/len(loader):.4f}")
    return True

# 外部调用接口
def predict_base64_with_label(base64_string, identifier=None, true_label=True, 
                             model_path='./best_nailong.pth', 
                             device='cuda' if torch.cuda.is_available() else 'cpu', 
                             save_csv=True, output_csv='predictions.csv'):
    """
    预测 Base64 编码的图像，并支持真实标签进行微调。
    
    Args:
        base64_string (str): Base64 编码的图片字符串
        identifier (str, optional): 图片标识（默认生成 UUID）
        true_label (bool): 图片真实标签（默认 True）
        model_path (str): 模型文件路径（默认 './best_nailong.pth'）
        device (str): 设备（默认 'cuda' 或 'cpu'）
        save_csv (bool): 是否保存结果到 CSV（默认 True）
        output_csv (str): CSV 文件名（默认 'predictions.csv'）
    
    Returns:
        dict: {'identifier': identifier, 'true_label': 'True'/'False', 
               'prediction': 'True'/'False', 'confidence': confidence}
    """
    try:
        # 生成 identifier（如果未提供）
        identifier = identifier if identifier is not None else str(uuid.uuid4())
        
        # 加载模型
        model = load_model(model_path, device)
        
        # 预测
        is_positive, confidence = predict_image_base64(base64_string, identifier, model, test_transform, device)
        
        # 真实标签（转换为 0 或 1）
        true_label_int = 1 if true_label else 0
        pred_label_int = 1 if is_positive else 0
        
        # 控制台输出
        print(f"Identifier: {identifier}, True Label = {'True' if true_label else 'False'}, "
              f"Prediction = {'True' if is_positive else 'False'}, Confidence = {confidence:.4f}")
        
        # 结果
        result = {
            'identifier': identifier,
            'true_label': 'True' if true_label else 'False',
            'prediction': 'True' if is_positive else 'False',
            'confidence': confidence
        }
        
        # 检查是否需要微调
        accuracy = 1.0 if pred_label_int == true_label_int else 0.0
        mistakes = []
        if accuracy < 1.0:
            # 将图像重新加载为张量用于微调
            cleaned_base64 = clean_base64(base64_string)
            image_data = base64.b64decode(cleaned_base64)
            image = Image.open(io.BytesIO(image_data)).convertovaniRGB()
            image_tensor = test_transform(image).to(device)
            mistakes.append((image_tensor.cpu(), true_label_int))
            image.close()
        
        # 如果准确率低于 0.9，触发微调
        if accuracy < 0.9 and mistakes:
            logging.info(f"Accuracy {accuracy:.4f} is below 0.9, fine-tuning model...")
            if fine_tune_model(model, mistakes, device):
                logging.info("Fine-tuning completed, saving model...")
                torch.save({'state_dict': model.state_dict(), 'accuracy': accuracy}, 
                          './nailong_finetuned.pth')
                logging.info("Saved fine-tuned model to ./nailong_finetuned.pth")
        
        # 保存到 CSV
        if save_csv:
            results_df = pd.DataFrame([{
                'file': identifier,
                'true_label': result['true_label'],
                'prediction': result['prediction'],
                'confidence': result['confidence']
            }])
            if os.path.exists(output_csv):
                results_df.to_csv(output_csv, mode='a', header=False, index=False)
            else:
                results_df.to_csv(output_csv, mode='w', header=True, index=False)
            logging.info(f"Results appended to {output_csv}")
        
        return result
    except Exception as e:
        logging.error(f"Error in predict_base64_with_label: {e}")
        raise

# 测试：从 ./input_true 和 ./input_false 文件夹读取图像，使用文件名作为 identifier
def main():
    input_dirs = [
        ('./input_true', True),
        ('./input_false', False)
    ]
    
    for input_dir, true_label in input_dirs:
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory {input_dir} does not exist")
            continue

        # 读取 JPG 和 GIF 文件
        image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.gif'))
        if not image_files:
            logging.warning(f"No images found in {input_dir}")
            continue

        for image_path in image_files:
            try:
                # 使用文件名（含扩展名）作为 identifier
                filename = os.path.basename(image_path)
                with open(image_path, 'rb') as f:
                    base64_string = base64.b64encode(f.read()).decode('utf-8')
                result = predict_base64_with_label(base64_string, identifier=filename, true_label=true_label)
                logging.info(f"Prediction result for {image_path}: {result}")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    main()
    
'''
from run_rl import predict_base64_with_label

base64_string = "data:image/jpeg;base64,/9j/4AAQSkZJRg==\n"
result = predict_base64_with_label(base64_string, true_label=True)  # 不提供 identifier
print(result)
'''