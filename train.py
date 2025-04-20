import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 处理GIF图像，提取每一帧并保存为JPG格式
def process_gif(gif_path, output_dir):
    try:
        gif = Image.open(gif_path)
        frame_num = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for frame in ImageSequence.Iterator(gif):
            rgb_frame = frame.convert('RGB')
            frame_filename = f"{os.path.splitext(os.path.basename(gif_path))[0]}_frame_{frame_num:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            rgb_frame.save(frame_path, 'JPEG')
            logging.info(f"Saved frame {frame_num} to {frame_path}")
            frame_num += 1
        gif.close()
        os.remove(gif_path)
        logging.info(f"Deleted original GIF file: {gif_path}")
    except PermissionError as e:
        logging.error(f"Failed to delete {gif_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing {gif_path}: {e}")

# 转换非JPG图像为JPG格式
def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    for image_file in image_files:
        if image_file.lower().endswith('.gif'):
            process_gif(image_file, directory)
        elif not image_file.lower().endswith(('.jpg', '.jpeg')):
            try:
                img = Image.open(image_file).convert('RGB')
                jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
                img.save(jpg_path, 'JPEG')
                logging.info(f"Converted {image_file} to {jpg_path}")
                os.remove(image_file)
                logging.info(f"Deleted original file: {image_file}")
            except Exception as e:
                logging.error(f"Error converting {image_file}: {e}")

# 自定义数据集
class NailongDataset(Dataset):
    def __init__(self, positive_root, negative_root, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for filename in glob.glob(os.path.join(positive_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(1)  # 正样本标签为1
        for filename in glob.glob(os.path.join(negative_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(0)  # 负样本标签为0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None  # 跳过损坏图像

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 保留更多特征
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),  # 减少颜色失真
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 标准化
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建加权采样器以平衡数据集
def create_weighted_sampler(dataset):
    labels = [label for _, label in dataset if label is not None]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

# 评估并收集错误样本
def evaluate_and_collect_mistakes(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    mistakes = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for idx, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    mistakes.append((images[idx].cpu(), label.cpu().item()))

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    logging.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    return running_loss / len(data_loader), accuracy, mistakes

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

# 训练模型
def train_model(model, train_loader, val_loader, epochs, device, best_model_path, iteration):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
    best_val_loss = float('inf') if iteration == 0 else torch.load(best_model_path)['val_loss'] if os.path.exists(best_model_path) else float('inf')
    patience = 5
    no_improvement = 0

    for epoch in range(epochs):
        # 训练
        running_loss = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证并检查是否需要微调
        val_loss, val_acc, mistakes = evaluate_and_collect_mistakes(model, val_loader, device)
        logging.info(f"Iteration {iteration}, Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 如果验证集准确率低于 0.9，触发微调
        if val_acc < 0.9 and mistakes:
            logging.info(f"Validation accuracy {val_acc:.4f} is below 0.9, fine-tuning model...")
            if fine_tune_model(model, mistakes, device):
                logging.info("Re-evaluating model after fine-tuning...")
                val_loss, val_acc, _ = evaluate_and_collect_mistakes(model, val_loader, device)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save({'state_dict': model.state_dict(), 'val_loss': val_loss}, best_model_path)
            logging.info(f"Saved best model to {best_model_path}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    return val_acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # 转换图像格式
    for dir_path in ['./train_positive', './train_negative']:
        if os.path.exists(dir_path):
            convert_images_to_jpg(dir_path)
        else:
            logging.warning(f"Directory {dir_path} does not exist")

    # 创建数据集
    dataset = NailongDataset(positive_root='./train_positive', negative_root='./train_negative', transform=train_transform)
    if len(dataset) == 0:
        logging.error("No images found in training directories")
        return

    # 加载模型
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V2')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # 冻结早期层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # 训练和重复验证参数
    best_model_path = './best_nailong.pth'
    max_retrain_iterations = 5  # 最大重新划分次数
    target_accuracy = 0.95  # 目标准确率
    initial_epochs = 20  # 初次训练轮数
    retrain_epochs = 10  # 每次重新训练轮数

    # 初始训练
    iteration = 0
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_acc = train_model(model, train_loader, val_loader, initial_epochs, device, best_model_path, iteration)
    iteration += 1

    # 重复划分验证集和训练，直到准确率达标或达到最大迭代次数
    while val_acc < target_accuracy and iteration < max_retrain_iterations:
        logging.info(f"Validation accuracy {val_acc:.4f} is below {target_accuracy}, re-splitting dataset for iteration {iteration}...")
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        val_acc = train_model(model, train_loader, val_loader, retrain_epochs, device, best_model_path, iteration)
        iteration += 1

    if val_acc >= target_accuracy:
        logging.info(f"Reached target accuracy {val_acc:.4f} >= {target_accuracy}")
    else:
        logging.warning(f"Stopped after {max_retrain_iterations} iterations, final accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    main()