import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from sobel_module import StructuredSobelLayer
from torch.amp import autocast, GradScaler
from torchvision.models import resnet34
import numpy as np

# 定義CNN+Transformer模型
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.sobel = StructuredSobelLayer()
        # 使用ResNet34作為backbone，去掉最後的fc與avgpool
        resnet = resnet34(weights=None)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 6通道for Sobel
        resnet.maxpool = nn.Identity()  # CIFAR-10不需要maxpool
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )  # 輸出shape: (B, 256, 8, 8)
        self.flatten = nn.Flatten(2)  # (B, C, H*W)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.sobel(x)  # (B,6,32,32)
        x = self.backbone(x)  # (B,256,8,8)
        x = self.flatten(x)  # (B,256,64)
        x = x.transpose(1, 2)  # (B,64,256)
        x = self.transformer(x)  # (B,64,256)
        x = x.mean(dim=1)  # (B,256)
        x = self.fc(x)  # (B,num_classes)
        return x

# 數據增強與載入
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Cutout增強
class Cutout(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
transform_train.transforms.append(Cutout(n_holes=1, length=8))
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scaler = GradScaler('cuda') if device.type == 'cuda' else None

# ========== 續訓與log功能 ===========
resume_path = 'cnn_transformer_epoch.pt'  # 你可以改成你要的檔名
end_epoch = 300
if os.path.exists(resume_path):
    print(f"載入權重: {resume_path}")
    model.load_state_dict(torch.load(resume_path, map_location=device))
    start_epoch = int(resume_path.split('epoch')[-1].split('.pt')[0]) + 1

log_file = open('train.log', 'a', encoding='utf-8')

def train(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    log_str = f'Epoch {epoch} | Train Loss: {total_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%'
    print(log_str)
    log_file.write(log_str + '\n')
    log_file.flush()

def test():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss/len(testloader)
    log_str = f'Test Loss: {avg_loss:.4f} | Acc: {100.*correct/total:.2f}%'
    print(log_str)
    log_file.write(log_str + '\n')
    log_file.flush()
    return avg_loss

if __name__ == '__main__':
    for epoch in range(1, end_epoch+1):
        train(epoch)
        val_loss = test()
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr}')
        log_file.write(f'Current LR: {current_lr}\n')
        log_file.flush()
        torch.save(model.state_dict(), f'cnn_transformer_epoch{epoch}.pt')
    log_file.close() 