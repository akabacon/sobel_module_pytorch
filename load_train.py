import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from sobel_module import StructuredSobelLayer
from torchvision.models import resnet18
import numpy as np

# 定義CNN+Transformer模型
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.sobel = StructuredSobelLayer()
        resnet = resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.flatten = nn.Flatten(2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.sobel(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
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
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

resume_path = 'cnn_transformer_epoch150.pt'
start_epoch = 151
end_epoch = 200  # 你可以改成250、300等
if os.path.exists(resume_path):
    print(f"載入權重: {resume_path}")
    model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))

log_file = open('train.log', 'a', encoding='utf-8')

def train(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
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
    for epoch in range(start_epoch, end_epoch+1):
        train(epoch)
        val_loss = test()
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current LR: {current_lr}')
        log_file.write(f'Current LR: {current_lr}\n')
        log_file.flush()
        torch.save(model.state_dict(), f'cnn_transformer_epoch{epoch}.pt')
    log_file.close()
