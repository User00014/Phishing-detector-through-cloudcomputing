import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# 定义Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# 定义1D卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        kernel_size = 9
        padding = ((kernel_size - 1) // 2)
        self.conv1 = nn.Conv1d(in_channels=85, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = x.permute(1, 0)
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(1, 0)
        x = self.pool(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 置信度过滤策略生成伪标签
def generate_pseudo_labels(model, data_loader, threshold):
    model.eval()
    pseudo_data = []
    pseudo_labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, pseudo_label = probs.max(dim=1)
            mask = conf >= threshold
            pseudo_data.append(inputs[mask])
            pseudo_labels.append(pseudo_label[mask])

    if len(pseudo_data) > 0 and len(pseudo_labels) > 0:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        return DataLoader(TensorDataset(pseudo_data, pseudo_labels), batch_size=100, shuffle=True)
    return None

# 加载有标签数据
train_data = pd.read_csv("splited_data/train_data_version.csv")
train_data = train_data.iloc[:, 1:]
labels = train_data.pop("label")
numpy_features = train_data.values
tensor_features = torch.from_numpy(numpy_features).float()
tensor_labels = torch.tensor(labels.values, dtype=torch.int)
train_dataset = TensorDataset(tensor_features, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 加载无标签数据
unlabeled_data = pd.read_csv("splited_data/unlabeled_data.csv")
unlabeled_data = unlabeled_data.iloc[:, 1:]
numpy_unlabeled_features = unlabeled_data.values
tensor_unlabeled_features = torch.from_numpy(numpy_unlabeled_features).float()
unlabeled_dataset = TensorDataset(tensor_unlabeled_features)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=False)

# 创建模型实例
model = Net()
criterion = FocalLoss(alpha=1, gamma=2)  # Focal Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 半监督训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    # 使用有标签数据训练
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

    # 每20个epoch更新一次无标签数据集的伪标签
    if (epoch + 1) % 20 == 0:
        pseudo_loader = generate_pseudo_labels(model, unlabeled_loader, threshold=0.6)
        if pseudo_loader is not None:
            print(f"Epoch [{epoch + 1}/{num_epochs}], 使用伪标签数据进行训练")
            for data, target in pseudo_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), f'./model_epoch_{epoch}.pth')

# 加载并验证模型
model = Net()
model.load_state_dict(torch.load("model_epoch_99.pth", map_location=torch.device('cpu')))
model.eval()
test_data = pd.read_csv("splited_data/test_data.csv")
test_data = test_data.iloc[:, 1:]
test_labels = test_data.pop("label")
test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.int)
test_input = test_data.values
test_input = torch.from_numpy(test_input).float()

with torch.no_grad():
    output = model(test_input)
    pred_labels = torch.argmax(output, dim=1)
    pred_labels_np = pred_labels.cpu().numpy()
    test_labels_np = test_labels_tensor.cpu().numpy()
    precision = precision_score(test_labels_np, pred_labels_np, zero_division=1, average='macro')
    recall = recall_score(test_labels_np, pred_labels_np, average='macro')
    f1 = f1_score(test_labels_np, pred_labels_np, average='macro')
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
