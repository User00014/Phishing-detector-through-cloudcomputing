
"""引入去噪策略的训练代码"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
"""定义标签平滑损失"""
# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.criterion = nn.KLDivLoss(reduction='batchmean')
#     def forward(self, output, target):
#         true_dist = torch.zeros_like(output)
#         true_dist.fill_(self.smoothing / (self.cls - 1))
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return self.criterion(F.log_softmax(output, dim=-1), true_dist)
"""定义噪声稳健损失函数（焦点损失）"""
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
"""定义一维卷积神经网络模型"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一维卷积层，它有853个输入通道、32个输出通道，卷积核大小为7，padding大小为4
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
        print('-------pool feature size:{}'.format(x.shape))
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""带置信度过滤的自我训练"""
"""用于实现带置信度过滤的自我训练策略。这个策略的目的是在模型训练过程中"""
"""根据模型对数据的预测置信度来过滤掉那些置信度较低的样本，只保留高置信度的样本，从而提高模型的鲁棒性和准确性。"""
def confidence_filtering(model, data_loader, threshold):
    model.eval()
    filtered_data = []
    filtered_labels = []
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            probs = F.softmax(output, dim=1)#对模型输出应用softmax函数，得到每个类别的预测概率。
            conf, pred = probs.max(dim=1)#获取每个样本的最大预测概率（置信度）和对应的预测类别。
            mask = conf >= threshold
            filtered_data.append(data[mask])
            filtered_labels.append(pred[mask])
    # 如果没有数据通过过滤，返回 None
    if len(filtered_data) == 0 or len(filtered_labels) == 0:
        print("没有数据通过置信度过滤条件")
        return None
    # 将过滤后的数据和标签拼接到一起
    filtered_data = torch.cat(filtered_data, dim=0)
    filtered_labels = torch.cat(filtered_labels, dim=0)
    if filtered_data.size(0) == 0:
        print("没有数据通过置信度过滤条件")
        return None
    # 创建新的 DataLoader
    return DataLoader(TensorDataset(filtered_data, filtered_labels), batch_size=100, shuffle=True)


# 创建模型实例
model = Net()
print(model)
# 使用标签平滑和焦点损失
num_classes = 2
# criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.2)
criterion = FocalLoss(alpha=1, gamma=2)  # 如果想用Focal Loss，请替换此行
optimizer = optim.Adam(model.parameters(), lr=0.001)
#读取数据
train_data = pd.read_csv("splited_data/train_data_version.csv")
train_data = train_data.iloc[:, 1:]
labels = train_data.pop("label")
numpy_features = train_data.values
tensor_features = torch.from_numpy(numpy_features).float()
tensor_labels = torch.tensor(labels.values, dtype=torch.int)
train_dataset = TensorDataset(tensor_features, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# 模型训练
num_epochs = 100
new_train_loader = train_loader
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print('----input shape:{}---'.format(data.shape))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
    # 每10个epoch进行一次数据过滤
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # 使用信心过滤方法更新训练数据集
        # if epoch>40:
        #     new_train_loader = confidence_filtering(model, train_loader, threshold=0.6)
        # 检查新数据加载器是否为空
        if new_train_loader is not None:
            train_loader = new_train_loader
        else:
            print(f"无需进行过滤过滤后的数据为空，跳过第 {epoch + 1} 轮训练")
            break
        # 保存模型权重
        torch.save(model.state_dict(), './model_epoch_{}.pth'.format(epoch))



#加载保存的模型权重
model = Net()
model.load_state_dict(torch.load("model_epoch_99.pth", map_location=torch.device('cpu'),weights_only=True))
# 用验证集进行验证
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
    precision = precision_score(test_labels_np, pred_labels_np, zero_division = 1, average='macro')
    recall = recall_score(test_labels_np, pred_labels_np, average='macro')
    f1 = f1_score(test_labels_np, pred_labels_np, average='macro')
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)



