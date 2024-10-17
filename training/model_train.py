import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
"""定义一维卷积神经网络模型"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一维卷积层，它有853个输入通道、32个输出通道，卷积核大小为7，padding大小为4
        kernel_size = 9
        padding = ((kernel_size - 1) // 2)  # 计算填充量
        self.conv1 = nn.Conv1d(in_channels=85, out_channels=32, kernel_size=kernel_size,padding = padding)
        #self.conv1 = nn.Linear(853, 32)
        # 最大池化层,池化窗口大小为2
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层,输入特征数量为32 * 46（这个值可能需要根据实际输入数据调整），输出特征数量为10。
        self.fc1 = nn.Linear(16, 10)  # 更新展平操作后的大小
        # Dropout层,丢弃概率为0.5，用于正则化以减少过拟合
        self.dropout = nn.Dropout(p=0.1)
        # 输出层
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = x.permute(1,0)
        # 对输入x应用ReLU激活函数和卷积层conv1，然后应用最大池化层pool。
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(1, 0)#矩阵转秩
        x = self.pool(x)
        print('-------pool feature size:{}'.format(x.shape))
        # 将卷积层的输出展平成一维向量，以匹配全连接层fc1的输入要求
        # 对展平后的特征应用通过全连接层fc1。再应用ReLU激活函数
        x = nn.functional.relu(self.fc1(x))
        # 进入Dropout层，减少过拟合
        x = self.dropout(x)
        # 通过输出层fc2，并使用Sigmoid激活函数（通常用于二分类问题），返回预测结果。
        x = self.fc2(x)
        return x


# 创建模型实例
model = Net()
# 打印模型结构
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
#读取数据
train_data = pd.read_csv("splited_data/train_data_label.csv")
print(train_data.columns)
"""对读取数据进行预处理（保存的有区别所以要调整）"""
#train_data = train_data.set_index('Unnamed: 0')
#train_data = train_data.iloc[:, :-1] #删除最后一列
train_data = train_data.iloc[:,1:]#删除第一列
"""分离标签和特征"""
labels = train_data.pop("label")  # 使用pop方法删除并返回'label'列
"""数据归一化"""
# train_data_normalized = train_data.copy()  # 使用copy()避免高度碎片化的警告
# for column in train_data_normalized.columns:
#     min_val = train_data_normalized[column].min()
#     max_val = train_data_normalized[column].max()
#     # 如果最大值和最小值相同，为了避免除以0，可以将该列的所有值设置为0
#     if max_val == min_val:
#         train_data_normalized[column] = 0
#     else:
#         train_data_normalized[column] = (train_data_normalized[column] - min_val) / (max_val - min_val)
# train_data = train_data_normalized.copy()
numpy_features = train_data.values  # 获取剩余特征
# 将NumPy数组转换为PyTorch张量
tensor_features = torch.from_numpy(numpy_features).float()  # 假设特征是浮点型
tensor_labels = torch.tensor(labels.values, dtype=torch.int)  # 假设标签是整型（如果是分类问题）
# 创建TensorDataset
train_dataset = TensorDataset(tensor_features, tensor_labels)
print(tensor_features.shape, tensor_labels.shape)
# 定义DataLoader
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

"""模型训练"""
num_epochs = 50  # 设置训练的轮数（epoch）为50。
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        print('----input shape:{}---'.format(data.shape))
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target.long())  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        if (epoch + 1) % 10 == 0:  # 每10轮输出一次训练状态。
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # 打印当前轮数和损失值。
            torch.save(model.state_dict(), './model_epoch_{}.pth'.format(epoch)) #每训练10轮保存一次模型


"""加载保存的模型权重"""
model = Net()
model.load_state_dict(torch.load("model_epoch_49.pth",map_location=torch.device('cpu'), weights_only=True))
"""用验证集进行验证"""
model.eval()
test_data = pd.read_csv("splited_data/test_data.csv")
test_data = test_data.iloc[:,1:]
test_labels = test_data.pop("label")  # 使用pop方法删除并返回'label'列、

test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.int) #将标签值转换为张量2
test_input = test_data.values  # 获取剩余的特征列
test_input = torch.from_numpy(test_input).float()  # 转为Pytorch张量
with torch.no_grad():
    output = model(test_input)  # 将测试输入通过模型，获取输出。
    pred_labels = torch.argmax(output, dim=1)
    pred_labels_np = pred_labels.cpu().numpy()
    test_labels_np = test_labels_tensor.cpu().numpy()
    # 计算精确率、召回率和F1分数
    precision = precision_score(test_labels_np, pred_labels_np, average='macro')
    recall = recall_score(test_labels_np, pred_labels_np, average='macro')
    f1 = f1_score(test_labels_np, pred_labels_np, average='macro')
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

"""打印预测结果"""
# batch_size = 10
# for i in range(0, pred_labels.size(0), batch_size):
#     print(pred_labels[i:i + batch_size])


