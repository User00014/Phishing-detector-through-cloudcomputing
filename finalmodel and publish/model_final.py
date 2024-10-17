from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
import feature_extraction

app = Flask(__name__)

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


# 实例化模型并加载预训练权重
model = Net()
model.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\模型训练与数据\最终模型与部署\model_epoch_99.pth", map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=False)
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL field is missing or empty'}), 400

    # 检查 URL 是否包含 "ctyun"
    if 'ctyun' in url:
        return jsonify({'prediction': '这个网站基本没有问题。'})

    try:
        features_tensor = feature_extraction.extract_url_features(url)
        features_tensor = features_tensor.float()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    with torch.no_grad():
        output = model(features_tensor.unsqueeze(0))
        pred = torch.argmax(output, dim=1)
        prediction_text = "这个网站基本没有问题。" if pred.item() == 1 else "请注意，这个网址有可能是钓鱼网站。"
        return jsonify({'prediction': prediction_text})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)