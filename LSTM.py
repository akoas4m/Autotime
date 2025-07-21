import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

# ====== 随机种子设置 ======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1024
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== 数据准备 ======
data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\AutoTimes-main\dataset\OBS_FD01_cleaned.csv",
    sep=',',
    header=0,
    engine='python'
)

X_raw = data.iloc[:, 1:2].values
y_raw = data.iloc[:, 2:3].values

total_size = len(X_raw)
train_size = int(total_size * 0.8)
test_size = total_size - train_size

X_train = X_raw[:train_size]
X_test = X_raw[train_size:]
y_train = y_raw[:train_size]
y_test = y_raw[train_size:]

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

def create_sequences_multi_step(X_data, y_data, window_size=576, pred_steps=96):
    sequences = []
    labels = []
    for i in range(len(X_data) - window_size - pred_steps + 1):
        X_seq = X_data[i:i + window_size]
        y_seq = y_data[i + window_size: i + window_size + pred_steps]
        sequences.append(X_seq)
        labels.append(y_seq)
    return np.array(sequences), np.array(labels)

window_size = 576
pred_steps = 96

X_train_seq, y_train_seq = create_sequences_multi_step(
    X_train_scaled, y_train_scaled,
    window_size=window_size,
    pred_steps=pred_steps
)
X_test_seq, y_test_seq = create_sequences_multi_step(
    X_test_scaled, y_test_scaled,
    window_size=window_size,
    pred_steps=pred_steps
)

print(f"训练输入形状：{X_train_seq.shape} → (样本数, 历史窗口长度, 特征数)")
print(f"训练目标形状：{y_train_seq.shape} → (样本数, 预测步数, 目标数)")
print(f"测试输入形状：{X_test_seq.shape}")
print(f"测试目标形状：{y_test_seq.shape}")

X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

class MultiStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2,
                 dropout_rate=0.1, pred_steps=96):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_steps = pred_steps

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, pred_steps)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.unsqueeze(-1)

model = MultiStepLSTM(
    input_dim=X_train_seq.shape[-1],
    hidden_dim=32,
    num_layers=2,
    dropout_rate=0.1,
    pred_steps=pred_steps
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# ====== 训练循环 ======
best_test_r2 = -np.inf
train_losses = []
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = test_outputs.cpu().numpy().reshape(-1, pred_steps)
        test_true = y_test_tensor.cpu().numpy().reshape(-1, pred_steps)
        test_r2 = r2_score(test_true.flatten(), test_pred.flatten())

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        torch.save(model.state_dict(), 'best_multi_step_model.pth')

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"loss：{avg_train_loss:.6f} | R²：{test_r2:.4f}")

# ====== 最终评估 ======
model.load_state_dict(torch.load('best_multi_step_model.pth', weights_only=True))
model.eval()

with torch.no_grad():
    final_pred = model(X_test_tensor).cpu().numpy()
    final_true = y_test_tensor.cpu().numpy()

# Inverse transform to get original scale
final_pred_original = scaler_y.inverse_transform(final_pred.reshape(-1, 1)).reshape(-1, pred_steps)
final_true_original = scaler_y.inverse_transform(final_true.reshape(-1, 1)).reshape(-1, pred_steps)

# Metrics calculation using original (non-normalized) data
metrics = {
    "R²": r2_score(final_true_original.flatten(), final_pred_original.flatten()),
    "MSE": mean_squared_error(final_true_original.flatten(), final_pred_original.flatten()),
    "RMSE": np.sqrt(mean_squared_error(final_true_original.flatten(), final_pred_original.flatten())),
    "MAE": mean_absolute_error(final_true_original.flatten(), final_pred_original.flatten()),
    "MAPE": np.mean(np.abs(
        (final_true_original.flatten() - final_pred_original.flatten()) /
        (final_true_original.flatten() + 1e-10)
    )) * 100
}

# Step-wise MSE using original data
step_mse = []
for step in range(pred_steps):
    step_pred = final_pred_original[:, step]
    step_true = final_true_original[:, step]
    step_mse.append(mean_squared_error(step_true, step_pred))

# Print metrics
print("Final Metrics (Original Scale):", metrics)
print("Step-wise MSE (Original Scale):", step_mse)
