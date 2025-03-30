---
icon: material/torch
---


[TOC]

---

## 一、`dataset`和`dataloader`

- **Dataset**：存储数据样本和期望值
- **Dataloader**：将数据分组为批次，支持多进程

```python
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
```

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file):
        self.data = ...  # 读取数据并进行预处理

    def __getitem__(self, index):
        return self.data[index]  # 每次返回一个样本

    def __len__(self):
        return len(self.data)  # 返回数据集的大小
```

---

## 二、`tensor`

- 概念：高维矩阵
    - 1-D：声音
      - 2-D：黑白图片
      - 3-D：彩色照片
      - ……

---

### 1、常见操作

- 查看大小、维度

    ```python
    .shape()	# 查看矩阵大小
    ```

- 直接从数据（列表或numpy.ndarray）创建张量
    ```python
    x = torch.tensor([[1, -1], [-1, 1]])
    x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))
    ```

- 创建全零或全一的张量
    ```python
    x = torch.zeros([2, 2])  # 形状为2x2的全零张量
    x = torch.ones([1, 2, 5])  # 形状为1x2x5的全一张量
    ```

- 转置

    ```python
    x = x.transpose(0, 1)
    ```

- `sqeeze`/`unsqeeze`

    ```python
    x = x.sqeeze(0)	# 压缩第一个维度
    x = x.unsqeeze(1)	# 第二个维度恢复成 1
    ```

- `cat`

    ```python
    w = torch.cat([x, y, z], dim=1)	# 在第二个维度上进行合并
    ```

---

### 2、设备

- 默认情况下，张量和模块将在**CPU**上进行计算，使用 `.to()` 方法将张量移动到适当的设备

- **CPU**
  
  ```python
  x = x.to('cpu')
  ```
  
- **GPU**
  
  ```python
  x = x.to('cuda')
  ```
  
!!! tip
    - **检查计算机是否有 NVIDIA GPU**
        - 使用 `torch.cuda.is_available()` 函数来检查你的计算机是否支持 NVIDIA GPU。
  
    这个函数会返回一个布尔值，如果计算机有可用的 NVIDIA GPU 并且正确安装了 CUDA，那么返回 `True`，否则返回 `False`。

---

### 3、自动求导

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)	# 创建张量并设置 requires_grad=True
z = x.pow(2).sum()
z.backward()	# 执行反向传播
x.grad		# 查看梯度
```

---

## 三、`torch.nn`

### 1、激活函数

```python
nn.sigmoid()
nn.ReLU()
nn.Linear(in_feature, out_feature)	# 全连接
```

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),  # 输入层到隐藏层的线性变换
            nn.Sigmoid(),       # 激活函数
            nn.Linear(32, 1)    # 隐藏层到输出层的线性变换
        )
    
    def forward(self, x):
        return self.net(x)    # 计算神经网络的输出
```

---

### 2、损失函数

- **均方误差（Mean Squared Error）**：用于回归任务。

   ```python
   criterion = nn.MSELoss()
   ```

- **交叉熵（Cross Entropy）**：用于分类任务。

   ```python
   criterion = nn.CrossEntropyLoss()
   ```

- **计算损失**：

   ```python
   loss = criterion(model_output, expected_value)
   ```

---

## 四、`torch.optim`

```python
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0)
```

---

## 五、训练流程

- 设置

```python
dataset = MyDataset(file)  # 读取数据 via MyDataset
tr_set = DataLoader(dataset, 16, shuffle=True)  # 将数据集放入 DataLoader
model = MyModel().to(device)  # 构建模型并移动到设备 (cpu/cuda)
criterion = nn.MSELoss()  # 设置损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.1)  # 设置优化器
```

- 训练循环

```python
for epoch in range(n_epochs):  # 迭代 n_epochs
    model.train()  # 设置模型为训练模式
    for x, y in tr_set:  # 迭代数据加载器
        optimizer.zero_grad()  # 重置梯度
        x, y = x.to(device), y.to(device)  # 将数据移动到设备 (cpu/cuda)
        pred = model(x)  # 前向传播 (计算输出)
        loss = criterion(pred, y)  # 计算损失
        loss.backward()  # 计算梯度 (反向传播)
        optimizer.step()  # 使用优化器更新模型参数
```

- 交叉验证循环

```python
model.eval()  # 设置模型为评估模式
total_loss = 0  # 初始化总损失
for x, y in dv_set:  # 迭代数据加载器
    x, y = x.to(device), y.to(device)  # 将数据移动到设备 (cpu/cuda)
    with torch.no_grad():  # 禁用梯度计算
        pred = model(x)  # 前向传播 (计算输出)
        loss = criterion(pred, y)  # 计算损失
        total_loss += loss.cpu().item() * len(x)  # 累积损失
avg_loss = total_loss / len(dv_set.dataset)  # 计算平均损失
```

- 测试集循环

```python
model.eval()  # 设置模型为评估模式
preds = []  # 初始化预测列表
for x in tt_set:  # 迭代数据加载器
    x = x.to(device)  # 将数据移动到设备 (cpu/cuda)
    with torch.no_grad():  # 禁用梯度计算
        pred = model(x)  # 前向传播 (计算输出)
        preds.append(pred.cpu())  # 收集预测
```

- 保存

```python
torch.save(model.state_dict(), path)
```

- 加载

```python
ckpt = torch.load(path)
model.load_state_dict(ckpt)
```

---

!!! tip
    - **model.eval()**

        - 改变某些模型层的行为，例如 dropout 和 batch normalization。
    - **with torch.no_grad()**

        - 防止计算被添加到梯度计算图中。通常用于防止在验证/测试数据上意外进行训练。

