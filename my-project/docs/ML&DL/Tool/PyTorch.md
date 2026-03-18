---
icon: simple/pytorch
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

---

??? example "MNIST"
    ```python title="Train + Test"
    # 导入所需的包
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # 设置超参数
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 15,
    }
    
    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 可视化样本数据
    examples = iter(train_loader)
    images, labels = next(examples)
    
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f"label:{labels[i]}")
        plt.axis('off')
    plt.show()
    
    # 构建模型
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Flatten(),                                 # 展平 28x28 图像
                nn.Linear(28*28, 512),                        # 输入层
                nn.BatchNorm1d(512),                          # 批归一化
                nn.LeakyReLU(0.1),                            # LeakyReLU 防止死神经元
                nn.Dropout(0.3),                              # Dropout 防止过拟合
    
                nn.Linear(512, 256),                          # 中间层1
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
    
                nn.Linear(256, 128),                          # 中间层2
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
    
                nn.Linear(128, 10),                           # 输出层
            )
    
        def forward(self, x):
            return self.model(x)
    
    model = SimpleNN().to(config["device"])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # 训练模型
    from tqdm import tqdm
    
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        model.train()
    
        print(f"\n🔄 Epoch {epoch+1}/{config['epochs']}")
        train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True, leave=False)
    
        for images, labels in train_bar:
            images, labels = images.to(config["device"]), labels.to(config["device"])
    
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
    
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    
    # 模型测试
    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        for images, labels in tqdm(test_loader, desc="Testing", dynamic_ncols=True, leave=False):
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    
    # 加载模型并预测
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(config["device"]), labels.to(config["device"])
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    # 可视化混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 显示热力图
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    ```
    
    ```python title="GUI"
    import tkinter as tk
    from tkinter import Canvas, Button, Label
    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw, ImageOps
    from torchvision import transforms
    
    
    # ======== 1. 模型结构 ========= #
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Flatten(),                                 # 展平 28x28 图像
                nn.Linear(28*28, 512),                        # 输入层
                nn.BatchNorm1d(512),                          # 批归一化
                nn.LeakyReLU(0.1),                            # LeakyReLU 防止死神经元
                nn.Dropout(0.3),                              # Dropout 防止过拟合
    
                nn.Linear(512, 256),                          # 中间层1
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
    
                nn.Linear(256, 128),                          # 中间层2
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
    
                nn.Linear(128, 10),                           # 输出层
            )
    
        def forward(self, x):
            return self.model(x)
    
    # ======== 2. 图像预处理函数 ========= #
    def preprocess_image(img):
        img = img.convert("L")  # 转换为灰度图 非常重要
        img = ImageOps.pad(img, (28, 28), color=0)  # 填充图像
    
        # 归一化参数：根据 MNIST 数据集的统计信息
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 使用 MNIST 数据集的标准化参数
        ])
        return transform(img).unsqueeze(0)
    
    # ======== 3. GUI 应用类 ========= #
    class App:
        def __init__(self):
            self.window = tk.Tk()
            self.window.title("MNIST")
    
            self.canvas = Canvas(self.window, width=280, height=280, bg='black')
            self.canvas.grid(row=0, column=0, columnspan=4)
            self.canvas.bind('<B1-Motion>', self.draw)
    
            self.label = Label(self.window, text="Prediction: None", font=("Arial", 18))
            self.label.grid(row=1, column=0, columnspan=4)
            self.prob_label = Label(self.window, text="Probabilities:", font=("Arial", 12), justify="left", anchor="w")
            self.prob_label.grid(row=3, column=0, columnspan=4, sticky="w")  # 左对齐
    
            Button(self.window, text="Predict", command=self.predict).grid(row=2, column=0)
            Button(self.window, text="Clear", command=self.clear).grid(row=2, column=1)
            Button(self.window, text="Exit", command=self.window.quit).grid(row=2, column=2)
    
            self.image = Image.new("L", (280, 280), color=0)
            self.draw_interface = ImageDraw.Draw(self.image)
    
            # 确保选择设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
            # 加载模型
            self.model = SimpleNN().to(self.device)
            self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
            self.model.eval()  # 切换到评估模式
    
            self.window.mainloop()
    
        def draw(self, event):
            x, y = event.x, event.y
            r = 8
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
            self.draw_interface.ellipse([x - r, y - r, x + r, y + r], fill=255)
    
        def clear(self):
            self.canvas.delete("all")
            self.image = Image.new("L", (280, 280), color=0)
            self.draw_interface = ImageDraw.Draw(self.image)
            self.label.config(text="Prediction: None")
    
        def predict(self):
            img_tensor = preprocess_image(self.image).to(self.device)  # 将图像数据传到设备
            output = self.model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            conf = prob[0][pred].item()
            self.label.config(text=f"Prediction: {pred} ({conf:.2%})")
    
            # 构建概率文本
            prob_text = "Probabilities:\n"
            for i, p in enumerate(prob[0]):
                prob_text += f"  {i}: {p.item():.2%}\n"
    
            # 设置到界面上
            self.prob_label.config(text=prob_text)
    
    
    if __name__ == '__main__':
        App()
    ```

---
