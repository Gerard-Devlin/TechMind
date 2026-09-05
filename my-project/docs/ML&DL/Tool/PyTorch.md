---
icon: simple/pytorch
---

[TOC]

---

## 一、速查

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

下面各行独立查阅。示例中的 `x`、`y` 是张量；涉及维度时，需要满足对应的形状要求。维度从 `0` 开始，`dim=-1` 表示最后一维。

### 1、创建

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 从数据创建 | `torch.tensor([1, 2, 3])` | 复制数据，默认推断类型 |
| 指定类型 | `torch.tensor([1, 2], dtype=torch.float32)` | 创建浮点张量 |
| 从 NumPy 创建 | `torch.from_numpy(arr)` | 与原数组共享内存 |
| 尽量复用数据 | `torch.as_tensor(data)` | 类型、设备等条件允许时不复制 |
| 全零 / 全一 | `torch.zeros(2, 3)` / `torch.ones(2, 3)` | 形状为 `[2, 3]` |
| 指定填充值 | `torch.full((2, 3), 5.0)` | 所有元素为 `5.0` |
| 单位矩阵 | `torch.eye(3)` | 形状为 `[3, 3]` |
| 等差序列 | `torch.arange(0, 10, 2)` | 左闭右开，得到 `0, 2, 4, 6, 8` |
| 等距取点 | `torch.linspace(0, 1, 5)` | 含两端，共取 `5` 个点 |
| 均匀随机数 | `torch.rand(2, 3)` | 范围 `[0, 1)` |
| 标准正态 | `torch.randn(2, 3)` | 均值 `0`、方差 `1` |
| 随机整数 | `torch.randint(0, 10, (2, 3))` | 范围 `[0, 10)` |
| 随机排列 | `torch.randperm(5)` | 打乱 `0` 到 `4` |
| 参照张量 | `torch.zeros_like(x)` / `torch.randn_like(x)` | 默认继承形状、类型和设备，随机正态要求浮点或复数类型 |
| 未初始化 | `torch.empty(2, 3)` | 分配空间，内容不确定 |
| 随机种子 | `torch.manual_seed(42)` | 设置随机数种子 |

### 2、属性

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 形状 | `x.shape` / `x.size()` | 如 `torch.Size([2, 3])` |
| 某一维长度 | `x.size(0)` | 第 `0` 维长度 |
| 维数 | `x.ndim` / `x.dim()` | 形状中轴的数量 |
| 元素数 | `x.numel()` | 所有维度长度的乘积 |
| 数据类型 | `x.dtype` | 如 `torch.float32` |
| 所在设备 | `x.device` | 如 `cpu`、`cuda:0` |
| 是否求导 | `x.requires_grad` | 是否需要跟踪梯度 |
| 是否连续 | `x.is_contiguous()` | 内存布局是否连续 |
| 步长 | `x.stride()` | 沿各轴移动一个位置跨过的元素数 |

### 3、索引

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 取行 / 列 | `x[0]` / `x[:, 0]` | 对二维张量，分别取第一行、第一列 |
| 切片 | `x[:, 1:3]` | 取第 `1`、`2` 列 |
| 保留维度 | `x[0:1]` | 取第一项，但不删掉第 `0` 维 |
| 指定下标 | `x[[0, 2]]` | 取第 `0`、`2` 项 |
| 布尔筛选 | `x[x > 0]` | 得到所有正元素的一维张量 |
| 条件选择 | `torch.where(x > 0, x, 0)` | 正数保留，其余置零 |
| 非零下标 | `torch.nonzero(x)` | 每行是一组非零元素坐标 |
| 按轴选择 | `torch.index_select(x, 0, idx)` | `idx` 是一维整数索引 |
| 逐行取值 | `torch.gather(x, 1, idx)` | 二维 `x` 与 `idx` 行数相同，按各行索引取列 |
| 掩码填充 | `x.masked_fill(mask, 0)` | 布尔掩码为真的位置填 `0` |

!!! tip
    多个条件组合时使用 `&`、`|`、`~`，每个比较加括号，如 `(x > 0) & (x < 1)`；不要使用 Python 的 `and`、`or`。

### 4、变形

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 改变形状 | `x.reshape(2, -1)` | 自动推算一维，元素总数不变 |
| 视图变形 | `x.view(2, -1)` | 要求原形状和步长兼容 |
| 展平 | `x.flatten()` | 合成一维 |
| 保留批次 | `x.flatten(1)` | 从第 `1` 维开始展平 |
| 增加维度 | `x.unsqueeze(0)` | 在第 `0` 维插入长度 `1` |
| 删除单维 | `x.squeeze(1)` | 仅当第 `1` 维长度为 `1` 时删除 |
| 交换两轴 | `x.transpose(0, 1)` | 交换第 `0`、`1` 维 |
| 调整轴序 | `x.permute(0, 2, 3, 1)` | 四维张量由 `NCHW` 变为 `NHWC` |
| 连续布局 | `x.contiguous()` | 必要时复制为连续内存 |
| 广播扩展 | `x.expand(4, -1)` | 将 `[1, D]` 扩为 `[4, D]`，共享存储 |
| 复制重复 | `x.repeat(4, 1)` | 将二维张量沿第 `0` 维重复 `4` 次 |

### 5、拼接

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 沿原轴拼接 | `torch.cat([x, y], dim=0)` | 除拼接轴外，其余维度相同 |
| 沿新轴堆叠 | `torch.stack([x, y], dim=0)` | 两个张量形状必须相同 |
| 按长度切分 | `x.split(2, dim=0)` | 每段长度为 `2`，最后一段可以更短 |
| 按段数切分 | `x.chunk(3, dim=0)` | 尝试分为 `3` 段，可能返回更少段 |
| 指定段数 | `torch.tensor_split(x, 3, dim=0)` | 恰好分成 `3` 段，允许空段 |
| 拆掉一轴 | `x.unbind(dim=0)` | 返回元组，每项少一维 |

### 6、运算

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 逐元素运算 | `x + y`、`x - y`、`x * y`、`x / y` | 形状相同或可广播 |
| 幂 / 开方 | `x.square()` / `x.sqrt()` | 开方时注意取值范围 |
| 指数 / 对数 | `x.exp()` / `x.log()` | 实数对数要求输入为正 |
| 绝对值 | `x.abs()` | 逐元素绝对值 |
| 截断范围 | `x.clamp(min=0, max=1)` | 限制在 `[0, 1]` |
| 逐元素极值 | `torch.maximum(x, y)` / `torch.minimum(x, y)` | 比较对应元素 |
| 矩阵乘法 | `x @ y` / `torch.matmul(x, y)` | 也支持向量和批次广播 |
| 二维矩阵乘法 | `torch.mm(x, y)` | `[M, K] @ [K, N] → [M, N]` |
| 批量矩阵乘法 | `torch.bmm(x, y)` | `[B, M, K] @ [B, K, N]`，不广播 |
| 向量点积 | `torch.dot(x, y)` | 两个等长一维张量 |
| 爱因斯坦求和 | `torch.einsum("bnd,bd->bn", x, y)` | 沿重复且未出现在输出中的 `d` 求和 |
| L2 范数 | `torch.linalg.vector_norm(x, dim=-1)` | 沿最后一维计算 |
| 单位化 | `F.normalize(x, dim=-1)` | 默认按 L2 范数归一化 |

### 7、统计

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 求和 / 均值 | `x.sum(dim=1)` / `x.mean(dim=1)` | 均值要求浮点或复数类型 |
| 保留轴 | `x.sum(dim=1, keepdim=True)` | 被归约的轴保留为长度 `1` |
| 全局极值 | `x.max()` / `x.min()` | 返回单个极值 |
| 沿轴极值 | `values, indices = x.max(dim=1)` | 同时返回值和下标 |
| 极值下标 | `x.argmax(dim=-1)` / `x.argmin(dim=-1)` | 常用于取预测类别 |
| 方差 / 标准差 | `x.var(correction=0)` / `x.std(correction=0)` | `correction=0` 除以 `N`，默认 `1` 除以 `N-1` |
| 累加 | `x.cumsum(dim=0)` | 沿指定轴累计求和 |
| 排序 | `values, indices = x.sort(dim=-1)` | 默认升序 |
| 排序下标 | `x.argsort(dim=-1, descending=True)` | 按降序返回下标 |
| 前 k 大 | `values, indices = x.topk(3, dim=-1)` | 最后一维至少有 `3` 个元素 |
| 去重 | `torch.unique(x)` | 默认对所有元素去重 |
| 任意 / 全部 | `(x > 0).any()` / `(x > 0).all()` | 返回布尔张量 |
| 精确比较 | `torch.equal(x, y)` | 形状和元素是否相同 |
| 近似比较 | `torch.allclose(x, y)` | 允许浮点误差 |
| 异常值 | `torch.isnan(x)` / `torch.isfinite(x)` | 逐元素检查 NaN / 有限值 |
| 概率分布 | `torch.softmax(x, dim=-1)` | 沿最后一维归一化 |
| 对数概率 | `torch.log_softmax(x, dim=-1)` | 比先 softmax 再 log 更稳定 |

### 8、转换

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 转浮点 / 整数 | `x.float()` / `x.long()` | 转为 `float32` / `int64` |
| 转布尔 | `x.bool()` | 零为假，非零为真 |
| 指定类型、设备 | `x.to(device=device, dtype=torch.float32)` | 接住返回值 |
| 移至 CPU | `x.cpu()` | 返回 CPU 张量 |
| 取单个值 | `x.item()` | 仅适用于只有一个元素的张量 |
| 转列表 | `x.tolist()` | 转成 Python 标量或嵌套列表 |
| 转 NumPy | `x.detach().cpu().numpy()` | 常规数值张量转数组 |
| 复制 | `x.clone()` | 独立存储，仍保留求导关系 |
| 脱离计算图 | `x.detach()` | 不再跟踪梯度，但共享存储 |
| 独立副本 | `x.detach().clone()` | 独立存储，且脱离计算图 |

### 9、训练

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 训练 / 评估 | `model.train()` / `model.eval()` | 切换 Dropout、BatchNorm 等层的行为 |
| 清空梯度 | `optimizer.zero_grad(set_to_none=True)` | 将梯度设为 `None` |
| 反向传播 | `loss.backward()` | 将梯度累积到叶子张量 |
| 更新参数 | `optimizer.step()` | 根据梯度更新参数 |
| 禁用梯度 | `with torch.no_grad():` | 不记录反向计算图 |
| 纯推理 | `with torch.inference_mode():` | 额外减少跟踪开销 |
| 冻结参数 | `model.requires_grad_(False)` | 模型参数不再计算梯度 |
| 查看参数 | `model.named_parameters()` | 遍历参数名和张量 |
| 梯度裁剪 | `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | 在 backward 后、step 前使用 |
| 保存权重 | `torch.save(model.state_dict(), path)` | 保存参数和持久缓冲区 |

---

## 二、张量

### 1、形状

**Tensor** 是多维数组，标量、向量、矩阵都可以用它表示。

| 数据 | 常见形状 | 含义 |
| --- | --- | --- |
| 标量 | `[]` | 单个值，如 loss |
| 向量 | `[D]` | `D` 个特征 |
| 表格 | `[N, D]` | `N` 个样本，每个 `D` 个特征 |
| 图像 | `[N, C, H, W]` | 批次、通道、高、宽 |
| 序列 | `[B, L, D]` | 批次、序列长度、特征维度 |

`dim` 指操作沿哪一根轴进行。对 `[N, D]`，`dim=0` 沿样本轴，`dim=1` 沿特征轴。

??? example "求和"
    ```python
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

    print(x.shape)                   # torch.Size([2, 3])
    print(x.sum(dim=0))               # tensor([5., 7., 9.])
    print(x.sum(dim=1))               # tensor([6., 15.])
    print(x.sum(dim=1, keepdim=True))  # tensor([[6.], [15.]])
    ```

### 2、类型

| 类型 | 常见用途 |
| --- | --- |
| `torch.float32` | 默认的浮点计算、模型参数 |
| `torch.float16` / `torch.bfloat16` | 混合精度训练、推理，需设备和算子支持 |
| `torch.float64` | 更高精度的数值计算 |
| `torch.int64`，即 `torch.long` | 类别标签、索引 |
| `torch.bool` | 掩码、条件筛选 |

```python
x = torch.tensor([1, 2, 3])  # int64
x = x.float()               # float32
y = torch.tensor([0, 1, 2], dtype=torch.long)
```

输入特征、模型参数和标签的类型要与算子要求一致。e.g. 全连接层通常接收浮点输入，交叉熵的类别下标使用 `long`。

### 3、设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3, device=device)
x = x.to("cpu")
```

`torch.cuda.is_available()` 检查当前 PyTorch 环境能否使用 CUDA；只有显卡还不够，还需要匹配的驱动和 PyTorch 安装。

模型与参与同一次计算的输入应放在兼容的设备上。训练时，特征和标签通常都要移动。

!!! tip
    `x.to(device)` 不会直接修改变量 `x`，要写成 `x = x.to(device)`。模型的 `model.to(device)` 会修改模块自身。

### 4、变形

**reshape 改形状，permute 改轴序。** e.g. 将图片从 `NCHW` 改成 `NHWC`，应该调整轴序，不能直接 reshape。

```python
x = torch.arange(24).reshape(2, 3, 4)

a = x.reshape(2, 12)       # [2, 12]
b = x.permute(0, 2, 1)     # [2, 4, 3]
c = b.reshape(2, 12)       # 必要时会复制
d = b.contiguous().view(2, 12)
```

`view` 只能在步长兼容时返回视图；`reshape` 可以返回视图，也可能复制。不确定内存布局时可以使用 `reshape`，但不要依赖它一定共享内存。

```python
x = torch.zeros(1, 3)
print(x.unsqueeze(1).shape)  # [1, 1, 3]
print(x.squeeze(0).shape)    # [3]
print(x.squeeze(1).shape)    # [1, 3]，第 1 维不是 1，不会删除
```

!!! warning
    `x.squeeze()` 会删除**所有**长度为 `1` 的维度。batch size 为 `1` 时可能连批次轴一起删掉，通常应明确指定 `dim`。

### 5、广播

逐元素运算时，从**最右侧**对齐形状。对应维度满足以下任意条件即可：

- 长度相同；
- 其中一个长度为 `1`；
- 某一边没有该维度，按 `1` 处理。

??? example "偏置"
    ```python
    x = torch.ones(2, 3)
    bias = torch.tensor([1., 2., 3.])

    print(x + bias)
    # tensor([[2., 3., 4.],
    #         [2., 3., 4.]])
    ```

    `[2, 3]` 和 `[3]` 可以广播；`[2, 3]` 和 `[2]` 不可以，因为最右侧的 `3` 与 `2` 不兼容。若想给每行加一个数，可将后者变成 `[2, 1]`。

`expand` 利用广播共享存储，`repeat` 会实际复制数据。需要修改扩展后的各个元素时，先 `clone()`。

### 6、拼接

??? example "cat 与 stack"
    ```python
    x = torch.zeros(2, 3)
    y = torch.ones(2, 3)

    print(torch.cat([x, y], dim=0).shape)    # [4, 3]
    print(torch.cat([x, y], dim=1).shape)    # [2, 6]
    print(torch.stack([x, y], dim=0).shape)  # [2, 2, 3]
    print(torch.stack([x, y], dim=1).shape)  # [2, 2, 3]
    ```

    后两者形状虽然相同，轴的含义不同：`dim=0` 时第一轴选择 x 或 y，`dim=1` 时第二轴选择 x 或 y。

### 7、乘法

```python
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.tensor([[5., 6.], [7., 8.]])

print(x * y)  # tensor([[5., 12.], [21., 32.]])
print(x @ y)  # tensor([[19., 22.], [43., 50.]])
```

- `*`：对应位置相乘，可以广播。
- `@`：矩阵乘法，内侧维度必须匹配。
- 批量矩阵乘法：`[B, M, K] @ [B, K, N] → [B, M, N]`，对每个批次分别计算。

### 8、内存

`torch.tensor(data)` 复制数据；`torch.from_numpy(arr)` 与 NumPy 数组共享存储。基本切片通常返回视图，修改它会影响原张量。

??? example "共享与复制"
    ```python
    x = torch.tensor([1., 2., 3.])
    a = x[:2]
    b = x.clone()

    a[0] = 9
    print(x)  # tensor([9., 2., 3.])
    print(b)  # tensor([1., 2., 3.])
    ```

带下划线的操作通常是**原地操作**，如 `x.add_(1)`、`x.zero_()`。它们会修改原数据，涉及共享存储或自动求导时要格外注意。

---

## 三、数据

### 1、Dataset

`Dataset` 负责按下标取样本，`DataLoader` 负责组装 batch、打乱顺序和加载数据。

```python
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


dataset = MyDataset(
    [[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
    [0, 1, 0, 1],
)
```

已经有张量时，可以直接用 `TensorDataset(features, labels)`；需要读文件或自定义预处理时，再实现自己的 Dataset。

### 2、DataLoader

```python
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for x, y in loader:
    print(x.shape, y.shape)  # [2, 2]、[2]
```

| 参数 | 含义 |
| --- | --- |
| `batch_size` | 每批样本数 |
| `shuffle` | 每轮是否打乱，训练通常开启，验证和测试通常关闭 |
| `num_workers` | 加载数据的子进程数，`0` 表示在主进程中加载 |
| `drop_last` | 是否丢弃最后不足一个 batch 的样本 |
| `pin_memory` | 使用锁页内存，常用于配合向 CUDA 传输 |
| `collate_fn` | 自定义多个样本如何组成一个 batch |

**1 epoch** = 遍历训练集一轮；**1 step** 通常指更新一次参数。普通训练中每个 batch 更新一次，梯度累积时则不一定。

!!! tip
    默认拼批会沿新轴堆叠张量，样本长度不同会报错。变长序列需要在 `collate_fn` 中做 padding，或保留为列表。Windows 使用多进程加载时，将入口放进 `if __name__ == "__main__":`。

---

## 四、模型

### 1、Module

网络继承 `nn.Module`，在 `__init__` 中定义层，在 `forward` 中描述计算。

```python
class MyModel(nn.Module):
    def __init__(self, in_features=2, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = MyModel()
logits = model(torch.randn(4, 2))
print(logits.shape)  # [4, 2]
```

调用时用 `model(x)`，这样模块的 hooks 等机制才能正常参与。可训练参数由 `model.parameters()` 交给优化器。

多个层动态组成列表时，使用 `nn.ModuleList`；需要注册自己的可训练张量时，使用 `nn.Parameter`。

### 2、常用层

| 层 | 写法 | 作用 |
| --- | --- | --- |
| 全连接 | `nn.Linear(D, H)` | 最后一维 `D → H`，计算线性变换 |
| 卷积 | `nn.Conv2d(3, 32, 3, padding=1)` | 提取局部空间特征 |
| 嵌入 | `nn.Embedding(V, D)` | 整数编号映射为 `D` 维向量 |
| 展平 | `nn.Flatten(start_dim=1)` | 保留批次轴，展平其余轴 |
| ReLU | `nn.ReLU()` | `max(0, x)` |
| Sigmoid | `nn.Sigmoid()` | 映射到 `(0, 1)` |
| GELU | `nn.GELU()` | Transformer 中常见的激活函数 |
| Dropout | `nn.Dropout(p=0.2)` | 训练时随机置零部分元素并缩放 |
| BatchNorm | `nn.BatchNorm1d(H)` | 对 `[N, H]` 按通道归一化 |
| LayerNorm | `nn.LayerNorm(H)` | 对最后的 `H` 维归一化 |

全连接层属于线性变换，**不是激活函数**。最后一层要不要加激活，取决于损失函数需要 logits 还是概率。

### 3、损失

| 任务 | 损失 | 输入与标签 |
| --- | --- | --- |
| 回归 | `nn.MSELoss()` | 预测、标签均为浮点，通常形状相同 |
| 回归 | `nn.L1Loss()` | 绝对误差 |
| 单标签多分类 | `nn.CrossEntropyLoss()` | 常见用法：logits `[N, C]`，类别下标 `[N]` |
| 二分类 / 多标签 | `nn.BCEWithLogitsLoss()` | logits 与浮点标签形状相同，标签通常为 `0/1` |

??? example "交叉熵"
    ```python
    logits = torch.tensor([[2., 0., -1.], [0., 1., 3.]])
    labels = torch.tensor([0, 2], dtype=torch.long)

    loss = nn.CrossEntropyLoss()(logits, labels)
    pred = logits.argmax(dim=1)
    print(pred)  # tensor([0, 2])
    ```

    `CrossEntropyLoss` 接收未经 softmax 的 logits。上例标签为类别下标，不需要 one-hot；如果使用概率标签，标签的形状、类型和含义也要相应改变。

    `BCEWithLogitsLoss` 同理，前面不要再接 sigmoid；需要展示概率时，再对输出做 sigmoid。

!!! warning
    回归预测为 `[N, 1]`、标签为 `[N]` 时，可能广播成 `[N, N]`，而不是逐个样本比较。计算 loss 前先核对形状。

### 4、优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
)
```

| 优化器 | 常见参数 |
| --- | --- |
| `SGD` | `lr`、`momentum`、`weight_decay` |
| `Adam` | `lr`、`betas`、`eps` |
| `AdamW` | `lr`、`betas`、`weight_decay`，权重衰减与梯度更新解耦 |

`lr` 是学习率，需要根据模型、数据和 batch size 调整。优化器负责更新参数，梯度由 `backward()` 计算。

---

## 五、求导

### 1、计算图

当输入中存在需要梯度的张量时，可求导运算会被记录到计算图中。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = x.square().sum()
loss.backward()

print(loss.item())  # 14.0
print(x.grad)       # tensor([2., 4., 6.])
```

这里 $L = \sum_i x_i^2$，所以 $\frac{\partial L}{\partial x_i} = 2x_i$。

- `requires_grad=True`：跟踪求导。
- `backward()`：从输出沿计算图反向计算。
- `.grad`：默认保留在需要梯度的叶子张量上，如模型参数。
- 非标量输出调用 `backward()` 时，需要传入匹配的外部梯度，或先归约为标量。

### 2、累积

**梯度默认累加，不会自动清零。**

??? example "重复求导"
    ```python
    x = torch.tensor(2., requires_grad=True)

    (x * 3).backward()
    print(x.grad)  # tensor(3.)

    (x * 4).backward()  # 新建一次计算图
    print(x.grad)      # tensor(7.)

    x.grad = None
    (x * 4).backward()
    print(x.grad)  # tensor(4.)
    ```

普通训练每个 batch 前清空梯度。若有意做梯度累积，则多个 batch 后再更新，并按累积方式调整 loss 的缩放。

一次反向传播后，图中的中间缓存通常会释放。下一步训练重新前向即可，不需要每次都加 `retain_graph=True`。

### 3、推理

| 操作 | 影响 |
| --- | --- |
| `model.eval()` | 切换层的行为，不关闭梯度 |
| `torch.no_grad()` | 不记录反向计算图，减少内存开销 |
| `torch.inference_mode()` | 适合纯推理，进一步减少跟踪开销 |
| `x.detach()` | 让某个张量脱离计算图，仍与原张量共享存储 |

```python
model.eval()
with torch.inference_mode():
    logits = model(torch.randn(4, 2))
    pred = logits.argmax(dim=1)
```

!!! tip
    验证和推理通常需要同时设置 `eval()` 与关闭梯度。若得到的张量还要参与后续需记录梯度的计算，优先考虑 `no_grad()`，因为 inference mode 创建的张量有额外限制。

---

## 六、训练

### 1、循环

下面沿用前面的 `MyModel` 和 `loader`。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    print(f"Epoch {epoch + 1}: loss={total_loss / total_samples:.4f}")
```

**清梯度 → 前向 → loss → 反向 → 更新。**

上面是普通单标签分类，每个 batch 的 loss 为样本平均，因此乘回 batch 大小后再累计。这样最后一个 batch 较小时，也不会被赋予和其他 batch 一样的权重。

### 2、验证

训练集用于更新参数，验证集用于选择模型和调参，测试集用于最终评估。普通验证循环不等于交叉验证；交叉验证还要轮换训练与验证的划分。

```python
@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total
```

这里同样假设使用未加权、未忽略标签的默认平均交叉熵，且 loader 非空。验证结束后若继续训练，需要重新 `model.train()`。

### 3、保存

`state_dict` 包含模型参数和持久缓冲区，如 BatchNorm 的运行统计量，不包含模型类的实现。

```python
torch.save(model.state_dict(), "model.pth")

loaded_model = MyModel().to(device)
state = torch.load("model.pth", map_location=device, weights_only=True)
loaded_model.load_state_dict(state)
loaded_model.eval()
```

加载时要先构造相同结构。`map_location` 决定权重加载到哪里，方便在 CPU 与 GPU 之间切换。

断点续训还需要保存优化器、轮数等状态；若使用调度器或混合精度，也应保存对应状态。

??? example "断点"
    ```python
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, "checkpoint.pth")

    checkpoint = torch.load(
        "checkpoint.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    ```

    从 `start_epoch` 继续循环。若要精确复现中断前的随机过程，还需保存随机数生成器等状态。

---

## 七、实例

??? example "MNIST"
    用一个两层全连接网络识别手写数字。依赖 `torch` 和 `torchvision`；保存为 Python 脚本运行，首次运行会下载 MNIST。

    训练集拆成 `54000` 个训练样本和 `6000` 个验证样本，按验证 loss 保存最佳权重，最后在测试集上评估一次。

    ```python title="mnist.py"
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms


    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)


    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        return total_loss / total


    @torch.inference_mode()
    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total


    def main():
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path("mnist_best.pth")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        # ToTensor 将像素缩放到 [0, 1]，再用 (x - 0.5) / 0.5 映射到 [-1, 1]。
        # 这里的 0.5 是选定的缩放参数，不是 MNIST 的实际均值和标准差。
        full_train = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )
        train_set, val_set = random_split(
            full_train, [54000, 6000],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        model = MNISTModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss = float("inf")

        for epoch in range(5):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)

            print(
                f"Epoch {epoch + 1}: train={train_loss:.4f}, "
                f"val={val_loss:.4f}, acc={val_acc:.2%}"
            )

        # 重新构造模型，读取验证集表现最好的权重。
        best_model = MNISTModel().to(device)
        state = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        best_model.load_state_dict(state)
        test_loss, test_acc = evaluate(
            best_model, test_loader, criterion, device
        )
        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.2%}")

        # 单张图片需要补上批次轴：[1, 28, 28] → [1, 1, 28, 28]。
        image, label = test_set[0]
        best_model.eval()
        with torch.inference_mode():
            logits = best_model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).item()
        print(f"Prediction: {prediction}, label: {label}")


    if __name__ == "__main__":
        main()
    ```

    看代码时可以沿着形状检查：`[N, 1, 28, 28] → [N, 784] → [N, 128] → [N, 10]`，标签始终是 `[N]`。

    换成自己的数据时，主要调整 Dataset、预处理、输入维度和类别数。固定种子便于复查结果，但不保证不同设备、版本下完全一致。

---
