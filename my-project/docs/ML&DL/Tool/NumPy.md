---
icon: simple/numpy
---

[TOC]

---

## 一、速查

```bash
python -m pip install numpy
```

```python
import numpy as np
```

下面各行独立查阅。示例中的 `a`、`b` 是数组，`axis=-1` 表示最后一根轴。

### 1、创建

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 从序列创建 | `np.array([1, 2, 3])` | 从列表或元组创建数组 |
| 指定类型 | `np.array([1, 2], dtype=np.float32)` | 创建 `float32` 数组 |
| 尽量复用数据 | `np.asarray(data)` | 输入已是合适数组时避免复制 |
| 全零 / 全一 | `np.zeros((2, 3))` / `np.ones((2, 3))` | 形状为 `(2, 3)` |
| 指定填充值 | `np.full((2, 3), 5)` | 所有元素为 `5` |
| 单位矩阵 | `np.eye(3)` | 形状为 `(3, 3)` |
| 等差序列 | `np.arange(0, 10, 2)` | 左闭右开，得到 `0, 2, 4, 6, 8` |
| 等距取点 | `np.linspace(0, 1, 5)` | 包含两端，共取 `5` 个点 |
| 网格坐标 | `np.meshgrid(x, y, indexing="xy")` | 根据坐标轴生成网格 |
| 参照数组 | `np.zeros_like(a)` / `np.ones_like(a)` | 默认继承形状和类型 |
| 随机生成器 | `rng = np.random.default_rng(42)` | 推荐的随机数入口 |
| 均匀随机数 | `rng.random((2, 3))` | 范围 `[0, 1)` |
| 标准正态 | `rng.standard_normal((2, 3))` | 均值 `0`、方差 `1` |
| 随机整数 | `rng.integers(0, 10, size=(2, 3))` | 范围 `[0, 10)` |

!!! tip
    浮点等差序列更适合用 `linspace`。`arange(0, 1, 0.1)` 受浮点误差影响，元素个数和末尾值可能不符合直觉。

### 2、属性

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 形状 | `a.shape` | 各轴长度组成的元组 |
| 维数 | `a.ndim` | 轴的数量 |
| 元素数 | `a.size` | 所有轴长度的乘积 |
| 数据类型 | `a.dtype` | 如 `int64`、`float32` |
| 单元素字节数 | `a.itemsize` | 每个元素占用的字节数 |
| 总字节数 | `a.nbytes` | 数组数据占用的字节数 |
| 步长 | `a.strides` | 沿各轴前进一格跨过的字节数 |
| 转置 | `a.T` | 反转全部轴；二维时交换行列 |

### 3、索引

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 取行 / 列 | `a[0]` / `a[:, 0]` | 对二维数组取第一行、第一列 |
| 切片 | `a[:, 1:3]` | 取第 `1`、`2` 列 |
| 保留维度 | `a[0:1]` | 取第一项，但保留第 `0` 轴 |
| 反向 | `a[::-1]` | 沿第一轴倒序 |
| 间隔取值 | `a[::2]` | 每隔一个取一个 |
| 指定下标 | `a[[0, 2]]` | 花式索引，取第 `0`、`2` 项 |
| 逐项坐标 | `a[[0, 1], [2, 0]]` | 取 `a[0, 2]` 和 `a[1, 0]` |
| 布尔筛选 | `a[a > 0]` | 返回所有正元素的一维数组 |
| 条件选择 | `np.where(a > 0, a, 0)` | 正数保留，其余置零 |
| 非零坐标 | `np.argwhere(a != 0)` | 每行是一组坐标 |
| 按轴取值 | `np.take(a, indices, axis=0)` | 沿指定轴取多个位置 |
| 逐行取值 | `np.take_along_axis(a, idx, axis=1)` | `idx` 与结果形状一致 |

!!! tip
    多个条件组合时使用 `&`、`|`、`~`，每个比较加括号，如 `(a > 0) & (a < 1)`；不要使用 Python 的 `and`、`or`。

### 4、变形

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 改变形状 | `a.reshape(2, -1)` | 自动推算一维，元素总数不变 |
| 展平 | `a.ravel()` | 尽可能返回视图 |
| 复制展平 | `a.flatten()` | 总是返回副本 |
| 增加维度 | `np.expand_dims(a, axis=0)` | 在第 `0` 轴插入长度 `1` |
| 简写增维 | `a[:, None]` | 在最后增加一根轴 |
| 删除单维 | `np.squeeze(a, axis=1)` | 第 `1` 轴长度必须为 `1` |
| 交换两轴 | `np.swapaxes(a, 0, 1)` | 交换指定轴 |
| 调整轴序 | `a.transpose(0, 2, 1)` | 指定新的轴顺序 |
| 移动轴 | `np.moveaxis(a, 0, -1)` | 将第 `0` 轴移到最后 |
| 广播视图 | `np.broadcast_to(a, (4, 3))` | 广播到目标形状，通常只读 |
| 逐元素重复 | `np.repeat(a, 3, axis=0)` | 每个元素沿轴重复 |
| 整体平铺 | `np.tile(a, (2, 3))` | 将整个数组平铺 |

### 5、拼接

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 沿原轴拼接 | `np.concatenate([a, b], axis=0)` | 除拼接轴外，其余形状相同 |
| 沿新轴堆叠 | `np.stack([a, b], axis=0)` | 输入数组形状必须相同 |
| 纵向拼接 | `np.vstack([a, b])` | 对二维数组沿行拼接 |
| 横向拼接 | `np.hstack([a, b])` | 对二维数组沿列拼接 |
| 组成列 | `np.column_stack([x, y])` | 将一维数组作为列 |
| 平均切分 | `np.split(a, 3, axis=0)` | 必须能够等分 |
| 非平均切分 | `np.array_split(a, 3, axis=0)` | 允许各段长度不同 |

### 6、运算

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 逐元素运算 | `a + b`、`a - b`、`a * b`、`a / b` | 形状相同或可以广播 |
| 整除 / 余数 | `a // b` / `a % b` | 逐元素计算 |
| 幂 / 开方 | `a ** 2` / `np.sqrt(a)` | 开方注意取值范围 |
| 指数 / 对数 | `np.exp(a)` / `np.log(a)` | 实数对数要求输入为正 |
| 绝对值 | `np.abs(a)` | 逐元素绝对值 |
| 截断范围 | `np.clip(a, 0, 1)` | 限制在 `[0, 1]` |
| 四舍五入 | `np.round(a, 2)` | 保留两位小数 |
| 逐元素极值 | `np.maximum(a, b)` / `np.minimum(a, b)` | 比较对应元素 |
| 矩阵乘法 | `a @ b` / `np.matmul(a, b)` | 也支持批次广播 |
| 向量点积 | `np.vdot(a, b)` | 展平后计算，复数会共轭第一个参数 |
| 外积 | `np.outer(a, b)` | 两个向量的外积 |
| 爱因斯坦求和 | `np.einsum("bnd,bd->bn", a, b)` | 按下标描述乘法和求和 |

### 7、统计

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 求和 / 均值 | `a.sum(axis=0)` / `a.mean(axis=0)` | 沿指定轴归约 |
| 保留轴 | `a.mean(axis=0, keepdims=True)` | 被归约的轴保留为长度 `1` |
| 极值 | `a.min(axis=0)` / `a.max(axis=0)` | 沿指定轴取极值 |
| 极值下标 | `a.argmin(axis=0)` / `a.argmax(axis=0)` | 返回位置 |
| 方差 / 标准差 | `a.var(ddof=0)` / `a.std(ddof=0)` | 默认除以 `N` |
| 中位数 | `np.median(a, axis=0)` | 对异常值更稳健 |
| 分位数 | `np.quantile(a, [0.25, 0.5, 0.75])` | 计算指定分位点 |
| 累加 / 累乘 | `a.cumsum(axis=0)` / `a.cumprod(axis=0)` | 沿轴累计 |
| 排序 | `np.sort(a, axis=-1)` | 返回排好序的副本 |
| 排序下标 | `np.argsort(a, axis=-1)` | 返回排序后的原下标 |
| 前 k 小 | `np.partition(a, k - 1)[:k]` | 不保证前 `k` 项内部有序 |
| 去重 | `np.unique(a)` | 默认返回排好序的唯一值 |
| 计数 | `values, counts = np.unique(a, return_counts=True)` | 唯一值及出现次数 |
| 任意 / 全部 | `(a > 0).any()` / `(a > 0).all()` | 逻辑归约 |

### 8、判断

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 逐元素比较 | `a == b` | 返回布尔数组 |
| 数组完全相同 | `np.array_equal(a, b)` | 形状和元素均相同 |
| 浮点近似相同 | `np.allclose(a, b)` | 允许相对和绝对误差 |
| 是否含 NaN | `np.isnan(a)` | 逐元素检查 |
| 是否为有限值 | `np.isfinite(a)` | 排除 `NaN` 和正负无穷 |
| 忽略 NaN 求均值 | `np.nanmean(a, axis=0)` | 全为 NaN 时会警告并返回 NaN |
| 替换异常值 | `np.nan_to_num(a)` | 替换 NaN 和正负无穷 |
| 集合包含 | `np.isin(a, values)` | 判断元素是否在集合中 |

### 9、转换

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 改类型 | `a.astype(np.float32)` | 通常返回新数组 |
| 转列表 | `a.tolist()` | 转成嵌套 Python 列表 |
| 取标量 | `a.item()` | 数组只能有一个元素 |
| 复制 | `a.copy()` | 复制数组和底层数据 |
| 写入 `.npy` | `np.save("data.npy", a)` | 保存单个数组 |
| 读取 `.npy` | `np.load("data.npy", allow_pickle=False)` | 读取单个数组 |
| 写入 `.npz` | `np.savez_compressed("data.npz", x=a, y=b)` | 保存多个命名数组并压缩 |
| 文本写入 | `np.savetxt("data.csv", a, delimiter=",")` | 保存规则二维数值表 |
| 文本读取 | `np.loadtxt("data.csv", delimiter=",")` | 读取规则数值文本 |

---

## 二、数组

### 1、形状

`ndarray` 是同类型元素组成的多维数组。标量、向量和矩阵都只是不同维数的数组。

| 数据 | 常见形状 | 含义 |
| --- | --- | --- |
| 标量 | `()` | 单个值 |
| 向量 | `(D,)` | `D` 个元素 |
| 表格 | `(N, D)` | `N` 行、`D` 列 |
| 图像 | `(H, W, C)` | 高、宽、通道 |
| 批次 | `(N, H, W, C)` | 批次、高、宽、通道 |

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)  # (2, 3)
print(a.ndim)   # 2
print(a.size)   # 6
```

数组必须是规则的长方体。各行长度不同的嵌套列表不能直接构成普通数值数组，应先补齐、拆开保存，或明确使用 `dtype=object`。

### 2、类型

| 类型 | 常见用途 |
| --- | --- |
| `np.bool_` | 掩码、条件判断 |
| `np.int32` / `np.int64` | 整数、类别编号 |
| `np.float32` | 机器学习、较省内存的数值计算 |
| `np.float64` | NumPy 默认浮点类型、一般科学计算 |
| `np.complex64` / `np.complex128` | 复数计算 |
| `np.str_` | 定长 Unicode 字符串 |

整数和浮点数混合运算时，NumPy 会根据类型提升规则决定结果类型。内存和精度敏感时，应显式指定 `dtype`。

```python
a = np.array([1, 2, 3], dtype=np.float32)
b = a.astype(np.float64)

print(a.dtype)  # float32
print(b.dtype)  # float64
```

!!! warning
    整数类型范围有限，会发生溢出。e.g. 图像常见的 `uint8` 只能表示 `0` 到 `255`，做减法或乘法前通常要先转换成更宽的整数或浮点类型。

### 3、轴

`axis` 指被归约或操作的轴。二维数组形状为 `(行, 列)`：

- `axis=0`：消去行轴，对每一列计算；
- `axis=1`：消去列轴，对每一行计算。

??? example "均值"
    ```python
    a = np.array([[1., 2., 3.], [4., 5., 6.]])

    print(a.mean())        # 3.5
    print(a.mean(axis=0))  # [2.5 3.5 4.5]，每列均值
    print(a.mean(axis=1))  # [2. 5.]，每行均值
    ```

    `keepdims=True` 会把被消去的轴保留为长度 `1`，便于后续广播。

### 4、副本

赋值、切片、花式索引的内存行为不同：

| 操作 | 结果 |
| --- | --- |
| `b = a` | 同一个数组的另一个变量名 |
| `b = a[1:3]` | 基本切片通常是视图，共享数据 |
| `b = a[[1, 3]]` | 花式索引返回副本 |
| `b = a[a > 0]` | 布尔索引返回副本 |
| `b = a.copy()` | 显式创建独立副本 |
| `b = a.reshape(...)` | 可能是视图，也可能是副本 |

??? example "视图"
    ```python
    a = np.array([1, 2, 3, 4])
    view = a[1:3]
    copy = a[[1, 2]]

    view[0] = 99
    copy[1] = 88

    print(a)     # [ 1 99  3  4]
    print(copy)  # [ 2 88]
    ```

    可以用 `np.shares_memory(a, b)` 检查两个数组是否确定共享内存。准备修改数据且不想影响原数组时，直接使用 `copy()`。

---

## 三、索引

### 1、切片

```python
a = np.arange(12).reshape(3, 4)

print(a[1, 2])    # 第 1 行、第 2 列
print(a[0])       # 第一行，形状 (4,)
print(a[:, 0])    # 第一列，形状 (3,)
print(a[:2, 1:])  # 前两行、第 1 列到末尾
print(a[..., -1]) # 最后一轴的最后一个元素
```

整数索引会消去对应轴，切片会保留轴。`a[0]` 的形状是 `(4,)`，`a[0:1]` 的形状是 `(1, 4)`。

### 2、花式

整数数组可以一次取多个位置。

```python
a = np.arange(12).reshape(3, 4)

rows = a[[2, 0]]
items = a[[0, 2], [1, 3]]  # [a[0, 1], a[2, 3]]
block = a[np.ix_([0, 2], [1, 3])]

print(rows.shape)   # (2, 4)
print(items.shape)  # (2,)
print(block.shape)  # (2, 2)
```

`a[[0, 2], [1, 3]]` 是逐对取坐标；想取行列组合形成的矩形区域，使用 `np.ix_`。

### 3、掩码

```python
a = np.array([-2, -1, 0, 1, 2])
mask = (a >= 0) & (a <= 1)

print(a[mask])               # [0 1]
print(np.where(mask, a, -1)) # [-1 -1  0  1 -1]
print(np.where(mask)[0])     # [2 3]
```

- `a[mask]`：筛选元素，结果通常变成一维；
- `np.where(condition, x, y)`：在相同位置二选一，保留广播后的形状；
- `np.where(condition)`：返回各轴下标组成的元组，查坐标时 `argwhere` 更直观。

### 4、赋值

```python
a = np.arange(6)
a[a % 2 == 0] = -1
print(a)  # [-1  1 -1  3 -1  5]
```

布尔掩码赋值会直接修改原数组。右侧可以是标量，也可以是与选中元素数量匹配的数组。

!!! warning
    链式索引容易产生临时副本。要修改原数组，优先写成一次索引，如 `a[rows, cols] = value`。

---

## 四、形状

### 1、变形

```python
a = np.arange(24).reshape(2, 3, 4)

b = a.reshape(2, -1)       # (2, 12)
c = a.transpose(0, 2, 1)   # (2, 4, 3)
d = np.moveaxis(a, 0, -1)  # (3, 4, 2)
```

**reshape 改形状，transpose 改轴序。** e.g. 图像从 `HWC` 改为 `CHW` 应使用 `a.transpose(2, 0, 1)`，不能直接 reshape。

`-1` 最多出现一次，由 NumPy 根据元素总数自动推算。`reshape` 不保证一定复制或共享数据，不应依赖其内存行为。

### 2、增减

```python
a = np.zeros((3, 4))

print(a[None, ...].shape)                     # (1, 3, 4)
print(a[..., None].shape)                     # (3, 4, 1)
print(np.expand_dims(a, axis=(0, -1)).shape)  # (1, 3, 4, 1)
```

`None` 与 `np.newaxis` 等价。`squeeze` 只删除长度为 `1` 的轴：

```python
x = np.zeros((1, 3, 1))
print(np.squeeze(x).shape)          # (3,)
print(np.squeeze(x, axis=0).shape)  # (3, 1)
```

!!! warning
    直接 `squeeze()` 会删除所有长度为 `1` 的轴。若某一轴有固定含义，明确指定 `axis`。

### 3、拼接

??? example "拼接与堆叠"
    ```python
    a = np.zeros((2, 3))
    b = np.ones((2, 3))

    print(np.concatenate([a, b], axis=0).shape)  # (4, 3)
    print(np.concatenate([a, b], axis=1).shape)  # (2, 6)
    print(np.stack([a, b], axis=0).shape)        # (2, 2, 3)
    print(np.stack([a, b], axis=1).shape)        # (2, 2, 3)
    ```

    `concatenate` 沿已有轴接起来；`stack` 创建一根新轴。后两项形状虽然相同，新轴的位置和含义并不相同。

### 4、广播

两个数组逐元素运算时，从最右侧开始比较形状。对应维度满足以下任意条件即可：

- 长度相同；
- 其中一个长度为 `1`；
- 某一边没有该维度，按 `1` 处理。

??? example "标准化"
    ```python
    x = np.array([[1., 10.], [3., 20.], [5., 30.]])
    mean = x.mean(axis=0, keepdims=True)  # (1, 2)
    std = x.std(axis=0, keepdims=True)    # (1, 2)
    z = (x - mean) / std                  # (3, 2)
    ```

    `(3, 2)` 与 `(1, 2)` 可以广播，均值和标准差会作用于每一行。

    `(3, 2)` 与 `(3,)` 不能广播，因为最右侧的 `2` 与 `3` 不兼容。若要给每行加一个值，可将后者变为 `(3, 1)`。

广播通常不会真的复制较小数组，但结果本身仍可能很大。e.g. `(10000, 1, 3)` 与 `(1, 10000, 3)` 相减会产生 `(10000, 10000, 3)` 的结果，应留意内存。

---

## 五、运算

### 1、向量化

NumPy 的核心写法是让运算作用于整个数组。

```python
a = np.arange(1000000)
b = a * 2 + 1
```

这比逐个元素执行 Python 循环更简洁，通常也更快。可以广播或用 ufunc 表达的操作，优先写成数组运算。

### 2、ufunc

`np.add`、`np.sqrt`、`np.exp` 等是通用函数，即 **ufunc**。它们支持广播、类型处理和按轴归约。

```python
a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

print(np.add(a, b))              # [5. 7. 9.]
print(np.add.reduce(a))          # 6.0
print(np.add.accumulate(a))      # [1. 3. 6.]
print(np.add.outer(a, b).shape)  # (3, 3)
```

遇到除零、溢出或非法值时，可以临时设置处理方式：

```python
with np.errstate(divide="ignore", invalid="ignore"):
    result = np.log(np.array([1., 0., -1.]))
```

### 3、排序

```python
a = np.array([30, 10, 20, 10])

print(np.sort(a))                        # [10 10 20 30]
print(np.argsort(a))                     # [1 3 2 0]
print(np.unique(a))                      # [10 20 30]
print(np.unique(a, return_counts=True))  # (values, counts)
```

若只需要第 `k` 小或前 `k` 小，不必完整排序：

```python
a = np.array([8, 2, 5, 1, 9, 3])
k = 3
smallest = np.partition(a, k - 1)[:k]
print(np.sort(smallest))  # [1 2 3]
```

### 4、缺失值

浮点数组常用 `NaN` 表示缺失值。

```python
a = np.array([1., np.nan, 3., np.inf])

print(np.isnan(a))
print(np.isfinite(a))
print(np.nanmean(a[:3]))  # 2.0
clean = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
```

`NaN` 不等于自身，因此不要用 `a == np.nan` 判断。整数数组不能直接保存 `NaN`，需要转为浮点，或另建布尔掩码记录缺失位置。

---

## 六、随机

新代码推荐创建独立的 `Generator`：

```python
rng = np.random.default_rng(42)
```

| 操作 | 写法 |
| --- | --- |
| `[0, 1)` 均匀分布 | `rng.random((2, 3))` |
| `[low, high)` 均匀分布 | `rng.uniform(low, high, size)` |
| 标准正态分布 | `rng.standard_normal(size)` |
| 一般正态分布 | `rng.normal(mean, std, size)` |
| 随机整数 | `rng.integers(low, high, size)` |
| 伯努利试验 | `rng.binomial(1, p, size)` |
| 随机选择 | `rng.choice(a, size=3, replace=False)` |
| 原地打乱 | `rng.shuffle(a)` |
| 返回乱序副本 | `rng.permutation(a)` |

??? example "划分数据"
    ```python
    rng = np.random.default_rng(42)
    indices = rng.permutation(100)

    train_idx = indices[:80]
    val_idx = indices[80:90]
    test_idx = indices[90:]
    ```

    相同的种子和调用顺序通常产生相同序列，便于复查实验。不要在函数内部反复使用同一个种子重新创建生成器，否则每次都可能得到完全相同的样本。

---

## 七、线代

### 1、乘法

```python
a = np.array([[1., 2.], [3., 4.]])
b = np.array([[5., 6.], [7., 8.]])

print(a * b)  # 逐元素乘法
print(a @ b)  # 矩阵乘法
```

`matmul` 对最后两维做矩阵乘法，前面的维度作为批次维并参与广播。

### 2、方程

线性方程 $Ax=b$ 使用 `solve`：

```python
A = np.array([[3., 1.], [1., 2.]])
b = np.array([9., 8.])
x = np.linalg.solve(A, b)

print(x)                       # [2. 3.]
print(np.allclose(A @ x, b))   # True
```

已知右侧向量时，直接求解比先计算 `np.linalg.inv(A) @ b` 更准确、更高效。

### 3、分解

| 操作 | 写法 | 说明 |
| --- | --- | --- |
| 范数 | `np.linalg.norm(a)` | 默认计算二维 Frobenius 范数 |
| 行列式 | `np.linalg.det(a)` | 方阵行列式 |
| 矩阵秩 | `np.linalg.matrix_rank(a)` | 数值秩 |
| 特征分解 | `values, vectors = np.linalg.eig(a)` | 一般方阵 |
| 对称特征分解 | `values, vectors = np.linalg.eigh(a)` | 实对称或复 Hermitian 矩阵 |
| 奇异值分解 | `u, s, vh = np.linalg.svd(a, full_matrices=False)` | $A=U\Sigma V^H$ |
| 最小二乘 | `coef, *_ = np.linalg.lstsq(X, y, rcond=None)` | 求过定或欠定系统的解 |

对实对称矩阵优先使用 `eigh`，它利用矩阵结构，返回实特征值并且通常更稳定。

---

## 八、读写

### 1、二进制

NumPy 自有格式会保留形状和数据类型。

```python
a = np.arange(12).reshape(3, 4)

np.save("array.npy", a)
loaded = np.load("array.npy", allow_pickle=False)
```

多个数组可以存进同一个文件：

```python
np.savez_compressed(
    "dataset.npz",
    features=a,
    labels=np.array([0, 1, 0]),
)

with np.load("dataset.npz", allow_pickle=False) as data:
    features = data["features"]
    labels = data["labels"]
```

### 2、文本

```python
np.savetxt("array.csv", a, delimiter=",", fmt="%d")
loaded = np.loadtxt("array.csv", delimiter=",", dtype=np.int64)
```

- `loadtxt`：适合字段规则、没有缺失值的纯数值文本；
- `genfromtxt`：可以处理缺失值，但规则较复杂；
- 表头、混合类型、日期等表格数据通常更适合使用 pandas。

### 3、打印

```python
np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=100,
    threshold=1000,
)
```

`precision` 控制显示精度，不会改变数组中的真实数值；`suppress=True` 尽量不用科学计数法显示较小数字。

---

## 九、实例

??? example "标准化"
    对每个特征分别做标准化，并处理标准差为 `0` 的常量列。

    ```python
    import numpy as np

    x = np.array([
        [1., 10., 5.],
        [3., 20., 5.],
        [5., 30., 5.],
    ])

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    safe_std = np.where(std == 0, 1, std)
    z = (x - mean) / safe_std

    print(z.mean(axis=0))
    print(z.std(axis=0))
    ```

    常量列减去均值后全为 `0`，令其除数为 `1` 可以避免除零。真实机器学习任务中，均值和标准差只用训练集计算，再应用到验证集和测试集。

??? example "距离"
    计算每个点到每个中心的平方欧氏距离，并找到最近中心。

    ```python
    import numpy as np

    points = np.array([[0., 0.], [2., 1.], [5., 4.]])  # (N, D)
    centers = np.array([[0., 1.], [4., 4.]])           # (K, D)

    diff = points[:, None, :] - centers[None, :, :]    # (N, K, D)
    distance2 = np.sum(diff ** 2, axis=-1)              # (N, K)
    nearest = np.argmin(distance2, axis=1)              # (N,)

    print(distance2)
    print(nearest)
    ```

    这里两次增加维度，让 `(N, 1, D)` 与 `(1, K, D)` 广播成 `(N, K, D)`。当 `N`、`K` 很大时，中间数组也会很大，需要分块计算。

??? example "回归"
    用最小二乘拟合 $y=wx+b$。

    ```python
    import numpy as np

    x = np.array([0., 1., 2., 3., 4.])
    y = np.array([1.1, 2.9, 5.2, 7.0, 9.1])

    design = np.column_stack([x, np.ones_like(x)])
    w, b = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.array([w, b])
    mse = np.mean((prediction - y) ** 2)

    print(f"w={w:.3f}, b={b:.3f}, mse={mse:.4f}")
    ```

    `column_stack` 将输入和常数 `1` 组成设计矩阵；`lstsq` 直接求最小二乘解，不需要手动计算逆矩阵。

---
