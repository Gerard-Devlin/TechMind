---
icon: simple/numpy
---


[TOC]

---

## 一、安装导入

- 在 Python 中，我们有列表，可以达到数组的目的，但它们的处理速度很慢。
- NumPy 旨在提供一个数组对象，它比传统的 Python 列表快 50 倍。
- NumPy 中的数组对象被称为 `ndarray`，它提供了很多支持性的函数，使得使用 `ndarray` 非常容易。
- 数组在数据科学中使用得非常频繁，速度和资源都非常重要。

```python
!pip install numpy
import numpy as np
```

---

## 二、数组

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)
print(type(arr))  # <class 'numpy.ndarray'>
```

要创建 `ndarray`，我们可以将列表、元组或任何类似数组的对象传递给 `array()` 方法，然后它将被转换为 `ndarray`

```python
# 使用元组创建 NumPy 数组
arr = np.array((1, 2, 3, 4, 5))

print(arr)
```



---

## 三、数据类型

---

## 四、形状

---

## 五、操作

---

