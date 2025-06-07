---
icon: material/math-compass
---

[TOC]

---

## 一、安装导入

```python
!pip install matplotlib
import matplotlib
```

---

## 二、`PyPlot`

- `plt()` 函数从点绘制直线
- 参数一是 $x$ 的一串坐标，参数二是 $y$ 的一串坐标

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 3])
y = np.array([4, 10])

plt.plot(x, y)
plt.show()
```

```python
# 无线绘图
plt.plot(x, y, "o")
plt.show()
```

```python
# 默认 x 点为 0、1、2、3、4
```

---

## 三、标记

### 1、点标记 `marker`

```python
plt.plot(x, y, marker="o")
plt.show()
```

??? success "标记样式"
    | marker                 | description                   |
    | ---------------------- | ----------------------------- |
    | `"."`                  | point                         |
    | `","`                  | pixel                         |
    | `"o"`                  | circle                        |
    | `"v"`                  | triangle_down                 |
    | `"^"`                  | triangle_up                   |
    | `"<"`                  | triangle_left                 |
    | `">"`                  | triangle_right                |
    | `"1"`                  | tri_down                      |
    | `"2"`                  | tri_up                        |
    | `"3"`                  | tri_left                      |
    | `"4"`                  | tri_right                     |
    | `"8"`                  | octagon                       |
    | `"s"`                  | square                        |
    | `"p"`                  | pentagon                      |
    | `"P"`                  | plus (filled)                 |
    | `"*"`                  | star                          |
    | `"h"`                  | hexagon1                      |
    | `"H"`                  | hexagon2                      |
    | `"+"`                  | plus                          |
    | `"x"`                  | x                             |
    | `"X"`                  | x (filled)                    |
    | `"D"`                  | diamond                       |
    | `"d"`                  | thin_diamond                  |
    | `"|"`                  | vline                         |
    | `"_"`                  | hline                         |
    | `0` (`TICKLEFT`)       | tickleft                      |
    | `1` (`TICKRIGHT`)      | tickright                     |
    | `2` (`TICKUP`)         | tickup                        |
    | `3` (`TICKDOWN`)       | tickdown                      |
    | `4` (`CARETLEFT`)      | caretleft                     |
    | `5` (`CARETRIGHT`)     | caretright                    |
    | `6` (`CARETUP`)        | caretup                       |
    | `7` (`CARETDOWN`)      | caretdown                     |
    | `8` (`CARETLEFTBASE`)  | caretleft (centered at base)  |
    | `9` (`CARETRIGHTBASE`) | caretright (centered at base) |
    | `10` (`CARETUPBASE`)   | caretup (centered at base)    |
    | `11` (`CARETDOWNBASE`) | caretdown (centered at base)  |

### 2、线标记 `linestyle`

| Line Syntax | description   |
| ----------- | ------------- |
| `'-'`       | Solid line    |
| `':'`       | Dotted line   |
| `'--'`      | Dashed line   |
| `'-.'`      | Dash-dot line |

### 3、线颜色 `color`

| color | description |
| ----- | ----------- |
| b     | blue        |
| g     | green       |
| r     | red         |
| c     | cyan        |
| m     | magenta     |
| y     | yellow      |
| k     | black       |
| w     | white       |

### 4、格式化字符串

```python
plt.plot(x, y, "marker | line | color")
plt.plot(x, y, "o:r")  # 红色虚线
plt.show()
```

### 5、点大小 `ms`

- `markersize` 简称 `ms`

### 6、点颜色 `mec/mfc`

- `markeredgecolor` 简称 `mec`
- `markerfacecolor` 简称 `mfc`

!!! info
    颜色都可以用十六进制

---

## 四、线

### 1、线宽 `linewidth`

- `linewidth` 简称 `lw`

### 2、多行

```python
y1 = np.array([2, 10, 23, 25])
y2 = np.array([3, 6, 34, 23])

plt.plot(y1, linestyle="-.")
plt.plot(y2, linestyle="-.")

plt.show()
```

---

## 五、标签

### 1、横纵坐标标题

```python
plt.title("report")  # 图表标题
plt.xlabel("date")  # 横坐标
plt.ylabel("price")  # 纵坐标

plt.show()
```

### 2、图表标题定位

```python
plt.title("report", loc="left")  # 图表标题居左
```

---

## 六、网格线

- 同样可以设置 `linewidth`、`linestyle`、`color` 等参数

```python
plt.grid()
plt.grid(axis="y")
plt.grid(axis="x")
plt.grid(axis="both")
```

---

## 七、`subplot()`

```python
# Subplot 1
x = np.array([1, 2, 4, 3, 2])
y = np.array([1, 2, 3, 4, 5])
plt.subplot(1, 2, 1)  # 一行两列 第一个子图
plt.title("Plot 1")
plt.plot(x, y)

# Subplot 2
x = np.array([1, 2, 3, 3, 2])
y = np.array([1, 2, 3, 4, 5])
plt.subplot(1, 2, 2)  # 一行两列 第二个子图
plt.title("Plot 2")
plt.plot(x, y)

plt.suptitle("Plot suptitle")  # 总标题
plt.show()
```

---

## 八、散点图 `scatter()`

- 把 `plt.plot` 一句改成 `plt.scatter` 即可

### 1、多组

```python
# 第一组散点
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
plt.scatter(x, y, color='hotpink')

# 第二组散点
x = np.array([2, 2, 8, 1, 15, 8, 12, 9, 7, 3, 11, 4, 7, 14, 12])
y = np.array([100, 105, 84, 105, 90, 99, 90, 95, 94, 100, 79, 112, 91, 80, 85])
plt.scatter(x, y, color='#88c999')

plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
```

### 2、单独上色

- 不能使用 `color` 参数，只能用 `c` 参数为每个点设置特定颜色

```python
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
colors = np.array([
    'red', 'green', 'blue', 'yellow', 'pink', 'black', 'orange',
    'purple', 'beige', 'brown', 'gray', 'cyan', 'magenta'
])

plt.scatter(x, y, c=colors)
plt.title("Muti-color plot")
plt.show()
```

### 3、颜色图

- 可以使用带有颜色图值的关键字参数 `cmap` 指定颜色图。在本例中为 `'viridis'`，它是 Matplotlib 中可用的**内置颜色图**之一
- 此外，必须创建一个包含值（从 0 到 100）的数组，散点图中的每个点都有一个值

```python
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

# 每个点对应的颜色值（数值），用于映射到 colormap
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

# 使用 colormap 进行颜色映射
plt.scatter(x, y, c=colors, cmap='viridis')

# 添加颜色条
plt.colorbar()

plt.title("color-bar")
plt.show()
```

### 4、大小、透明度

- 大小用 `s` 参数，同一样要开一个 `size` 数组，确保大小和 `x,y` 的数组长度相同
- 透明度用 `alpha` 参数

```python
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
sizes = np.random.randint(100, size=x.size)

plt.scatter(x, y, alpha=0.3, s=sizes)

plt.show()
```

---

## 九、柱状图 `bar()`

- 把 `plt.plot` 一句改成 `plt.bar` 即可
- `barh()` 函数绘制水平柱状图
- 传入参数仍然可以有 `color`、`width`、`height`

```python
x = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

plt.bar(x, y)

plt.show()
```

---

## 十、直方图 `hist()`

```python
x = np.random.normal(0, 1, 100) # 正态分布，期望为0，方差为1，生成100个
print(x)

plt.hist(x)
plt.show()
```

---

## 十一、饼图 `pie()`

- 传入标题 `labels`
- 起始角度 `startangle`，默认从 $x$ 轴开始

```python
y = np.array([35, 25, 25, 15])
nylabels = ["Apples", "Bananas", "Cherries", "Pears"]

plt.pie(y, labels=nylabels)

plt.show()
```

- 使用 `explode` 可以使一块饼突出

```python
y = np.array([35, 25, 25, 15])
n_ylabels = ["Apples", "Bananas", "Cherries", "Pears"]
n_explode = [0, 0, 0.2, 0.3]

plt.pie(y, labels=n_ylabels, explode=n_explode, shadow=True) # 可以传入阴影参数
plt.legend() # 图例

plt.show()
```

---



