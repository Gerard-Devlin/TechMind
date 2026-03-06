[TOC]

---

## 一、基础

### 1、概念

- 模式（schema）：定义属性/列、数据类型、约束。也叫元数据（metadata）
- 实例（instance）：满足该模式的数据集合
- 属性/列（attribute）：属性是唯一的，每个属性必须是原子值（**第一范式**）
- 元组/行（tuple）


??? bug "为什么不是一个关系"

    - 属性不是**原子类型**
    
    | first text | last text | addr address          |
    | ---------- | --------- | --------------------- |
    | wei        | jones     | (84, 'Maple', 54704)  |
    | apurva     | lee       | (22, 'High', 76425)   |
    | sara       | manning   | (75, 'Hearst', 94720) |
    
    - 属性必须**唯一**
    
    | num integer | street text | num integer |
    | ----------- | ----------- | ----------- |
    | 84          | Maple Ave   | 54704       |
    | 22          | High Street | 76425       |
    | 75          | Hearst Ave  | 94720       |

### 2、优缺点

| 优点                                      | 缺点      |
          |-----------------------------------------|---------|
          | 声明式语言：用户只需要说明**想要什么结果**，不需要说明**如何计算结果** | 不是图灵完备  |
          | 功能丰富                                    | 实现效率不一致 |
          | 被几乎所有**关系数据库管理系统**支持                    |         |

### 3、子语言

| 类型                                               | 作用           | 操作对象 |
| -------------------------------------------------- | -------------- | -------- |
| **数据定义语言（DDL,Data Definition Language）**   | 定义数据库结构 | schema   |
| **数据操作语言（DML,Data Manipulation Language）** | 操作数据       | data     |

!!! example "区别"

    DDL定义了：
    
    - 表结构
    - 数据类型
    - 主键
    
    ```sql
    CREATE TABLE Sailors (
      sid INTEGER,
      sname CHAR(20),
      rating INTEGER,
      age FLOAT,
      PRIMARY KEY (sid)
    );
    ```
    
    DML用于增删查改数据：
    
    ```sql
    SELECT *
    FROM Sailors
    WHERE age = 27;
    
    INSERT INTO Sailors
    VALUES (1, 'Fred', 7, 22);
    
    UPDATE Sailors
    SET rating = 8
    WHERE sid = 1;
    
    DELETE FROM Sailors
    WHERE sid = 1;
    ```

---

## 二、主键/外键

### 1、主键

**主键（Primary Key）**是**表的唯一标识**，有些表**一列无法唯一标识一行**，必须**多列组合** 才能唯一。

一个 sailor 可以**预订多条船**。所以 sid 不唯一；一条船可以被**不同人预订**。同一天会有**很多预订记录**。

### 2、外键

**外键（Foreign Key）**是一种**约束**，用于在两个表之间建立关联关系，并保证**参照完整性**。

| 特性           | 含义                         |
| -------------- | ---------------------------- |
| 外键不唯一     | 可以重复，因为表示多对一关系 |
| 外键不一定同名 | 外键列名不必与主键列名相同   |

```sql
CREATE TABLE Sailors (
  sid INTEGER,
  sname CHAR(20),
  rating INTEGER,
  age FLOAT,
  PRIMARY KEY (sid)
);
```

```sql
CREATE TABLE Reserves (
  sid INTEGER,
  bid INTEGER,
  day DATE,
  PRIMARY KEY (sid, bid, day),
  FOREIGN KEY (sid2) REFERENCES Sailors
);
```

表示：`Reserves.sid2 → Sailors.sid` （外键可以名字不一样）

| sid  | bid  | day  |
| ---- | ---- | ---- |
| 1    | 102  | 9/12 |
| 2    | 102  | 9/13 |

| sid  | sname |
| ---- | ----- |
| 1    | Fred  |
| 2    | Jim   |

如果插入：

| sid  | bid  | day  |
| ---- | ---- | ---- |
| 10   | 102  | 9/12 |

但 Sailors 表里 **没有 sid=10**。

数据库会报错：

```
Foreign key constraint violation
```

---

## 三、单表查询

!!! dangerous
	SQL查询结果顺序是不确定的

**执行顺序：**

```mermaid
flowchart LR
    A[FROM<br/>读取表] --> B[WHERE<br/>过滤行]
    B --> C[GROUP BY<br/>分组]
    C --> D[HAVING<br/>过滤组]
    D --> E[SELECT<br/>选择输出列]
    E --> F[ORDER BY<br/>排序]
    F --> G[LIMIT<br/>限制结果数量]
```

```sqlite
SELECT [DISTINCT] <column expression list>
FROM <single table>
[WHERE <predicate>]
[GROUP BY <column list>
[HAVING <predicate>]]
[ORDER BY <column list>]
[LIMIT <integer>];
```

### 1、`FROM`

表与别名（Alias）

```sql
SELECT S.name, S.gpa
FROM Students AS S
WHERE S.dept = 'CS';
```

作用：

- `S` 是 **表别名**
- 方便引用列：

```
S.name
S.gpa
```

### 2、`WHERE`

```sql
SELECT S.name, S.gpa
FROM Students S
WHERE S.dept = 'CS';
```

作用：删除不满足条件的行

```sql
SELECT *
FROM Sailors
WHERE sname LIKE 'B_%';
```

| 符号 | 含义         |
| ---- | ------------ |
| =    | 相等         |
| <>   | 不等         |
| LIKE | 模式匹配     |
| %    | 任意长度字符 |
| _    | 单个字符     |

#### （1）NULL

NULL 表示未知值。SQL 中：$x = NULL → NULL$

WHERE 只保留 TRUE，**FALSE 和 NULL 都会被过滤。**

判断 NULL 必须使用：

```sql
IS NULL
IS NOT NULL
```

```sql
SELECT *
FROM Person
WHERE age IS NULL;
```

!!!tip "聚合函数与 `NULL`"

    聚合函数 `SUM/AVG/MIN/MAX` 会忽略 NULL
    
    - `COUNT(column)` 只统计非 NULL
    
    - `COUNT(*)` 统计所有行

### 3、`GROUP BY`

```sql
SELECT S.dept, AVG(S.gpa)
FROM Students S
GROUP BY S.dept;
```

作用：按列分组，并在每组上计算聚合函数

| dept | avg_gpa |
| ---- | ------- |
| CS   | 3.7     |
| Math | 3.3     |

!!! tip "重要规则"
    如果使用 `GROUP BY X`：


    SELECT 中只能出现
       
    1. X
    
    2. 聚合函数
    
    否则 SQL 不合法。

### 4、`HAVING`

```sql
SELECT S.dept, AVG(S.gpa)
FROM Students S
GROUP BY S.dept
HAVING COUNT(*) > 2;
```

作用：删除不满足条件的组

| 子句   | 作用                  |
| ------ | --------------------- |
| WHERE  | 过滤行 → 普通列条件   |
| HAVING | 过滤组 → 聚合函数条件 |

### 5、`SELECT`

```sql
SELECT AVG(S.gpa)
FROM Students S
WHERE S.dept = 'CS';
```

作用：只输出 SELECT 中的列

!!! note "常见聚合函数"

    | 函数  | 作用 |
    | ----- | ---- |
    | COUNT | 数量 |
    | AVG   | 平均 |
    | SUM   | 求和 |
    | MAX   | 最大 |
    | MIN   | 最小 |

### 6、`DISTINCT`

```sql
SELECT DISTINCT S.name, S.gpa
FROM Students S
WHERE S.dept = 'CS';
```

作用：删除重复行

### 7、`ORDER BY`

```sql
SELECT S.name, S.gpa, S.age*2 AS a2
FROM Students S
WHERE S.dept = 'CS'
ORDER BY S.gpa, S.name, a2;
```

排序规则：

1. 先按 `gpa`
2. 若相同按 `name`
3. 再按 `a2`

计算列：`S.age*2 AS a2`

表示：计算 age*2 并命名为 a2

!!! tip "升序 / 降序"

    ```sql
    ORDER BY S.gpa DESC, S.name ASC;
    ```
    
    | 关键字 | 含义         |
    | ------ | ------------ |
    | ASC    | 升序（默认） |
    | DESC   | 降序         |

### 8、`LIMIT`

```sql
SELECT S.name, S.gpa
FROM Students S
WHERE S.dept = 'CS'
ORDER BY S.gpa DESC
LIMIT 3;
```

作用：只显示 GPA 最高的 3 个学生

---

## 四、多表查询

### 1、`CROSS JOIN`

`CROSS JOIN` 会将两个表的每一行组合。

如果：courses (3 rows)，enrollment (2 rows)

结果：3 × 2 = 6 rows

```sql
SELECT *
FROM courses, enrollment;
```

### 2、`INNER JOIN`

```sql
SELECT S.sid
FROM Sailors S, Reserves R
WHERE S.sid = R.sid; -- 找出预订过船的水手
```

先做笛卡尔积得到一共 $S\times R$ 行，然后保留符合要求的行

这其实就是 **INNER JOIN**，等价写法：

```sql
SELECT S.sid
FROM Sailors S
JOIN Reserves R
ON S.sid = R.sid;
```

!!! danger "NATURAL JOIN"

	`NATURAL JOIN` = `INNER JOIN` + **自动匹配同名列**
	
	但是工程中表的结构变化之后使用 `NATURAL JOIN` 就是很危险的

---

### 3、`OUTER JOIN`

**Outer Join（外连接）**在普通连接基础上，**保留未匹配的行**。普通 `INNER JOIN` 只保留 **匹配成功的行**。

Outer Join 有三种：

| 类型             | 含义           |
| ---------------- | -------------- |
| LEFT OUTER JOIN  | 保留左表所有行 |
| RIGHT OUTER JOIN | 保留右表所有行 |
| FULL OUTER JOIN  | 保留两边所有行 |

#### （1）LEFT OUTER JOIN

```sql
SELECT S.sid, R.bid
FROM Sailors S
LEFT JOIN Reserves R
ON S.sid = R.sid;
```

含义：

- 所有 `Sailors` 都会出现
- 如果没有预订船，则 `R.bid = NULL`


!!! example

    Sailors
    
    | sid  | sname |
    | ---- | ----- |
    | 1    | Bob   |
    | 2    | Alice |
    | 3    | Tom   |
    
    Reserves
    
    | sid  | bid  |
    | ---- | ---- |
    | 1    | 101  |
    
    结果：
    
    | sid  | bid  |
    | ---- | ---- |
    | 1    | 101  |
    | 2    | NULL |
    | 3    | NULL |

#### （2）RIGHT OUTER JOIN

```sql
SELECT S.sid, R.bid
FROM Sailors S
RIGHT JOIN Reserves R
ON S.sid = R.sid;
```

含义：

- 保留 **Reserves** 表所有行
- Sailors 没有匹配的部分为 NULL

#### （3）FULL OUTER JOIN

```sql
SELECT S.sid, R.bid
FROM Sailors S
FULL JOIN Reserves R
ON S.sid = R.sid;
```

含义：

- 保留 **两张表的所有行**
- 没有匹配的部分填 NULL

---

### 4、`SELF JOIN`

就是同一张表用两次，但是必须用**不同别名**，否则 SQL 分不清

```sql
SELECT S1.sname, S2.sname
FROM Sailors S1, Sailors S2
WHERE S1.age = S2.age
AND S1.sid < S2.sid;
```

----

### 5、`UNION/UNION ALL`

- `UNION` 会做 **去重 (DISTINCT)**。
- `UNION ALL` **不会去重**，只是简单拼接结果。

???+ example

    Students_CS
    
    | name  |
    | ----- |
    | Alice |
    | Bob   |
    
    Students_Math
    
    | name |
    | ---- |
    | Bob  |
    | Tom  |
    
    `UNION` 结果：
    
    | name  |
    | ----- |
    | Alice |
    | Bob   |
    | Tom   |
    
    `UNION ALL` 结果：
    
    | name  |
    | ----- |
    | Alice |
    | Bob   |
    | Bob   |
    | Tom   |


| 特性     | UNION    | UNION ALL |
| -------- | -------- | --------- |
| 去重     | 会       | 不会      |
| 执行速度 | 较慢     | 更快      |
| 本质     | DISTINCT | CONCAT    |

数据库需要 **排序或哈希去重**，所以更慢。

---

### 6、`INTERSECT`

问题：找既订过 red 又订过 green 的 sailor

❌逻辑：

```sql
WHERE B.color='red' AND B.color='green'
```

这是不可能的：**一条船不可能同时是 red 和 green**，所以要换思路。

正确写法：

```sql
SELECT R.sid
FROM Boats B, Reserves R
WHERE R.bid = B.bid
AND B.color = 'red'

INTERSECT

SELECT R.sid
FROM Boats B, Reserves R
WHERE R.bid = B.bid
AND B.color = 'green';
```

---

## 五、子查询

### 1、`IN`

```sql
-- 找预订过 boat 102 的 sailor
SELECT S.sname
FROM Sailors S
WHERE S.sid IN (
    SELECT R.sid
    FROM Reserves R
    WHERE R.bid = 102
);
```

### 2、`NOT IN`

```sql
-- 找没有预订 boat 103 的 sailors
SELECT S.sname
FROM Sailors S
WHERE S.sid NOT IN (
    SELECT R.sid
    FROM Reserves R
    WHERE R.bid = 103
);
```

### 3、`EXISTS`

`EXISTS` 判断**子查询是否返回至少一行**，如果子查询返回任何行 → TRUE

### 4、相关子查询

✳这是子查询里**最重要的概念**

```sql
-- 找预订过 boat 102 的 sailors
SELECT S.sname
FROM Sailors S
WHERE EXISTS (
    SELECT *
    FROM Reserves R
    WHERE R.bid = 102
    AND S.sid = R.sid
);
```

这里：`S.sid` 来自**外层查询**，子查询必须**对每一行重新执行**。

------

### 5、`ANY / ALL`

- `>ANY` ：大于集合中的某一个
- `>ALL` ：大于集合中的所有

```sql
-- 找 rating > 某个 Popeye 的 sailor
SELECT *
FROM Sailors S
WHERE S.rating > ANY (
    SELECT S2.rating
    FROM Sailors S2
    WHERE S2.sname = 'Popeye'
);
```

---

## 六、视图

### 1、`VIEW`

**View（视图）**在 SQL 中本质上是：**一个保存起来的查询**。

它看起来像一张表，但其实 **不存数据**，只是一个查询定义。

```sql
CREATE VIEW Redcount AS
SELECT B.bid, COUNT(*) AS scount
FROM Boats2 B, Reserves2 R
WHERE R.bid = B.bid
AND B.color = 'red'
GROUP BY B.bid;
```

- 简化 SQL 查询
- 安全控制
- 逻辑抽象

### 2、`WITH`

```sql
WITH Reds AS (
    SELECT bid, COUNT(*)
    FROM Boats2 B, Reserves2 R
    WHERE R.bid = B.bid
    GROUP BY bid
)
SELECT *
FROM Reds;
```

|          | View        | WITH     |
| -------- | ----------- | -------- |
| 生命周期 | 永久        | 当前查询 |
| 创建方式 | CREATE VIEW | WITH     |
| 用途     | 长期使用    | 临时查询 |
