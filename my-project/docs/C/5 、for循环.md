[TOC]

---

## 一、`for` 循环

```c++
for (i = 1; i <= n; i++) {
    // 循环体内容
}
/*
 * ① 初始条件：i = 1
 * 在循环开始之前，初始化循环变量 i 为 1。
 *
 * ② 循环继续的条件：i <= n
 * 在每次循环开始之前，检查 i 是否小于等于 n。
 * 如果条件为真，则继续执行循环体；如果条件为假，则退出循环。
 *
 * ③ 循环每轮要做的事情：i++
 * 在每次循环体执行完毕后，将 i 的值加 1。
 */
```

---

## 二、`for`  与 `while`


=== "`while`"
    ```c
    int main() {
        int n;
        int count = 0;
        int factorial = 1;
        printf("Enter the number of factorial: ");
        scanf("%d", &n);
        while (count < n) {
            count++;
            factorial = factorial * count;
        }

        printf("The factorial is %d", factorial);
        return 0;
    }
    ```

=== "`for`"
    ```c++
    int main() {
        int n;
        int fact = 1;
        int i = 1;
        printf("Enter the number of factorial: ");
        scanf("%d", &n);
        // ①从1开始乘
        for (i = 1; i <= n; i++) {
            fact = fact * i;
        }
        // ②从n开始乘
        for (i = n; i >= 2; i--) {
            fact = fact * i
        }
        printf("The factorial of %d is %d", n, fact);
    ```

        return 0;
    }
    ```

!!! warning "`for == while`"
	for循环中的每一个表达式都可以省略，但是分号不能省略
    
    ```c++
    for(;条件;) == while(条件)
    ```



!!! tip
    - 有固定次数，用 `for`
    
    - 必须执行一次，用 `do-while`
    
    - 其他情况，用 `while`

---

## 三、`break` 与 `continue`

```c
int main() {
    int x;
    int i;
    int isPrime = 1;
    scanf("%d", &x);

    for (i = 2; i < x; i++) {
        if (x % i == 0) {
            isPrime = 0;
            break; // 跳出整个循环
        }
    }

    if (isPrime == 1) {
        printf("yes");
    } else {
        printf("no");
    }

    return 0;
}
```

| `break`      | `continue`                 |
| ------------ | -------------------------- |
| 跳出整个循环 | 跳出本轮循环，去下一轮循环 |

!!! warning
	`break` `continue` 都只能跳出本层循环


??? example "输出素数"

    ```c
    int main() {
        int x;
        int i;
    
        for (x = 2; x < 100; x++) {
            int isPrime = 1;
            for (i = 2; i < x; i++) {
                if (x % i == 0) {
                    isPrime = 0;
                    break;
                }
            }
    
            if (isPrime == 1) {
                printf("%d\n", x);
            }
        }
    
        return 0;
    }
    ```

---

## 四、⚠️~~`goto`~~

`goto` 会破坏程序结构，导致流程混乱、可读性差，增加维护和调试难度，容易引发隐藏错误，因此现代编程中应尽量避免使用。

---

??? example "凑硬币"
    ```c++
    #include <stdio.h>
    ```

    int main() {
        int target;
        scanf("%d", &target);
    
        // 遍历所有可能的硬币组合
        for (int x = 0; x <= 99; x++) {         // 1角硬币的数量
            for (int y = 0; y <= 49; y++) {     // 2角硬币的数量
                for (int z = 0; z <= 19; z++) { // 5角硬币的数量
                    if (x * 1 + y * 2 + z * 5 == target) {
                        printf("1 cent：%d, 2 cent：%d, 5 cent：%d\n", x, y, z);
                    }
                }
            }
        }
    
        return 0;
    }
    ```

??? example "整数正序分解"

    ```c
    int main() {
        int x; // 输入的整数
        int count = 0; // 初始化位数计数器
        printf("Enter a number: ");
        scanf("%d", &x);
    
        // 统计位数
        int num = x; // 用 num 保存原始值
        while (num > 0) {
            num = num / 10; // 去掉最低位
            count += 1; // 位数加1
        }
    
        // 分离数位
        num = x; // 重新使用原始值
        int div = pow(10, count - 1); // 计算最高位的除数
        while (num > 0) {
            int n = num / div; // 获取当前最高位的数字
            num = num % div; // 去掉当前最高位
            div = div / 10; // 更新除数
    
            // 打印分离的数位
            printf("%d ", n);
        }
    
        return 0;
    }
    ```



???+ example "最大公约数"

    === "枚举"
    
        ```c
        int main() {
            int a, b;
            int min;
            printf("Enter two numbers:");
            scanf("%d%d", &a, &b);
    
            if (a > b) {
                min = b;
            } else {
                min = a;
            }
    
            int i = 1;
            int ret = 0;
            for (i = 1; i < min; i++) {
                if (a % i == 0) {
                    if (b % i == 0) {
                        ret = i;
                    }
                }
                if (b % i == 0) {
                }
            }
    
            printf("GCD is %d", ret);
    
            return 0;
        }
        ```
    
    === "辗转相除"
        ```c
        int main() {
            int a, b; // 定义两个整数变量 a 和 b
            printf("Enter two numbers:\n");
            scanf("%d %d", &a, &b); // 从用户输入中读取两个整数
    
            // 检查 b 是否为0
            if (b == 0) {
                printf("%d", a); // 如果 b 为0，直接输出 a，因为 a 是最大公因数
            } else {
                // 使用欧几里得算法计算最大公因数
                while (b != 0) {   // 当 b 不为0时，继续循环
                    int t = a % b; // 计算 a 除以 b 的余数
                    a = b;         // 将 b 的值赋给 a
                    b = t;         // 将余数 t 赋给 b
                }
                printf("%d", a); // 循环结束后，a 的值即为最大公因数
            }
    
            return 0; // 程序结束
        }
        ```
    === "`__gcd`"
        ```c++
        #include <bits/stdc++.h>
        using namespace std;
        int main() {
            int a, b;
            cin >> a >> b;
            cout << __gcd(a, b) << endl; // 求最大公约数
        }
        ```

---

