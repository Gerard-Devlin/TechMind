[TOC]

---

## 一、`while` 循环

```c++
while () {
}
```

??? example "数位判断"
    ```c
    int main() {
        int a;
        int digits = 0;
    

        printf("Enter a number:");
        scanf("%d", &a);
        // 以下两句写在循环外是为了处理“0”的情况，否则将会输出0位数
        digits++;
        a = a / 10;
    
        while (a > 0) {
            digits++;
            a = a / 10;
        }
        printf("%d", digits);
    
        return 0;
    }
    // 电脑中有数位限制，太大的数字不行
    ```



!!! bug "debug"
 	- 测试程序常使用 ==边界数据== ,如有效范围两端的数据、特殊的倍数等

	- 循环体内要有改变条件的机会 → 不然会变成死循环

---

## 二、`do-while` 循环

```c++
do {
} while ();
```

| `do-while` 循环          | `while` 循环             |
| ------------------------ | ------------------------ |
| 先**进入循环**再判断条件 | 先**判断条件**再进入循环 |

```c++
int main() {
    int a;
    int digits = 0;
    printf("Enter a number:");
    scanf("%d\n", a);

    do {
        x = x / 10;
        digits++;
    } while (a > 0);

    return 0;
}
```

---

??? example "猜数字"

    ```c
    #include <bits/stdc++.h>
    using namespace std;
    
    int generateRandom() {
        srand(time(NULL));
        return rand() % 100;
    }
    
    int main() {
        int num = generateRandom();
        int guess;
        cout << "Guess a number between 0 and 99: ";
    
        bool flag = false;
        while (flag == false) {
            cin >> guess;
            if (guess == num) {
                cout << "Congratulations! You guessed the number correctly." << endl;
                flag = true;
            } else if (guess < num) {
                cout << "The number you guessed is too low. Try again: ";
            } else {
                cout << "The number you guessed is too high. Try again: ";
            }
        }
    }
    ```

??? example "算平均数"

    ```c
    #include <bits/stdc++.h>
    int main() {
        int x;
        int sum = 0;
        int count = 0;
        int number[100];
    
        scanf("%d", &x);
        while (x != -1) {
            number[count] = x;
            sum = sum + x;
            count++;
            scanf("%d", &x);
        }
        if (count > 0) {
            printf("The average of all numbers is %f\n", 1.0 * sum / count);
            int i;
            printf("%d\t", number[i]);
            for (i = 0; i < count; i++) {
                if (number[i] > sum / count) {
                    printf("%d\n", number[i]);
                }
            }
        }
    
        return 0;
    }
    ```

??? example "整数逆序分解"

    要分解一个整数的各个位上的数字并进行计算，可以按照以下步骤进行：
    
    - 进行 `% 10` 操作，得到个位数。
    -  `/ 10` 操作，去掉个位数。
    - 重复上述两个步骤，直到整数变为0。
    
    ```c
    #include <bits/stdc++.h>
    int main() {
        int x;
        int reverse = 0;
        int digits = 0;
        printf("Enter a number:");
        scanf("%d", &x);
    
        while (x > 0) {
            digits = x % 10;
            x = x / 10;
            reverse = reverse * 10 + digits;
            printf("%d,%d,%d\n", digits, x, reverse); // 测试语句
        }
        printf("The reverse is: %d", reverse); // 逆序输出
        return 0;
    }
    ```

---
