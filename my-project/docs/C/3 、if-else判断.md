[TOC]

# 3

## 3.1.2

e.g. BCD数是用一个字节来表达两位十进制的数，每四个比特表示一位。所以如果一个BCD数是0x12,它表达的就是十进制的12。
但是小明没学过BCD,把所有的BCD数都当作二进制数转换成十进制输出了。于是BCD的0x12被输出成了十进制的18了！
现在，你的程序要读入这个错误的十进制数，然后输出正确的十进制数。提示：你可以把18转换回0×12，然后再转换回12。

>输入格式：

​	输入在一行中给出一个[0,153]范围内的整数，保证能转换回有效的BCD数，就是说这个整数转换成十六进制时不会出现A-F的数字。
>输出格式：

​	输出对应的十进制数。

```c
DEC 18 
BIN 0001 0010 
HEX 0x12
BCD 12
```

```c
//方法一：
18/16 →1
18%16 →2
1*10+2=12
----------------
int main() {
    int a;
    int b;
    int c;
    printf("Enter a number: ");
    scanf("%d", &a);
    b=a/16;
    c=a%16;
    printf("%d\n",b*10+c);
    return 0;
}
```

```c
//方法二：
printf("%x",x)
-----------------
int main() {
    int a ;
    printf("Enter a number: ");
    scanf("%d",&a);
    printf("The number is %x",a);
    
    return 0;
}
```

!!! note
    >
    > %d：十进制输出
    > %x：十六进制输出

## 3.1.3

```c
//0 O有所不同
```

## 3.2.1

### 判断语句--if

```c
int main() {
    int hour1,minute1,hour2,minute2;
    printf("Enter the first time: ");
    scanf("%d:%d",&hour1,&minute1);
    printf("Enter the second time: ");
    scanf("%d:%d",&hour2,&minute2);

    int ih = hour2-hour1;
    int im =minute2-minute1;
    //判断分钟差是否在一小时以内
    if(im<0) {
        im=60+im;
        ih --;

    }
    printf("%d,%d",ih,im);
   
    return 0;
}
------------------
//改进，用判断拒绝分钟大于60的输入
    int main() {
    int hour1,minute1,hour2,minute2;
    printf("Enter the first time: ");
    scanf("%d:%d",&hour1,&minute1);
    printf("Enter the second time: ");
    scanf("%d:%d",&hour2,&minute2);

    int ih = hour2-hour1;
    int im =minute2-minute1;
   
    if (minute1,minute2>60) {
        printf("Time error!!!");
    }
    else if (im<0) {

        im=60+im;
        ih --;
    }
    printf("%d,%d",ih,im);



    return 0;
}
```



### 条件

| 运算符 | 意义       |
| ------ | ---------- |
| ==     | 相等       |
| !=     | 不相等     |
| >      | 大于       |
| >=     | 大于或等于 |
| <      | 小于       |
| <=     | 小于或等于 |

!!! abstract
    >
    > - 关系运算结果只有1/0
    >   - ==/!=运算优先级比其他关系运算符更低

### 优先级

赋值运算<关系运算<算术运算

```c
//验证程序
printf("%d\n",7>=3+4)
    →[output]=1
```

## 3.2.3

### 找零计算器（优化）

```c
int main() {
    double price;
    double bill;
    printf("Please enter your price:");
    scanf("%lf", &price);
    printf("Please enter your bill:");
    scanf("%lf", &bill);
    //判断价格和票面
    if (price < bill) {
        printf("Your price is less than your bill");
    }
    else if (bill<price) {
        printf("Your bill change: %f", price-bill);
    }

    return 0;
}
```

## 3.2.4

### 否则 else （if）

## 3.2.5

if语句逻辑：①if后有一个括号，若括号内成立→执行大括号内语句

​		    ②if后有一个括号，括号内成立→无大括号→==只执行if后面一句==

```c
//实例
int main() {
    const int PASS=60;
    int score;
    printf("Please enter your score:\n");
    scanf("%d",%score);

    if (score<PASS)
        printf("You failed\n");
    else
        printf("You passed\n");
    
    return 0;
}
```

!!! note
    >
    > if后无大括号的话，本句结尾没有<u>==；==</u>



## 3.3.1

### 嵌套的if-else

e.g. 判断三个数中最大的

```c
int main() {
    int a,b,c;
    printf("Enter three numbers:");
    scanf("%d %d %d",&a,&b,&c);
    if (a>b) {
        if (a>c) {
            printf("%d is max",a);
        }
        else {
            printf("%d is max",c);
        }
    }

    if (a<b) {
        if (b>c) {
            printf("%d is max",b);
        }
        else {
            printf("%d is max",c);
        }
    }

    return 0;
}
```

!!! warning
    >
    > else（无大括号的情况下）总与最近的if配对
    >
    > C语言中缩进不表示同一层级，但是最好编写时注意→便于阅读
    >
    > ∴最好if/else总是用  { }

## 3.3.2

### 级联的if/else

```c
if(){
    
}
else if(){
    
}
else(){
    
}
```

> 单一出口灵活性

## 3.3.4

### 多路if/else→switch case
```c
//格式
switch case(控制表达式){
    case constant:
    ……
    case constant:
    ……
    default:
    ……
    
}
//控制表达式的输入只能是“整数”
//const 必须是整数，或者整数计算表达式
//switch-case必须要有“break”，if不需要
```

```c
//级联的
int main() {

    int type;
    printf("type a number:");
    scanf("%d",&type);
    if (type==1)
        printf("Good morning!");
    else if (type==2)
        printf("Good afternoon!");
    else if (type==3)
        printf("Good evening!");
    else if (type==4)
        printf("Good night!");

    return 0;
}
//多路的
int main() {
     int type;

     printf("Enter a number:");
     scanf("%d", &type);
  switch (type) {
      case 1:
          printf("Good morning!");
      break;
      case 2:
          printf("Good afternoon!");
      break;
      case 3:
          printf("Good evening!");
      break;
      case 4:
          printf("Good night!");
      break;
      default:
          printf("Wrong number!");
      break;
  }
     return 0;
 }
```



e.g.成绩互转

```c
int main() {
    int score;
    int grade;
    printf("Enter your score: ");
    scanf("%d", &score);
    grade=score/10;
    switch(grade) {
        case 10:
        case 9:
            printf("A\n");
        break;
        case 8:
            printf("B\n");
        break;
        case 7:
            printf("C\n");
        break;
        case 6:
            printf("D\n");
        break;
        default:
            printf("F\n");
        break;
    }
      return 0;
}
```

