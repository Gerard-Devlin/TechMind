[TOC]

# 4

## 4.1.1

### 循环

e.g. 数位判断

```c
int main() {
   int a;
   int digits=0;

   printf("Enter a number:");
   scanf("%d",&a);
//以下两句写在循环外是为了处理“0”的情况，否则将会输出0位数
    digits++;
    a=a/10;
    
   while (a>0) {
     digits++;
       a=a/10;
 }
 printf("%d",digits);

    return 0;
}
//电脑中有数位限制，太大的数字不行
```



> ### 验证
>
> 测试程序常使用 ==边界数据== ,如有效范围两端的数据、特殊的倍数等
> 个位数；
> 负数；
> 10;
> 0；
> ……



## 4.1.2

循环体内要有改变条件的机会→不然会变成死循环

while循环先判断条件
!!! tip
    >
    > ①可以使用 ==* 调试 *== 来检查代码
    >
    > ②可以在适当的地方使用 ==*printf*==

## 4.1.3

### do-while循环

| do-while循环         | while循环            |
| -------------------- | -------------------- |
| 先进入循环再判断条件 | 先判断条件再进入循环 |

```c
int main(){
    int a;
    int digits=0;
    printf("Enter a number:");
    scanf("%d\n",a);
    
    do{
        x=x/10;
        digits++;
    }while(a>0);
    
    return 0;
}
```

## 4.2.2

### 猜数游戏

#### 生成随机数 rand()

```c
int main(){
    srand(time(0));
    int a =rand();
    printf("%d\n",a)
    return 0;
}
--------------------->
int main(){
    srand(time(0));
    int a =rand();
    //保证a在100以内
    a=a%100
    printf("%d\n",a)
    return 0;
}
```

## 4.2.3

### 算平均数

```c
int main() {
    int x;
    int sum=0;
    int count=0;
    int number[100];

    scanf("%d",&x);
    while(x!=-1) {
        number[count]=x;
        sum=sum+x;
        count++;
        scanf("%d",&x);
    }

    if(count>0){

        printf("The average of all numbers is %f\n", 1.0*sum/count);
        int i;
        printf("%d\t",number[i]);
        for(i=0;i<count;i++){

            if (number[i]>sum/count){
                printf("%d\n",number[i]);
            }
        }
    }
    return 0;
}
```

## 4.2.4

### 整数的分解（逆序）

> 一个整数是由1至多位数字组成的,如何分解出整数的各个位上的数字，然后加以计算
> 对一个整数做%10的操作,就得到它的个位数;
> 对一个整数做/10的操作，就去掉了它的个位数；
> 然后再对2的结果做%10，就得到原来数的十位数了
> 依此类推

```c
int main(){
    int  x;
    int  reverse=0;
    int  digits=0;
    printf("Enter a number:");
    scanf("%d",&x);

    while(x>0) {
        digits=x%10;
        x=x/10;
        reverse=reverse*10+digits;
        printf("%d,%d,%d\n",digits,x,reverse);//测试语句
    }
printf("The reverse is: %d",reverse);//逆序输出
return 0;
}
```

