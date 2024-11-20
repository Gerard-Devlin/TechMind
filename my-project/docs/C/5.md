[TOC]

# 5

## 5.1.1

### for循环

```c
//用while语句
int main() {
    int n;
    int count=0;
    int factorial=1;
    printf("Enter the number of factorial: ");
    scanf("%d", &n);
    while(count<n) {
        count ++;
        factorial=factorial*count;
    }

    printf("The factorial is %d",factorial);
    return 0;
}
//用for语句
int main() {

 int n;
 int fact=1;
 int i=1;
 printf("Enter the number of factorial: ");
 scanf("%d", &n);
//①从1开始乘
for(i=1;i<=n;i++) {
 fact=fact*i;
}
//②从n开始乘
for(i=n;i>=2;i--){
    fact=fact*i
}

 printf("The factorial of %d is %d",n,fact);

 return 0;
}
```

```c
for (i=1① ; i<=n ②; i++③ )
/*
①初始条件
②循环继续的条件
③循环每轮要做的事情
 */
```

## 5.1.2

### 循环的选择

for==while

!!! warning
    >
    > for循环中的每一个表达式都可以省略，但是分号不能省略
    >
    > for(；条件；)==while（条件）

- for循环
- while循环
- do-while循环

!!! tip
    >
    > 有固定次数，用for
    >
    > 必须执行一次，用do-while
    >
    > 其他情况，用while

## 5.2.1

### 循环控制

```c
int main(){

 int x;
 int i;
 int isPrime=1;
 scanf("%d",&x);
 
 for(i=2;i<x;i++) {
    if(x%i==0) {
     isPrime=0;
     break;//跳出整个循环
    }
 }

if(isPrime==1) {
 printf("yes");
}else {
 printf("no");
}

return 0;
}
```

#### break v.s. continue

- break：跳出整个循环

- continue：跳出本轮循环，去下一轮循环

break/continue只能跳出本层循环
## 5.2.2

### 循环嵌套

e.g.输出100以内的素数

```c
int main(){

 int x;
 int i;



 for (x=2;x<100;x++) {
  int isPrime=1;
       for(i=2;i<x;i++) {
          if(x%i==0) {
           isPrime=0;
           break;
          }
       }

       if(isPrime==1) {
        printf("%d\n",x);
       }

 }
return 0;
}
```



## 5.2.3

### 跳出循环 （goto）

e.g.凑硬币：1角、2角、5角凑出10元以下金额

## 5.3.1

### 求前n项和

$$
f(n)=1+  \frac {1}{2}  +
  \frac {1}{3}  + \frac {1}{4}  +  \cdots  ++  \frac {1}{n}  
$$



## 5.3.2

### 整数分解（正序）



```c
/*13425/10000->1
13425%10000->3425
10000/10->1000

3425/1000->3
3425%1000->425
1000/10->100

^^^^
*/
```

```c
int main() {
    int x=13425;
     int div=10000;
     int n;
     while(x>0) {

          n=x/div;
          x=x%div;
          div=div/10;

          printf("%d ",n);
         }

 return 0;
}

//////////////////////////////
//改进：可输入数字自动判断位数//
/////////////////////////////
int main() {

 int x;
 int count=0;
 printf("Enter a number: ");
 scanf("%d", &x);
//统计位数
 int num=x;
 while(x>0) {
  x=x/10;
  count+=1;
 }
//分离数位
 int div=pow(10,count-1);
 int n=0;
 while(num>0) {

      n=num/div;
      num=num%div;
      div=div/10;

  printf("%d ",n);
 }


 return 0;
}
```

## 5.3.3

### 最大公约数

（1）枚举法

```c
int main() {
    int a,b;
    int min;
    printf("Enter two numbers:");
    scanf("%d%d",&a,&b);

    if(a>b) {
        min=b;
    }else {min=a;}


    int i=1;
    int ret=0;
    for(i=1;i<min;i++) {
        if(a%i==0) {
            if(b%i==0) {
                ret=i;
            }
        }
        if(b%i==0) {
        }

    }

    printf("GCS is %d",ret);

    return 0;
}
```

（2）辗转相除法

```c
/*如果b等于0，计算结束，a就是最大公约数；
否则，计算a除以b的余数，让a等于b,而b等于那个余数
回到第一步*/
```

| a     | b    | t    |
| ----- | ---- | ---- |
| 12    | 18   | 12   |
| 18    | 12   | 6    |
| 12    | 6    | 0    |
| ==6== | 0    |      |

```c
int main() {

    int a,b;
    printf("Enter two numbers:\n");
    scanf("%d %d",&a,&b);

    if(b==0) {
        printf("%d",a);
    }
    else {
        while(b!=0) {

            int t = a % b;
            a=b;
            b=t;
        }
        printf("%d",a);
    }

    return 0;
}
```

