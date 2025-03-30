---
icon: material/test-tube
---

Colab，即“Colaboratory”，允许你在浏览器中编写和执行 Python，具有以下特点：

- 无需配置
- 免费访问 GPU
- 易于共享

可以运行`python`和`shell`脚本，辨识`shell`→前面加`!`

`%` 作为magic command 影响全局，比如要使用`cd`指令就必须用`%`

---

!!! tip
    改变runtime为GPU
    
    检查GPU，使用
    
    ```python
    !nvidia-smi
    ```

!!! info

     常用指令
    
     - `ls`：列出当前目录中的所有文件。
    
       - `ls -l`：以更详细的格式列出当前目录中的所有文件，包括文件权限、所有者、大小和修改日期等。
    
       - `pwd`：输出当前的工作目录路径。
    
       - `mkdir <dirname>`：创建一个名为 `<dirname>` 的新目录。
    
       - `cd <dirname>`：切换到名为 `<dirname>` 的目录。
    
       - `gdown`：从 Google Drive 下载文件。这个命令通常需要先安装 `gdown` 包。
    
       - `wget`：从互联网下载文件。
    
       - `python <python_file>`：执行一个 Python 文件。


