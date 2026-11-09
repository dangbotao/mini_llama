这三个头文件都使用统一的“流”接口，主要区别是数据来源或去向不同：

| 头文件 | 数据来源/去向 | 主要类型 |
|---|---|---|
| `<iostream>` | 终端标准输入输出 | `std::cin`、`std::cout`、`std::cerr`、`std::clog` |
| `<fstream>` | 文件 | `std::ifstream`、`std::ofstream`、`std::fstream` |
| `<sstream>` | 内存中的字符串 | `std::istringstream`、`std::ostringstream`、`std::stringstream` |

### `<iostream>`：终端输入输出

常见对象：

```cpp
std::cin    // 标准输入，通常是键盘
std::cout   // 标准输出，通常是终端
std::cerr   // 错误输出，通常不缓冲
std::clog   // 日志输出，通常有缓冲
```

示例：

```cpp
#include <iostream>
#include <string>

int age;
std::string name;

std::cin >> age;
std::getline(std::cin, name);

std::cout << "age = " << age << '\n';
std::cerr << "error message\n";
```

常用接口：

```cpp
stream >> value;           // 格式化读取
stream << value;           // 格式化输出

stream.get();              // 读取一个字符
stream.read(buf, size);    // 读取指定字节数
stream.ignore();           // 忽略输入
stream.peek();             // 查看下一个字符但不取走

stream.put(ch);            // 输出一个字符
stream.write(buf, size);   // 输出指定字节数
stream.flush();            // 刷新输出缓冲区
```

### `<fstream>`：文件输入输出

提供三个常用类型：

```cpp
std::ifstream   // 只读文件流
std::ofstream   // 只写文件流
std::fstream    // 可读可写文件流
```

文本读取示例：

```cpp
#include <fstream>
#include <string>

std::ifstream file("config.txt");

std::string line;
while (std::getline(file, line)) {
    // 处理每一行
}
```

二进制读取示例：

```cpp
std::ifstream file("model.bin", std::ios::binary);

char buffer[128];
file.read(buffer, sizeof(buffer));

std::streamsize count = file.gcount(); // 实际读取的字节数
```

文件写入示例：

```cpp
std::ofstream file("output.txt");

file << "hello\n";
file.write(data, size);
```

常用接口：

```cpp
file.open(path, mode);  // 打开文件
file.close();           // 关闭文件
file.is_open();         // 是否成功打开

file.seekg(offset);     // 设置读取位置
file.tellg();           // 获取当前读取位置

file.seekp(offset);     // 设置写入位置
file.tellp();           // 获取当前写入位置
```

常见打开模式：

```cpp
std::ios::in       // 读取
std::ios::out      // 写入
std::ios::binary   // 二进制模式
std::ios::app      // 追加到文件末尾
std::ios::trunc    // 打开时清空原文件
std::ios::ate      // 打开后定位到文件末尾
```

模式可以组合：

```cpp
std::fstream file(
    "data.bin",
    std::ios::in | std::ios::out | std::ios::binary
);
```

### `<sstream>`：字符串输入输出

把 `std::string` 当作输入输出流，常用于类型转换、文本解析和拼接。

提供三个类型：

```cpp
std::istringstream   // 从字符串读取
std::ostringstream   // 向字符串写入
std::stringstream    // 可读可写
```

解析字符串：

```cpp
#include <sstream>
#include <string>

std::istringstream input("123 3.14 hello");

int number;
double value;
std::string word;

input >> number >> value >> word;
```

拼接、格式化字符串：

```cpp
std::ostringstream output;

output << "layer=" << 2 << ", size=" << 128;

std::string result = output.str();
```

读写字符串：

```cpp
std::stringstream stream;

stream << 100 << ' ' << 3.14;

int number;
double value;
stream >> number >> value;
```

常用接口：

```cpp
stream.str();          // 获取内部字符串
stream.str("10 20");   // 替换内部字符串
```

### 三者共有的状态接口

因为它们都属于 C++ 流体系，所以都有类似的状态检查方法：

```cpp
stream.good();  // 当前状态正常
stream.eof();   // 到达输入末尾
stream.fail();  // 格式解析失败或操作失败
stream.bad();   // 发生严重输入输出错误
stream.clear(); // 清除错误状态
```

例如：

```cpp
std::ifstream file("model.bin", std::ios::binary);

if (!file) {
    std::cerr << "打开文件失败\n";
}
```

可以简单记成：

- `<iostream>`：和终端交流。
- `<fstream>`：和文件交流。
- `<sstream>`：和字符串交流。
- 三者都使用 `>>`、`<<`、`read()`、`write()` 等相似的流式接口。
