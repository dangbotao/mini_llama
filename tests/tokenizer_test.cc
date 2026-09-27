#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "mini_llama/tokenizer.h"

namespace {

void PrintUtf8(char32_t code_point) {
    try {
        const std::string utf8 =
            mini_llama::UnicodePointToUtf8(code_point);

        std::cout << "U+"
                  << std::uppercase
                  << std::hex
                  << std::setw(6)
                  << std::setfill('0')
                  << static_cast<std::uint32_t>(code_point)
                  << " -> bytes:";

        // 必须转成 unsigned char，否则大于 0x7F 的 char
        // 可能被当成负数打印。
        for (unsigned char byte : utf8) {
            std::cout << " "
                      << std::setw(2)
                      << static_cast<unsigned int>(byte);
        }

        std::cout << std::dec
                  << "  text=[" << utf8 << "]\n";
    } catch (const std::exception& e) {
        std::cout << "U+"
                  << std::uppercase
                  << std::hex
                  << static_cast<std::uint32_t>(code_point)
                  << " -> exception: "
                  << e.what()
                  << '\n';
    }
}

}  // namespace

int main() {
    PrintUtf8(0x0041);    // A，1 字节
    PrintUtf8(0x00E9);    // é，2 字节
    PrintUtf8(0x4E2D);    // 中，3 字节
    PrintUtf8(0x1F600);   // 😀，4 字节

    PrintUtf8(0x007F);    // 1 字节上界
    PrintUtf8(0x0080);    // 2 字节下界
    PrintUtf8(0x07FF);    // 2 字节上界
    PrintUtf8(0x0800);    // 3 字节下界
    PrintUtf8(0xFFFF);    // 3 字节上界
    PrintUtf8(0x10000);   // 4 字节下界
    PrintUtf8(0x10FFFF);  // Unicode 最大值

    PrintUtf8(0xD800);    // 非法代理项
    PrintUtf8(0x110000);  // 超出 Unicode 范围

    return 0;
}