cmake_minimum_required(VERSION 3.17)
project(mini_llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(tokenizer_test
    tests/tokenizer_test.cc
    src/bpe_tokenizer.cc)

set(MINI_LLAMA_CORE_SOURCES
    src/tensor.cc
    src/matmul_optimized.cc
    src/operations.cc
    src/bpe_tokenizer.cc
    src/kv_cache.cc
    src/context.cc
    src/batch.cc
    src/forward.cc
)

option(MINI_LLAMA_BUILD_GTEST "Build optional GoogleTest-based tests" OFF)
option(MINI_LLAMA_USE_SYSTEM_GTEST "Use system-installed GoogleTest" OFF)
if(MINI_LLAMA_BUILD_GTEST)
    if(MINI_LLAMA_USE_SYSTEM_GTEST)
        find_package(GTest CONFIG QUIET HINTS /opt/homebrew/lib/cmake/GTest)
    endif()
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    set(GTEST_SOURCES ${MINI_LLAMA_CORE_SOURCES})
    file(GLOB GTEST_TEST_FILES CONFIGURE_DEPENDS tests/gtest/*.cc)
    list(APPEND GTEST_SOURCES ${GTEST_TEST_FILES})

    add_executable(mini-llama-gtests ${GTEST_SOURCES})
    target_link_libraries(mini-llama-gtests PRIVATE GTest::gtest_main)

    enable_testing()
    add_test(NAME mini-llama-gtests COMMAND mini-llama-gtests
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
