cmake_minimum_required(VERSION 3.10)

project(Network2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Network2
    main.cpp
    Matrix.cpp
    Network.cpp
)
