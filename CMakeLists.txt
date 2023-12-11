# CMakeLists.txt: проект CMake для PyTorchNN; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.

cmake_minimum_required(VERSION 3.8)
project(PyTorchNN)

# Добавьте исходный код в исполняемый файл этого проекта.
add_executable(PyTorchNN "PyTorchNN.cpp" "PyTorchNN.h")

if(CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PyTorchNN PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.

set(Torch_DIR "F:/modules/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
target_link_libraries(PyTorchNN ${TORCH_LIBRARIES})
include_directories(${TORCH_INCLUDE_DIRS})

# Set the path to the OpenCV installation directory
set(OpenCV_DIR "D:/Programs/opencv/opencv/build")

# Find and include the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link the OpenCV libraries to your target
target_link_libraries(PyTorchNN ${OpenCV_LIBS})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET PyTorchNN
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:PyTorchNN>)
endif (MSVC)

# Укажите путь к заголовочным файлам SFML
set(SFML_INCLUDE_DIR "SFML/include")
include_directories(${SFML_INCLUDE_DIR})

# Укажите путь к библиотекам SFML
set(SFML_LIBRARY_DIR "SFML/lib")

# Укажите имена библиотек SFML
set(SFML_LIBRARIES
    sfml-system-d
    sfml-graphics-d
    sfml-audio-d
    sfml-network-d
    sfml-window-d
)

# Поиск и подключение библиотек SFML
foreach(SFML_LIB ${SFML_LIBRARIES})
    find_library(${SFML_LIB}_PATH ${SFML_LIB} PATHS "F:/Projects/c++/modularNN/PyTorchNN/SFML/lib")
    if(NOT ${SFML_LIB}_PATH)
        message(FATAL_ERROR "Failed to find ${SFML_LIB} library.")
    endif()
    list(APPEND SFML_LIBRARY_PATHS ${${SFML_LIB}_PATH})
endforeach()

# Добавьте ссылки на библиотеки SFML
target_link_libraries(PyTorchNN ${SFML_LIBRARY_PATHS})

# Укажите путь к директории, содержащей исполняемый файл SFML
if(MSVC)
  set(SFML_BIN_DIR "SFML/bin")
  file(GLOB SFML_DLLS "${SFML_BIN_DIR}/*.dll")
  add_custom_command(TARGET PyTorchNN POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${SFML_DLLS}
                     $<TARGET_FILE_DIR:PyTorchNN>)
endif(MSVC)
