cmake_minimum_required(VERSION 3.20)
project(word-piece-encoder)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ICU COMPONENTS uc i18n REQUIRED)

add_executable(word-piece-encoder
        main.cpp
        encoder/src/VocabList.cpp
        encoder/src/WordPiece.cpp
        encoder/include/IEncoder.h
        encoder/include/VocabList.h
        encoder/include/WordPiece.h
)

target_include_directories(word-piece-encoder PRIVATE
        encoder/include
        ${ICU_INCLUDE_DIRS}
)

target_compile_definitions(word-piece-encoder PRIVATE
        PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)

if(TARGET ICU::uc)
    target_link_libraries(word-piece-encoder PRIVATE ICU::uc ICU::i18n)
else()
    target_link_libraries(word-piece-encoder PRIVATE ${ICU_LIBRARIES})
endif()
