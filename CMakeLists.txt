cmake_minimum_required(VERSION 3.15)
project(high-bitrate-video-generator CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Common compiler options
set(COMMON_FLAGS
    -march=native
    -mtune=native
    -ffast-math
)

# Set PGO flags for Clang
set(PROFILE_GEN_FLAGS "-fprofile-generate=${CMAKE_BINARY_DIR}/profile")
set(PROFILE_USE_FLAGS "-fprofile-use=${CMAKE_BINARY_DIR}/profile.profdata")

# PGO build options
option(ENABLE_PGO "Enable Profile-Guided Optimization" OFF)

if(ENABLE_PGO)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "PGO is only supported with Clang compiler")
    endif()

    # Profile generation build
    add_executable(high_bitrate_y4m_profile high_bitrate_y4m.cpp)
    target_link_libraries(high_bitrate_y4m_profile PRIVATE ${OpenCV_LIBS})
    target_include_directories(high_bitrate_y4m_profile PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_options(high_bitrate_y4m_profile PRIVATE 
        -O3 
        ${COMMON_FLAGS}
        -flto
        ${PROFILE_GEN_FLAGS}
    )
    target_link_options(high_bitrate_y4m_profile PRIVATE ${PROFILE_GEN_FLAGS})

    # Add custom target to generate profile data
    add_custom_target(generate_profile
        COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profile.profraw 
                ./high_bitrate_y4m_profile > /dev/null 2>&1
        COMMAND llvm-profdata merge -output=${CMAKE_BINARY_DIR}/profile.profdata 
                ${CMAKE_BINARY_DIR}/profile.profraw
        DEPENDS high_bitrate_y4m_profile
    )

    # Final PGO-optimized build
    add_executable(high_bitrate_y4m high_bitrate_y4m.cpp)
    target_link_libraries(high_bitrate_y4m PRIVATE ${OpenCV_LIBS})
    target_include_directories(high_bitrate_y4m PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_options(high_bitrate_y4m PRIVATE 
        -O3 
        ${COMMON_FLAGS}
        -flto
        ${PROFILE_USE_FLAGS}
    )
    target_link_options(high_bitrate_y4m PRIVATE ${PROFILE_USE_FLAGS})
    add_dependencies(high_bitrate_y4m generate_profile)
else()
    # Main executable without PGO
    add_executable(high_bitrate_y4m high_bitrate_y4m.cpp)
    target_link_libraries(high_bitrate_y4m PRIVATE ${OpenCV_LIBS})
    target_include_directories(high_bitrate_y4m PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_options(high_bitrate_y4m PRIVATE 
        -O3 
        ${COMMON_FLAGS}
        -flto
    )
endif()
