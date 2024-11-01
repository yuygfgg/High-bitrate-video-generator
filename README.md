# High-bitrate-video-generator
A C++ program generating video that is impossible to compress.

## Build
For the highest performance
```shell
$CXX -o high_bitrate_y4m_profile high_bitrate_y4m.cpp \
    -O3 -march=native -mtune=native -ffast-math -flto  -fprofile-instr-generate \
    `pkg-config --cflags --libs opencv4`

LLVM_PROFILE_FILE="high_bitrate_y4m.profraw" ./high_bitrate_y4m_profile > /dev/null 2>&1
llvm-profdata merge -output=high_bitrate_y4m.profdata high_bitrate_y4m.profraw
$CXX -o high_bitrate_y4m_pgo high_bitrate_y4m.cpp \
    -O3 -fprofile-instr-use=high_bitrate_y4m.profdata \
    -march=native -mtune=native \
    -ffast-math -flto \
    `pkg-config --cflags --libs opencv4`
```

Or

```shell
$CXX -o high_bitrate_y4m high_bitrate_y4m.cpp -O3 -march=native -mtune=native -ffast-math -flto `pkg-config --cflags --libs opencv4`
```

![smaller](https://github.com/user-attachments/assets/59815017-4d68-4be4-9363-7a164fe80817)
