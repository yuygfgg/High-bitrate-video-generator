# High-bitrate-video-generator
A C++ program generating video that is impossible to compress.

Ported from the python code https://www.123684.com/s/nd0djv-S998d, originally posted at https://www.bilibili.com/video/BV1h2SVYXEGL

## Usage
With x265:
```
./high_bitrate_y4m | x265 --input - --y4m --preset placebo --lossless -o high_bitrate.265
```

Output to file:
```
./high_bitrate_y4m > output.y4m    
```

Surprisingly, x265 (or x264, or whatever encoder), can't compress the video any smaller while maintaining acceptable visual quality.~~A simple 7zip does the job much better.~~

## Build
For the highest performance
```shell
$CXX -o high_bitrate_y4m_profile high_bitrate_y4m.cpp \
    -O3 -march=native -mtune=native -ffast-math -flto  -fprofile-instr-generate \
    `pkg-config --cflags --libs opencv4`

LLVM_PROFILE_FILE="high_bitrate_y4m.profraw" ./high_bitrate_y4m_profile > /dev/null 2>&1
llvm-profdata merge -output=high_bitrate_y4m.profdata high_bitrate_y4m.profraw
$CXX -o high_bitrate_y4m high_bitrate_y4m.cpp \
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
