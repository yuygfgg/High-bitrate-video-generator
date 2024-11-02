# High-bitrate-video-generator
A C++ program generating video that is impossible to compress.

Ported from the python code https://www.123684.com/s/nd0djv-S998d, originally posted at https://www.bilibili.com/video/BV1h2SVYXEGL

## Usage


With x265:
```
./high_bitrate_y4m | x265 --input - --y4m --preset placebo --crf 10 -o high_bitrate.265
```

Output to file:
```
./high_bitrate_y4m > output.y4m    
```

```
Options:
  --width N       Width in pixels (default: 3840)
  --height N      Height in pixels (default: 2160)
  --fps N         FPS (default: 60)
  --duration N    Seconds (default: 20)
  --lines N       Lines per frame (default: 50)
  --seed N        Random seed (default: random)
  --threads N     Thread count (default: auto)
```

## Build

### Prerequisites
- CMake 3.15 or higher
- OpenCV 4
- A C++17 compatible compiler (GCC or Clang)

### Normal Build
```bash
mkdir build && cd build
cmake ..
make
```

### Build with Profile-Guided Optimization (PGO)
For the highest performance:
```bash
mkdir build && cd build
cmake -DENABLE_PGO=ON ..
make
```

![smaller](https://github.com/user-attachments/assets/59815017-4d68-4be4-9363-7a164fe80817)
