#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <cstring>
#include <queue>
#include <mutex>
#include <condition_variable>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

template<typename T>
T* aligned_alloc(size_t size, size_t alignment = 32) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

class FastRandom {
    alignas(32) std::array<uint32_t, 16> state_;
    int index_;

public:
    FastRandom(uint32_t seed = std::random_device{}()) : index_(0) {
        std::mt19937 gen(seed);
        for (auto& s : state_) {
            s = gen();
        }
    }

    void generate_bytes(uint8_t* dest, size_t count) {
        size_t i = 0;
        
#if defined(__ARM_NEON__)
        for (; i + 16 <= count; i += 16) {
            uint32x4_t x = vld1q_u32(&state_[index_]);
            
            x = veorq_u32(x, vshlq_n_u32(x, 13));
            x = veorq_u32(x, vshrq_n_u32(x, 17));
            x = veorq_u32(x, vshlq_n_u32(x, 5));
            
            vst1q_u32(&state_[index_], x);
            vst1q_u8(dest + i, vreinterpretq_u8_u32(x));
            
            index_ = (index_ + 4) & 15;
        }

#elif defined(__SSE2__)
        for (; i + 16 <= count; i += 16) {
            __m128i x = _mm_load_si128((__m128i*)&state_[index_]);
            
            x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
            x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
            x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
            
            _mm_store_si128((__m128i*)&state_[index_], x);
            _mm_storeu_si128((__m128i*)(dest + i), x);
            
            index_ = (index_ + 4) & 15;
        }
#endif
        for (; i < count; ) {
            uint32_t& x = state_[index_];
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            
            size_t bytes_to_copy = std::min(size_t(4), count - i);
            std::memcpy(dest + i, &x, bytes_to_copy);
            
            i += bytes_to_copy;
            index_ = (index_ + 1) & 15;
        }
    }

    uint32_t generate_uint32() {
        uint32_t result;
        generate_bytes(reinterpret_cast<uint8_t*>(&result), sizeof(result));
        return result;
    }

    int generate_int(int min, int max) {
        return min + (generate_uint32() % (max - min + 1));
    }
};

class FrameGenerator {
    const int width_;
    const int height_;
    FastRandom rng_;
    std::vector<uint8_t> buffer_;

public:
    FrameGenerator(int width, int height)
        : width_(width)
        , height_(height)
        , buffer_(width * height * 3)
    {}

    cv::Mat generateFrame() {
        // 使用NEON生成随机噪声
        rng_.generate_bytes(buffer_.data(), buffer_.size());

        cv::Mat frame(height_, width_, CV_8UC3, buffer_.data());

        // 添加随机线条
        for (int i = 0; i < 50; ++i) {
            cv::Point pt1(
                rng_.generate_int(0, width_ - 1),
                rng_.generate_int(0, height_ - 1)
            );
            cv::Point pt2(
                rng_.generate_int(0, width_ - 1),
                rng_.generate_int(0, height_ - 1)
            );

            cv::Scalar color(
                rng_.generate_int(0, 255),
                rng_.generate_int(0, 255),
                rng_.generate_int(0, 255)
            );

            cv::line(frame, pt1, pt2, color, rng_.generate_int(1, 9));
        }

        return frame.clone();
    }
};

class BatchProcessor {
    const size_t batch_size_;
    std::vector<cv::Mat> frames_;
    std::mutex mutex_;
    std::condition_variable producer_cv_;
    std::condition_variable consumer_cv_;
    bool done_ = false;
    size_t produced_ = 0;
    size_t consumed_ = 0;

public:
    BatchProcessor(size_t batch_size)
        : batch_size_(batch_size)
        , frames_(batch_size)
    {}

    void push(cv::Mat&& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        producer_cv_.wait(lock, [this] {
            return (produced_ - consumed_) < batch_size_;
        });

        frames_[produced_ % batch_size_] = std::move(frame);
        produced_++;

        if (produced_ - consumed_ >= batch_size_) {
            consumer_cv_.notify_one();
        }
    }

    bool pop(std::vector<cv::Mat>& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        consumer_cv_.wait(lock, [this] {
            return (produced_ - consumed_ >= batch_size_) || (done_ && produced_ > consumed_);
        });

        if (consumed_ >= produced_ && done_) {
            return false;
        }

        size_t count = std::min(produced_ - consumed_, batch_size_);
        batch.clear();

        for (size_t i = 0; i < count; ++i) {
            batch.push_back(std::move(frames_[(consumed_ + i) % batch_size_]));
        }

        consumed_ += count;
        producer_cv_.notify_all();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        consumer_cv_.notify_all();
    }
};

class Y4MWriter {
    const int width_;
    const int height_;
    const int fps_;
    std::mutex write_mutex_;
    std::vector<uint8_t> yuv_buffer_;

public:
    Y4MWriter(int width, int height, int fps)
        : width_(width)
        , height_(height)
        , fps_(fps)
        , yuv_buffer_(width * height * 3)
    {
        writeHeader();
    }

    void writeHeader() {
        std::fprintf(stdout, "YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 C444\n",
                    width_, height_, fps_);
        std::fflush(stdout);
    }

    void writeFrameBatch(const std::vector<cv::Mat>& frames) {
        std::lock_guard<std::mutex> lock(write_mutex_);

        for (const auto& frame : frames) {
            std::fprintf(stdout, "FRAME\n");
            cv::Mat yuv;
            cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV);
            fwrite(yuv.data, 1, yuv.total() * yuv.elemSize(), stdout);
        }
        std::fflush(stdout);
    }
};

void generateHighBitrateVideoY4M(int duration_seconds) {
    const int width = 3840;
    const int height = 2160;
    const int fps = 60;
    const int total_frames = duration_seconds * fps;
    const int batch_size = 4;

    Y4MWriter writer(width, height, fps);
    BatchProcessor batch_processor(batch_size);

    const int thread_count = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::cerr << "Using " << thread_count << " CPU cores" << std::endl;

    std::atomic<int> frames_generated{0};
    std::vector<std::thread> producer_threads;
    std::vector<FrameGenerator> generators;

    generators.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
        generators.emplace_back(width, height);
    }

    // 添加计时器
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    int last_frames_written = 0;

    for (int i = 0; i < thread_count; ++i) {
        producer_threads.emplace_back([&, i]() {
            while (true) {
                int current_frame = frames_generated++;
                if (current_frame >= total_frames) break;

                cv::Mat frame = generators[i].generateFrame();
                batch_processor.push(std::move(frame));
            }
        });
    }

    int frames_written = 0;
    std::vector<cv::Mat> batch;
    batch.reserve(batch_size);

    while (frames_written < total_frames) {
        if (batch_processor.pop(batch)) {
            writer.writeFrameBatch(batch);
            frames_written += batch.size();

            // 计算并输出FPS
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

            if (elapsed >= 1000) {  // 每秒更新一次
                double fps = (frames_written - last_frames_written) * 1000.0 / elapsed;
                double average_fps = frames_written * 1000.0 /
                    std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

                fprintf(stderr, "\rProgress: %d%% | Current FPS: %.2f | Average FPS: %.2f",
                    (frames_written * 100) / total_frames, fps, average_fps);

                last_time = current_time;
                last_frames_written = frames_written;
            }
        }
    }

    // 计算总体平均FPS
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    double final_average_fps = total_frames / total_time;

    fprintf(stderr, "\nTotal time: %.2f seconds | Final Average FPS: %.2f\n", total_time, final_average_fps);
    fprintf(stderr, "Done!\n");

    batch_processor.finish();
    for (auto& thread : producer_threads) {
        thread.join();
    }
}

int main() {
    try {
        const int duration = 20;
        generateHighBitrateVideoY4M(duration);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
/*
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
*/
