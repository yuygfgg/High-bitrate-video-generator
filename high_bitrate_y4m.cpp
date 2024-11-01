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
#include <arm_neon.h>

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
        while (count >= 16) {
            uint32x4_t a = vld1q_u32(&state_[index_]);
            uint32x4_t b = vld1q_u32(&state_[index_ + 4]);

            // xorshift
            a = veorq_u32(a, vshlq_n_u32(a, 13));
            a = veorq_u32(a, vshrq_n_u32(a, 17));
            a = veorq_u32(a, vshlq_n_u32(a, 5));

            b = veorq_u32(b, vshlq_n_u32(b, 13));
            b = veorq_u32(b, vshrq_n_u32(b, 17));
            b = veorq_u32(b, vshlq_n_u32(b, 5));

            vst1q_u32(&state_[index_], a);
            vst1q_u32(&state_[index_ + 4], b);

            // 将32位数压缩到8位
            uint16x4_t narrow1 = vmovn_u32(a);
            uint16x4_t narrow2 = vmovn_u32(b);
            uint16x8_t combined = vcombine_u16(narrow1, narrow2);
            uint8x8_t narrow3 = vmovn_u16(combined);
            uint8x16_t result = vcombine_u8(narrow3, narrow3);

            vst1q_u8(dest, result);

            dest += 16;
            count -= 16;
            index_ = (index_ + 8) & 15;
        }

        // 处理剩余字节
        if (count > 0) {
            uint32x4_t a = vld1q_u32(&state_[index_]);
            a = veorq_u32(a, vshlq_n_u32(a, 13));
            a = veorq_u32(a, vshrq_n_u32(a, 17));
            a = veorq_u32(a, vshlq_n_u32(a, 5));
            vst1q_u32(&state_[index_], a);

            uint16x4_t narrow1 = vmovn_u32(a);
            uint16x8_t combined = vcombine_u16(narrow1, narrow1);
            uint8x8_t narrow2 = vmovn_u16(combined);
            uint8x16_t result = vcombine_u8(narrow2, narrow2);

            uint8_t temp[16];
            vst1q_u8(temp, result);
            std::memcpy(dest, temp, count);
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
        rng.generate_bytes(buffer.data(), buffer.size());

        cv::Mat frame(height_, width_, CV_8UC3, buffer.data());

        // 添加随机线条
        for (int i = 0; i < 50; ++i) {
            cv::Point pt1(
                rng.generate_int(0, width_ - 1),
                rng.generate_int(0, height_ - 1)
            );
            cv::Point pt2(
                rng.generate_int(0, width_ - 1),
                rng.generate_int(0, height_ - 1)
            );

            cv::Scalar color(
                rng.generate_int(0, 255),
                rng.generate_int(0, 255),
                rng.generate_int(0, 255)
            );

            cv::line(frame, pt1, pt2, color, rng.generate_int(1, 9));
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
        producer_cv.wait(lock, [this] {
            return (produced_ - consumed_) < batch_size_;
        });

        frames_[produced_ % batch_size_] = std::move(frame);
        produced_++;

        if (produced_ - consumed_ >= batch_size_) {
            consumer_cv.notify_one();
        }
    }

    bool pop(std::vector<cv::Mat>& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        consumer_cv.wait(lock, [this] {
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
        producer_cv.notify_all();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        consumer_cv.notify_all();
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
        const int duration = 15;
        generateHighBitrateVideoY4M(duration);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
