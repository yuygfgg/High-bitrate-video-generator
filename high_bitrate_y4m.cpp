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
#include <string>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

struct VideoConfig {
    int width = 3840;
    int height = 2160;
    int fps = 60;
    int duration = 20;
    int batch_size = 4;
    int num_lines = 50;
    int min_thickness = 1;
    int max_thickness = 9;
    int thread_count = -1;
    uint32_t random_seed = 0;
};

void printUsage(const char* programName) {
    fprintf(stderr, "Usage: %s [options]\n"
            "Options:\n"
            "  --width N      Video width (default: 3840)\n"
            "  --height N     Video height (default: 2160)\n"
            "  --fps N        Frames per second (default: 60)\n"
            "  --duration N   Duration in seconds (default: 20)\n"
            "  --batch N      Batch size for frame processing (default: 4)\n"
            "  --lines N      Number of random lines per frame (default: 50)\n"
            "  --min-thick N  Minimum line thickness (default: 1)\n"
            "  --max-thick N  Maximum line thickness (default: 9)\n"
            "  --threads N    Number of threads (default: auto)\n"
            "  --seed N       Random seed (default: random)\n"
            "  --help         Show this help message\n",
            programName);
}

bool parseCommandLine(int argc, char* argv[], VideoConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        
        if (i + 1 >= argc) {
            fprintf(stderr, "Error: Missing value for %s\n", arg.c_str());
            return false;
        }
        
        int value = std::atoi(argv[i + 1]);
        
        if (arg == "--width") config.width = value;
        else if (arg == "--height") config.height = value;
        else if (arg == "--fps") config.fps = value;
        else if (arg == "--duration") config.duration = value;
        else if (arg == "--batch") config.batch_size = value;
        else if (arg == "--lines") config.num_lines = value;
        else if (arg == "--min-thick") config.min_thickness = value;
        else if (arg == "--max-thick") config.max_thickness = value;
        else if (arg == "--threads") config.thread_count = value;
        else if (arg == "--seed") config.random_seed = static_cast<uint32_t>(value);
        else {
            fprintf(stderr, "Error: Unknown option %s\n", arg.c_str());
            return false;
        }
        
        i++;
    }
    
    if (config.width <= 0 || config.height <= 0 || config.fps <= 0 || 
        config.duration <= 0 || config.batch_size <= 0 || config.num_lines < 0 ||
        config.min_thickness <= 0 || config.max_thickness < config.min_thickness) {
        fprintf(stderr, "Error: Invalid parameter values\n");
        return false;
    }
    
    return true;
}

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
        uint32_t temp_state[16];
        std::memcpy(temp_state, state_.data(), sizeof(temp_state));
        
    #if defined(__ARM_NEON__)
        for (; i + 16 <= count; i += 16) {
            uint32x4_t x = vld1q_u32(&temp_state[index_]);
            x = veorq_u32(x, vshlq_n_u32(x, 13));
            x = veorq_u32(x, vshrq_n_u32(x, 17));
            x = veorq_u32(x, vshlq_n_u32(x, 5));
            vst1q_u32(&temp_state[index_], x);
            vst1q_u8(dest + i, vreinterpretq_u8_u32(x));
            index_ = (index_ + 4) & 15;
        }
    #elif defined(__SSE2__)
        for (; i + 16 <= count; i += 16) {
            __m128i x = _mm_load_si128((__m128i*)&temp_state[index_]);
            x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
            x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
            x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
            _mm_store_si128((__m128i*)&temp_state[index_], x);
            _mm_storeu_si128((__m128i*)(dest + i), x);
            index_ = (index_ + 4) & 15;
        }
    #endif

        while (i < count) {
            uint32_t& x = temp_state[index_];
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            
            size_t bytes_to_copy = std::min(size_t(4), count - i);
            std::memcpy(dest + i, &x, bytes_to_copy);
            
            i += bytes_to_copy;
            index_ = (index_ + 1) & 15;
        }

        std::memcpy(state_.data(), temp_state, sizeof(temp_state));
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
    const int num_lines_;
    const int min_thickness_;
    const int max_thickness_;
    FastRandom rng_;
    std::vector<uint8_t> buffer_;

public:
    FrameGenerator(const VideoConfig& config, uint32_t seed)
        : width_(config.width)
        , height_(config.height)
        , num_lines_(config.num_lines)
        , min_thickness_(config.min_thickness)
        , max_thickness_(config.max_thickness)
        , rng_(seed)
        , buffer_(width_ * height_ * 3)
    {}

    cv::Mat generateFrame() {
        rng_.generate_bytes(buffer_.data(), buffer_.size());
        cv::Mat frame(height_, width_, CV_8UC3, buffer_.data());

        for (int i = 0; i < num_lines_; ++i) {
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
            cv::line(frame, pt1, pt2, color, 
                    rng_.generate_int(min_thickness_, max_thickness_));
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

void generateHighBitrateVideoY4M(const VideoConfig& config) {
    const int total_frames = config.duration * config.fps;
    
    Y4MWriter writer(config.width, config.height, config.fps);
    BatchProcessor batch_processor(config.batch_size);

    const int thread_count = config.thread_count > 0 ? 
        config.thread_count : 
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    
    std::cerr << "Using " << thread_count << " CPU cores" << std::endl;

    std::atomic<int> frames_generated{0};
    std::vector<std::thread> producer_threads;
    std::vector<FrameGenerator> generators;

    generators.reserve(thread_count);
    std::random_device rd;
    uint32_t base_seed = config.random_seed == 0 ? rd() : config.random_seed;
    for (int i = 0; i < thread_count; ++i) {
        generators.emplace_back(config, base_seed + i);
    }

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
    batch.reserve(config.batch_size);

    while (frames_written < total_frames) {
        if (batch_processor.pop(batch)) {
            writer.writeFrameBatch(batch);
            frames_written += batch.size();

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();

            if (elapsed >= 1000) {
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

int main(int argc, char* argv[]) {
    try {
        VideoConfig config;
        if (!parseCommandLine(argc, argv, config)) {
            return 1;
        }
        generateHighBitrateVideoY4M(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
