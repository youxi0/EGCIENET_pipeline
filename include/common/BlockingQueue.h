#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

// 固定容量的单生产者/单消费者环形队列。
// 内部使用原子读写索引，不依赖互斥锁或条件变量。
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity)
        : capacity_(capacity == 0 ? 1 : capacity),
          buffer_(capacity_) {}

    BlockingQueue(const BlockingQueue&) = delete;
    BlockingQueue& operator=(const BlockingQueue&) = delete;

    // 阻塞式拷贝入队；队列满时短暂让出 CPU。
    bool push(const T& item) {
        T copy = item;
        return push(std::move(copy));
    }

    // 阻塞式移动入队；队列满时短暂让出 CPU。
    bool push(T&& item) {
        while (!stopped_.load(std::memory_order_acquire)) {
            if (tryPush(std::move(item))) {
                return true;
            }

            backoff();
        }

        return false;
    }

    // 阻塞式出队；队列空时短暂让出 CPU 等待新数据。
    bool pop(T& item) {
        while (true) {
            if (tryPop(item)) {
                return true;
            }

            if (stopped_.load(std::memory_order_acquire) && empty()) {
                return false;
            }

            backoff();
        }
    }

    // 非阻塞移动入队；队列满或已停止时返回 false。
    bool tryPush(T&& item) {
        if (stopped_.load(std::memory_order_acquire)) {
            return false;
        }

        const size_t tail = tail_.load(std::memory_order_relaxed);
        const size_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= capacity_) {
            return false;
        }

        buffer_[tail % capacity_] = std::move(item);
        tail_.store(tail + 1, std::memory_order_release);

        return true;
    }

    // 非阻塞拷贝入队；队列满或已停止时返回 false。
    bool tryPush(const T& item) {
        T copy = item;
        return tryPush(std::move(copy));
    }

    // 非阻塞出队；队列为空时返回 false。
    bool tryPop(T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_acquire);
        if (head == tail) {
            return false;
        }

        item = std::move(buffer_[head % capacity_]);
        buffer_[head % capacity_] = T{};
        head_.store(head + 1, std::memory_order_release);

        return true;
    }

    // 通知等待中的生产者/消费者退出。
    void stop() {
        stopped_.store(true, std::memory_order_release);
    }

    // 清空缓存并重新开放队列。
    void reset() {
        stopped_.store(false, std::memory_order_release);
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);

        for (auto& item : buffer_) {
            item = T{};
        }
    }

    bool stopped() const {
        return stopped_.load(std::memory_order_acquire);
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    bool full() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);

        return tail - head >= capacity_;
    }

    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);

        return tail - head;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    static void backoff() {
        std::this_thread::yield();
    }

private:
    const size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<bool> stopped_{false};
};
