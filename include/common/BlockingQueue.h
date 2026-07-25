#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// 固定容量的单生产者/单消费者环形队列。
// 内部额外保留一个哨兵槽区分满和空，头尾索引始终限制在数组范围内。
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity)
        : capacity_(normalizeCapacity(capacity)),
          storageSize_(checkedStorageSize(capacity_)),
          buffer_(storageSize_) {}

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
        const size_t nextTail = advance(tail);
        const size_t head = head_.load(std::memory_order_acquire);
        if (nextTail == head) {
            return false;
        }

        buffer_[tail] = std::move(item);
        tail_.store(nextTail, std::memory_order_release);

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

        item = std::move(buffer_[head]);
        buffer_[head] = T{};
        head_.store(advance(head), std::memory_order_release);

        return true;
    }

    // 通知等待中的生产者/消费者退出。
    void stop() {
        stopped_.store(true, std::memory_order_release);
    }

    // 清空缓存并重新开放队列。
    // 只能在生产者和消费者都已退出后调用。
    void reset() {
        for (auto& item : buffer_) {
            item = T{};
        }

        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        stopped_.store(false, std::memory_order_release);
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

        return advance(tail) == head;
    }

    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);

        if (tail >= head) {
            return tail - head;
        }
        return storageSize_ - head + tail;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    static size_t normalizeCapacity(size_t capacity) noexcept {
        return capacity == 0 ? 1 : capacity;
    }

    static size_t checkedStorageSize(size_t capacity) {
        if (capacity == std::numeric_limits<size_t>::max()) {
            throw std::length_error("BlockingQueue capacity is too large");
        }
        return capacity + 1;
    }

    // index 最大为 storageSize_ - 1，因此加一不会越过 size_t 上限。
    size_t advance(size_t index) const noexcept {
        const size_t next = index + 1;
        return next == storageSize_ ? 0 : next;
    }

    static void backoff() {
        std::this_thread::yield();
    }

private:
    const size_t capacity_;
    const size_t storageSize_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<bool> stopped_{false};
};
