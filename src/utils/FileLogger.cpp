#include "utils/FileLogger.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace utils {

FileLogger& FileLogger::instance() {
    static FileLogger logger;
    return logger;
}

FileLogger::~FileLogger() {
    close();
}

bool FileLogger::open(const std::string& logDir, const std::string& runName) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (file_.is_open()) {
        file_.close();
    }

    std::error_code ec;
    fs::create_directories(logDir, ec);
    if (ec) {
        std::cerr << "[ERROR] failed to create log directory: "
                  << logDir << ", " << ec.message() << std::endl;
        return false;
    }

    logPath_ = buildLogPath(logDir, runName);
    file_.open(logPath_, std::ios::out);

    if (!file_.is_open()) {
        std::cerr << "[ERROR] failed to open log file: "
                  << logPath_ << std::endl;
        logPath_.clear();
        return false;
    }

    // 新增模块: log文件第一行记录本次运行的落盘路径，方便排查多次实验结果。
    file_ << timestamp() << " [INFO] log file opened: " << logPath_ << "\n";
    file_.flush();

    std::cout << "[INFO] log file: " << logPath_ << std::endl;
    return true;
}

void FileLogger::close() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (file_.is_open()) {
        file_ << timestamp() << " [INFO] log file closed\n";
        file_.close();
    }
}

void FileLogger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << timestamp() << " [" << levelName(level) << "] " << message;
    const std::string line = oss.str();

    if (level == LogLevel::kWarning || level == LogLevel::kError) {
        std::cerr << line << std::endl;
    } else {
        std::cout << line << std::endl;
    }

    if (file_.is_open()) {
        file_ << line << "\n";
        file_.flush();
    }
}

void FileLogger::debug(const std::string& message) {
    log(LogLevel::kDebug, message);
}

void FileLogger::info(const std::string& message) {
    log(LogLevel::kInfo, message);
}

void FileLogger::warning(const std::string& message) {
    log(LogLevel::kWarning, message);
}

void FileLogger::error(const std::string& message) {
    log(LogLevel::kError, message);
}

bool FileLogger::isOpen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return file_.is_open();
}

std::string FileLogger::logPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return logPath_;
}

std::string FileLogger::buildLogPath(const std::string& logDir, const std::string& runName) const {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};

#if defined(_WIN32)
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    std::ostringstream name;
    name << runName << "_"
         << std::put_time(&localTime, "%Y%m%d_%H%M%S")
         << ".log";

    return (fs::path(logDir) / name.str()).string();
}

std::string FileLogger::timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;

    std::tm localTime{};

#if defined(_WIN32)
    localtime_s(&localTime, &time);
#else
    localtime_r(&time, &localTime);
#endif

    std::ostringstream oss;
    oss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << millis.count();

    return oss.str();
}

const char* FileLogger::levelName(LogLevel level) const {
    switch (level) {
    case LogLevel::kDebug:
        return "DEBUG";
    case LogLevel::kInfo:
        return "INFO";
    case LogLevel::kWarning:
        return "WARN";
    case LogLevel::kError:
        return "ERROR";
    default:
        return "INFO";
    }
}

} // namespace utils
