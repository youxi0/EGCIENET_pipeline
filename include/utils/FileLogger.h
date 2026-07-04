#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace utils {

enum class LogLevel {
    kDebug,
    kInfo,
    kWarning,
    kError
};

// 新增模块: 线程安全的日志保存器，用于把运行日志同步写入控制台和本地log文件。
class FileLogger {
public:
    static FileLogger& instance();

    FileLogger(const FileLogger&) = delete;
    FileLogger& operator=(const FileLogger&) = delete;

    bool open(const std::string& logDir, const std::string& runName = "blade_pipeline");
    void close();

    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

    bool isOpen() const;
    std::string logPath() const;

private:
    FileLogger() = default;
    ~FileLogger();

    std::string buildLogPath(const std::string& logDir, const std::string& runName) const;
    std::string timestamp() const;
    const char* levelName(LogLevel level) const;

private:
    mutable std::mutex mutex_;
    std::ofstream file_;
    std::string logPath_;
};

} // namespace utils
