#include "infer/TensorRTCommon.h"

#include "utils/FileLogger.h"

#include <iostream>
#include <string>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > Severity::kWARNING) {
        return;
    }

    const std::string message = std::string("[TensorRT] ") + (msg == nullptr ? "" : msg);

    if (utils::FileLogger::instance().isOpen()) {
        if (severity <= Severity::kERROR) {
            utils::FileLogger::instance().error(message);
        } else {
            utils::FileLogger::instance().warning(message);
        }
    } else {
        std::cout << message << std::endl;
    }
}
