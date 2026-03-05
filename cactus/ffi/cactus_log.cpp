#include "cactus_ffi.h"
#include "graph/graph.h"

extern "C" {

void cactus_log_set_level(int level) {
    cactus::Logger::instance().set_level(static_cast<cactus::LogLevel>(level));
}

void cactus_log_set_callback(cactus_log_callback_t callback, void* user_data) {
    if (callback) {
        cactus::Logger::instance().set_callback(
            [callback, user_data](cactus::LogLevel level,
                                  const std::string& component,
                                  const std::string& message) {
                callback(static_cast<int>(level), component.c_str(), message.c_str(), user_data);
            });
    } else {
        cactus::Logger::instance().set_callback(nullptr);
    }
}

}
