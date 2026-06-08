#include "../cactus_graph.h"
#include <string>

std::string last_error_message;

extern "C" const char* cactus_get_last_error() {
    return last_error_message.c_str();
}
