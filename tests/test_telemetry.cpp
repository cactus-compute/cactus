#include "test_utils.h"
#include "../cactus/ffi/cactus_cloud.h"
#include "../cactus/telemetry/telemetry.h"
#include "../cactus/kernel/kernel_utils.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <future>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>

std::string make_temp_dir(const char* prefix) {
    char pattern[256] = {0};
    std::snprintf(pattern, sizeof(pattern), "/tmp/%s_XXXXXX", prefix);
    return std::string(mkdtemp(pattern));
}

int count_events(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in.is_open()) return 0;
    int count = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++count;
    }
    return count;
}

std::string random_project_id() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    uint64_t a = rng();
    uint64_t b = rng();
    a = (a & 0xffffffffffff0fffULL) | 0x0000000000004000ULL;
    b = (b & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
    char buf[37];
    std::snprintf(buf, sizeof(buf),
        "%08llx-%04llx-%04llx-%04llx-%012llx",
        (unsigned long long)((a >> 32) & 0xffffffffULL),
        (unsigned long long)((a >> 16) & 0xffffULL),
        (unsigned long long)(a & 0xffffULL),
        (unsigned long long)((b >> 48) & 0xffffULL),
        (unsigned long long)(b & 0xffffffffffffULL));
    return std::string(buf);
}

enum class CloudTelemetryTestResult {
    Passed,
    Failed,
    Skipped,
};

bool test_record_many_then_flush() {
    const std::string cache_dir = make_temp_dir("cactus_record_many_flush");

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(true);
    cactus::telemetry::init("telemetry-test-project", "record-many", nullptr);

    constexpr int expected_event_count = 200;
    for (int i = 0; i < expected_event_count; ++i) {
        cactus::telemetry::recordCompletion("test-model", true, 10.0, 25.0, 30.0, 32, "ok");
    }

    cactus::telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int event_count = count_events(completion_log);

    cactus::telemetry::shutdown();
    rmdir(cache_dir.c_str());
    return event_count == expected_event_count;
}

bool test_shutdown_then_reinit_then_record() {
    const std::string cache_dir = make_temp_dir("cactus_shutdown_reinit");

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(true);
    cactus::telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);

    cactus::telemetry::recordCompletion("test-model", true, 5.0, 20.0, 18.0, 16, "before-shutdown");
    cactus::telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int lines_before_shutdown = count_events(completion_log);

    cactus::telemetry::shutdown();

    cactus::telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);
    cactus::telemetry::recordCompletion("test-model", true, 6.0, 21.0, 19.0, 17, "after-reinit");
    cactus::telemetry::flush();

    const int lines_after_reinit = count_events(completion_log);

    cactus::telemetry::shutdown();
    rmdir(cache_dir.c_str());
    return lines_after_reinit > lines_before_shutdown;
}

bool test_record_and_flush_race_no_deadlock() {
    const std::string cache_dir = make_temp_dir("cactus_telemetry_race");

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(true);
    cactus::telemetry::init("telemetry-test-project", "race-test", nullptr);

    constexpr int producer_tasks = 8;
    constexpr int records_per_task = 200;
    constexpr int flusher_tasks = 4;
    constexpr int flushes_per_task = 40;
    constexpr int expected_event_count = producer_tasks * records_per_task;

    auto& pool = CactusThreading::get_thread_pool();
    std::vector<std::future<void>> futures;
    futures.reserve(producer_tasks + flusher_tasks);

    for (int i = 0; i < producer_tasks; ++i) {
        futures.push_back(pool.enqueue([]() {
            for (int j = 0; j < records_per_task; ++j) {
                cactus::telemetry::recordCompletion("race-model", true, 1.0, 1.0, 2.0, 1, "race");
            }
        }));
    }

    for (int i = 0; i < flusher_tasks; ++i) {
        futures.push_back(pool.enqueue([]() {
            for (int j = 0; j < flushes_per_task; ++j) {
                cactus::telemetry::flush();
            }
        }));
    }

    bool all_completed = true;
    for (auto& future : futures) {
        if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            all_completed = false;
            break;
        }
        future.get();
    }

    cactus::telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int event_count = count_events(completion_log);

    cactus::telemetry::shutdown();
    std::remove(completion_log.c_str());
    rmdir(cache_dir.c_str());

    return all_completed && event_count == expected_event_count;
}

CloudTelemetryTestResult test_cloud_upload_record_then_flush() {
    const char* no_cloud_tele = std::getenv("CACTUS_NO_CLOUD_TELE");
    if (no_cloud_tele && no_cloud_tele[0] != '\0') {
        return CloudTelemetryTestResult::Skipped;
    }

    const std::string telemetry_key = cactus::ffi::resolve_cloud_api_key(nullptr);
    if (telemetry_key.empty()) {
        return CloudTelemetryTestResult::Skipped;
    }

    const std::string cache_dir = make_temp_dir("cactus_cloud_telemetry");
    const std::string completion_log = cache_dir + "/completion.log";
    const std::string project_id = random_project_id();

    cactus::telemetry::setTelemetryEnvironment("cpp", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(false);
    cactus::telemetry::init(project_id.c_str(), "cloud-upload", telemetry_key.c_str());

    cactus::telemetry::recordCompletion("cloud-model", true, 7.0, 19.0, 15.0, 9, "cloud-ok");
    cactus::telemetry::flush();

    const int cached_count = count_events(completion_log);

    cactus::telemetry::shutdown();
    std::remove(completion_log.c_str());
    rmdir(cache_dir.c_str());

    return cached_count == 0 ? CloudTelemetryTestResult::Passed : CloudTelemetryTestResult::Failed;
}

int main() {
    TestUtils::TestRunner runner("Telemetry Tests");
    runner.run_test("Record many then Flush", test_record_many_then_flush());
    runner.run_test("Shutdown then Reinit", test_shutdown_then_reinit_then_record());
    runner.run_test("Record and Flush Race", test_record_and_flush_race_no_deadlock());
    CloudTelemetryTestResult cloud_result = test_cloud_upload_record_then_flush();
    if (cloud_result == CloudTelemetryTestResult::Skipped) {
        runner.log_skip("Cloud record + Flush", "--enable-telemetry and resolved cloud key (env/cache) required");
    } else {
        runner.run_test("Cloud record + Flush", cloud_result == CloudTelemetryTestResult::Passed);
    }
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
