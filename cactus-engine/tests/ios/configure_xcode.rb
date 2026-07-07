#!/usr/bin/env ruby
require 'xcodeproj'

def fail_with(message)
  STDERR.puts "Error: #{message}"
  exit 1
end

def generate_app_delegate(output_path, test_files)
  raw_level = ENV.fetch('CACTUS_TEST_LOG_LEVEL', 'WARN').to_s.upcase
  level_int = case raw_level
              when 'DEBUG' then 0
              when 'INFO'  then 1
              when 'WARN'  then 2
              when 'ERROR' then 3
              when 'NONE'  then 4
              else 2
              end

  test_names = test_files.map { |f| File.basename(f, '.cpp') }
  extern_declarations = test_names.map { |name| "extern int #{name}_main();" }.join("\n")
  test_calls = test_names.map { |name|
    filter_name = name.sub(/^test_/, '')
    "        if (should_run(\"#{filter_name}\")) failed |= (#{name}_main() != 0);"
  }.join("\n")

  app_delegate_content = <<~OBJC

#import "AppDelegate.h"
#import <TargetConditionals.h>
#import <unistd.h>
#include "cactus_engine.h"
#include <string>

#{extern_declarations}

@implementation AppDelegate

- (void)copyFromBundle:(NSString *)bundlePath toDocuments:(const char *)name {
    if (!name) return;
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *itemName = [NSString stringWithUTF8String:name];
    NSString *sourceItemPath = [NSString stringWithFormat:@"%@/%@", bundlePath, itemName];
    if (![fileManager fileExistsAtPath:sourceItemPath]) {
        fprintf(stderr, "[CactusTest] copyFromBundle: source not found: %s\\n", [sourceItemPath UTF8String]);
        return;
    }
    if ([fileManager fileExistsAtPath:itemName]) {
        NSError *removeError = nil;
        [fileManager removeItemAtPath:itemName error:&removeError];
    }
    NSError *copyError = nil;
    [fileManager copyItemAtPath:sourceItemPath toPath:itemName error:&copyError];
    if (copyError) {
        fprintf(stderr, "[CactusTest] copyFromBundle: failed to copy %s -> %s: %s\\n",
            [sourceItemPath UTF8String], [itemName UTF8String], [[copyError localizedDescription] UTF8String]);
    }
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = paths[0];
    chdir([documentsDirectory UTF8String]);

#if !TARGET_OS_SIMULATOR
    freopen("cactus_test.log", "w", stdout);
    dup2(fileno(stdout), fileno(stderr));
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#endif

    cactus_log_set_level(#{level_int});

    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    [self copyFromBundle:bundlePath toDocuments:getenv("CACTUS_TEST_MODEL")];
    [self copyFromBundle:bundlePath toDocuments:getenv("CACTUS_TEST_TRANSCRIPTION_MODEL")];
    [self copyFromBundle:bundlePath toDocuments:getenv("CACTUS_TEST_ASSETS")];

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NSEC_PER_SEC)), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        const char* only = getenv("CACTUS_TEST_ONLY");
        std::string filter = (only && only[0]) ? std::string(only) : "";
        auto should_run = [&](const char* name) {
            return filter.empty() || filter == name;
        };

        int failed = 0;
        #{test_calls}

        FILE* f = fopen("cactus_test.exitcode", "w");
        if (f) { fprintf(f, "%d\\n", failed); fclose(f); }
        exit(failed);
    });

    return YES;
}

@end

OBJC

  File.write(output_path, app_delegate_content)
  puts "Generated AppDelegate.mm with #{test_names.length} test(s)"
end

project_root = ENV['PROJECT_ROOT']
tests_root = ENV['TESTS_ROOT']
project_path = ENV['XCODEPROJ_PATH']
bundle_id = ENV['BUNDLE_ID']
team_id = ENV['DEVELOPMENT_TEAM']
device_type = ENV['DEVICE_TYPE']

fail_with("PROJECT_ROOT not set") unless project_root
fail_with("TESTS_ROOT not set") unless tests_root
fail_with("XCODEPROJ_PATH not set") unless project_path
fail_with("DEVICE_TYPE not set (should be 'device' or 'simulator')") unless device_type
fail_with("Xcode project not found") unless File.exist?(project_path)

project = Xcodeproj::Project.open(project_path) rescue fail_with("Failed to open Xcode project")
target = project.targets.first or fail_with("No targets found")

tests_group = project.main_group.find_subpath('Tests', true)
tests_group.set_path(tests_root)
tests_group.set_source_tree('<absolute>')

discovered_test_files = Dir.glob(File.join(tests_root, 'test_*.cpp'))
  .reject { |f| File.basename(f) == 'test_utils.cpp' }
  .sort
  .map { |f| File.basename(f) }

puts "Discovered #{discovered_test_files.length} test file(s):"
discovered_test_files.each { |f| puts "  - #{f}" }

test_files = {}
discovered_test_files.each { |f| test_files[f] = "#{File.basename(f, '.cpp')}_main" }
test_files['test_utils.cpp'] = nil

test_files.each do |filename, renamed_main|
  file_path = File.join(tests_root, filename)
  next unless File.exist?(file_path)
  existing = tests_group.files.find { |f| f.path == filename || f.real_path&.to_s == file_path }
  file_ref = existing || begin
    ref = tests_group.new_reference(file_path)
    ref.set_source_tree('<absolute>')
    ref
  end
  build_file = target.source_build_phase.files.find { |bf| bf.file_ref == file_ref }
  build_file ||= target.source_build_phase.add_file_reference(file_ref)
  build_file.settings = { 'COMPILER_FLAGS' => "-Dmain=#{renamed_main}" } if renamed_main
end

unless tests_group.files.any? { |f| f.path == 'test_utils.h' }
  ref = tests_group.new_reference(File.join(tests_root, 'test_utils.h'))
  ref.set_source_tree('<absolute>')
end

generate_app_delegate(File.join(File.dirname(project_path), 'CactusTest', 'AppDelegate.mm'), discovered_test_files)

cactus_sources_group = project.main_group.groups.find { |g| g.name == 'CactusSources' }
if cactus_sources_group
  cactus_sources_group.files.each do |f|
    bf = target.source_build_phase.files.find { |b| b.file_ref == f }
    target.source_build_phase.files.delete(bf) if bf
  end
  cactus_sources_group.clear
  cactus_sources_group.remove_from_project
end

apple_dir = File.join(project_root, 'apple')
static_lib_path = device_type == 'simulator' ?
  File.join(apple_dir, 'libcactus_engine-simulator.a') :
  File.join(apple_dir, 'libcactus_engine-device.a')

fail_with("Static library not found at: #{static_lib_path}") unless File.exist?(static_lib_path)
puts "Using static library: #{static_lib_path}"

cactus_engine_dir = File.join(project_root, 'cactus-engine')
cactus_graph_dir = File.join(project_root, 'cactus-graph')
cactus_kernels_dir = File.join(project_root, 'cactus-kernels')

curl_root = ENV['CACTUS_CURL_ROOT']
vendored_curl_lib = nil
if curl_root && !curl_root.empty?
  vendored_curl_lib = device_type == 'simulator' ?
    File.join(curl_root, 'ios', 'simulator', 'libcurl.a') :
    File.join(curl_root, 'ios', 'device', 'libcurl.a')
  if File.exist?(vendored_curl_lib)
    puts "Using vendored iOS libcurl: #{vendored_curl_lib}"
  else
    vendored_curl_lib = nil
    puts "Vendored iOS libcurl not found under CACTUS_CURL_ROOT=#{curl_root}; continuing without explicit curl link"
  end
end

target.frameworks_build_phase.files.to_a.each do |build_file|
  if build_file.file_ref&.path&.to_s&.include?('libcactus')
    target.frameworks_build_phase.files.delete(build_file)
  end
end

libs_group = project.main_group.groups.find { |g| g.name == 'Frameworks' }
if libs_group
  libs_group.files.to_a.each do |f|
    f.remove_from_project if f.path&.to_s&.include?('libcactus')
  end
end

target.build_configurations.each do |config|
  config.build_settings['HEADER_SEARCH_PATHS'] ||= ['$(inherited)']
  [tests_root, cactus_engine_dir, File.join(cactus_engine_dir, 'libs'), cactus_graph_dir, cactus_kernels_dir, File.join(cactus_kernels_dir, 'src')].each do |path|
    config.build_settings['HEADER_SEARCH_PATHS'] << path unless config.build_settings['HEADER_SEARCH_PATHS'].include?(path)
  end
  if curl_root && !curl_root.empty?
    curl_include = File.join(curl_root, 'include')
    if File.exist?(File.join(curl_include, 'curl', 'curl.h'))
      config.build_settings['HEADER_SEARCH_PATHS'] << curl_include unless config.build_settings['HEADER_SEARCH_PATHS'].include?(curl_include)
    end
  end

  config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++20'
  config.build_settings['CLANG_CXX_LIBRARY'] = 'libc++'
  config.build_settings['GCC_ENABLE_CPP_RTTI'] = 'NO'
  config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
  config.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = bundle_id if bundle_id
  config.build_settings['DEVELOPMENT_TEAM'] = team_id if team_id
  config.build_settings.delete('INFOPLIST_KEY_UILaunchStoryboardName')
  config.build_settings.delete('INFOPLIST_KEY_UIMainStoryboardFile')

  config.build_settings['OTHER_CPLUSPLUSFLAGS'] = ['$(inherited)', '-pthread', '-Wall', '-Wextra', '-pedantic', '-O3']

  config.build_settings['OTHER_LDFLAGS'] ||= ['$(inherited)']
  config.build_settings['OTHER_LDFLAGS'].reject! { |flag| flag.to_s.include?('libcactus') }
  config.build_settings['OTHER_LDFLAGS'] << static_lib_path

  ['-framework CoreML', '-framework Foundation', '-framework Accelerate', '-framework Security', '-framework SystemConfiguration', '-framework CFNetwork', '-framework Metal', '-framework MetalPerformanceShaders'].each do |framework|
    config.build_settings['OTHER_LDFLAGS'] << framework unless config.build_settings['OTHER_LDFLAGS'].include?(framework)
  end
  if vendored_curl_lib
    config.build_settings['OTHER_LDFLAGS'] << vendored_curl_lib unless config.build_settings['OTHER_LDFLAGS'].include?(vendored_curl_lib)
  end
end

project.save rescue fail_with("Failed to save Xcode project")
