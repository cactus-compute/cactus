#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CACTUS_CURL_ROOT="${CACTUS_CURL_ROOT:-$PROJECT_ROOT/cactus-engine/libs/curl}"
export CACTUS_CURL_ROOT

for var in CACTUS_TEST_MODEL CACTUS_TEST_TRANSCRIPTION_MODEL; do
    if [ -z "${!var:-}" ] || [ ! -d "${!var}" ]; then
        echo "Error: $var must point to a prepared bundle (got: '${!var:-}')" >&2
        exit 1
    fi
done

echo "Running Cactus tests on Android..."
echo "============================"

if ! command -v adb &>/dev/null; then
    echo "adb not found"
    echo "Install Android SDK Platform Tools and ensure it's in your PATH"
    echo "Installation:"
    echo "  - Via Android Studio: Settings > Android SDK > SDK Tools > Android SDK Platform-Tools"
    exit 1
fi

echo ""
echo "Step 1: Selecting Android device..."

adb start-server

connected_devices=$(adb devices | grep -E "device$|emulator" | grep -v "^List" | awk '{print $1}')

if command -v emulator &>/dev/null; then
    available_emulators=$(emulator -list-avds | grep -v "^INFO" || true)
else
    available_emulators=""
fi

all_devices=""

if [ -n "$connected_devices" ]; then
    while read -r device_id; do
        if [[ "$device_id" == emulator-* ]]; then
            device_model=$(adb -s "$device_id" shell getprop ro.product.model | tr -d '\r')
            device_android=$(adb -s "$device_id" shell getprop ro.build.version.release | tr -d '\r')
            avd_name=$(adb -s "$device_id" shell getprop ro.kernel.qemu.avd_name | tr -d '\r')
            if [ -n "$avd_name" ]; then
                all_devices=$(printf "%s\n%s|emulator|%s|running|%s|%s" "$all_devices" "$avd_name" "$device_id" "$device_model" "$device_android")
            else
                all_devices=$(printf "%s\n%s|emulator|%s|running|%s|%s" "$all_devices" "$device_id" "$device_id" "$device_model" "$device_android")
            fi
        else
            device_model=$(adb -s "$device_id" shell getprop ro.product.model | tr -d '\r')
            device_android=$(adb -s "$device_id" shell getprop ro.build.version.release | tr -d '\r')
            all_devices=$(printf "%s\n%s|device|%s|running|%s|%s" "$all_devices" "$device_model" "$device_id" "$device_model" "$device_android")
        fi
    done <<< "$connected_devices"
fi

if [ -n "$available_emulators" ]; then
    while read -r avd_name; do
        if ! echo "$all_devices" | grep -q "^$avd_name|emulator|"; then
            all_devices=$(printf "%s\n%s|emulator||offline||" "$all_devices" "$avd_name")
        fi
    done <<< "$available_emulators"
fi

all_devices=$(echo "$all_devices" | grep -v '^$')

if [ -z "$all_devices" ]; then
    echo "No devices or emulators found"
    echo "To use an emulator:"
    echo "  - Create one through Android Studio AVD Manager"
    echo "  - Or start one with: emulator -avd <avd_name>"
    echo "To use a physical device:"
    echo "  - Connect via USB"
    echo "  - Enable USB debugging in Developer Options"
    echo "  - Authorize the computer when prompted"
    exit 1
fi

physical_count=$(echo "$all_devices" | grep -c '|device|' || true)
device_num=0

if [ "$physical_count" -gt 0 ]; then
    echo "Devices:"
    while IFS='|' read -r name type device_id status model android_version; do
        if [ "$type" = "device" ]; then
            device_num=$((device_num + 1))
            if [ -n "$android_version" ]; then
                printf "  %2d. %s (Android %s)\n" "$device_num" "$name" "$android_version"
            else
                printf "  %2d. %s\n" "$device_num" "$name"
            fi
        fi
    done <<< "$all_devices"
    echo ""
fi

echo "Emulators:"
while IFS='|' read -r name type device_id status model android_version; do
    if [ "$type" = "emulator" ]; then
        device_num=$((device_num + 1))
        if [ "$status" = "offline" ]; then
            printf "  %2d. %s [not running]\n" "$device_num" "$name"
        else
            if [ -n "$android_version" ]; then
                printf "  %2d. %s (Android %s)\n" "$device_num" "$name" "$android_version"
            else
                printf "  %2d. %s\n" "$device_num" "$name"
            fi
        fi
    fi
done <<< "$all_devices"

echo ""
read -p "Select device number (1-$device_num): " device_number

if ! [[ "$device_number" =~ ^[0-9]+$ ]] || [ "$device_number" -lt 1 ] || [ "$device_number" -gt "$device_num" ]; then
    echo "Invalid selection"
    exit 1
fi

selected_line=$(echo "$all_devices" | sed -n "${device_number}p")
device_name=$(echo "$selected_line" | cut -d'|' -f1)
device_type=$(echo "$selected_line" | cut -d'|' -f2)
DEVICE_ID=$(echo "$selected_line" | cut -d'|' -f3)
device_status=$(echo "$selected_line" | cut -d'|' -f4)

if [ -z "$device_name" ]; then
    echo "Could not parse device information"
    exit 1
fi

echo ""
if [ "$device_type" = "emulator" ]; then
    if [ "$device_status" = "offline" ]; then
        echo "Selected: $device_name (Emulator - Not Running)"
        echo "Starting emulator..."

        if ! command -v emulator &>/dev/null; then
            echo "Emulator command not found"
            echo "Add Android SDK emulator to your PATH"
            exit 1
        fi

        emulator -avd "$device_name" -no-snapshot-load &
        emulator_pid=$!

        echo "Waiting for emulator to boot..."
        timeout=300
        elapsed=0
        while [ $elapsed -lt $timeout ]; do
            sleep 2
            elapsed=$((elapsed + 2))

            DEVICE_ID=$(adb devices | grep emulator | head -1 | awk '{print $1}')

            if [ -n "$DEVICE_ID" ]; then
                boot_complete=$(adb -s "$DEVICE_ID" shell getprop sys.boot_completed | tr -d '\r')
                if [ "$boot_complete" = "1" ]; then
                    echo "Emulator started successfully"
                    break
                fi
            fi
        done

        if [ -z "$DEVICE_ID" ] || [ "$boot_complete" != "1" ]; then
            echo "Failed to start emulator (timeout)"
            exit 1
        fi
    else
        echo "Selected: $device_name (Emulator)"
    fi
else
    echo "Selected: $device_name (Device)"
fi

if ! adb -s "$DEVICE_ID" shell echo "test" &>/dev/null; then
    echo "Device not responding"
    echo "Ensure the device is connected and USB debugging is authorized"
    exit 1
fi

if [ -z "$ANDROID_NDK_HOME" ]; then
    if [ -n "$ANDROID_HOME" ]; then
        ANDROID_NDK_HOME=$(ls -d "$ANDROID_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)
    elif [ -d "$HOME/Library/Android/sdk" ]; then
        ANDROID_NDK_HOME=$(ls -d "$HOME/Library/Android/sdk/ndk/"* 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "Android NDK not found"
    echo "Install the NDK through Android Studio: Settings > Android SDK > SDK Tools > NDK"
    exit 1
fi

cmake_toolchain="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
android_platform=${ANDROID_PLATFORM:-android-21}
n_jobs=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo ""
echo "Step 2: Building Cactus library for Android..."

if ! "$PROJECT_ROOT/android/build.sh"; then
    echo "Failed to build Cactus library"
    exit 1
fi

echo ""
echo "Step 3: Building Android tests..."

android_test_dir="$SCRIPT_DIR"
android_build_dir="$android_test_dir/build"

rm -rf "$android_build_dir"
mkdir -p "$android_build_dir"

if ! cmake -S "$android_test_dir" -B "$android_build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$cmake_toolchain" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM="$android_platform" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCACTUS_CURL_ROOT="$CACTUS_CURL_ROOT" \
    -DCMAKE_RULE_MESSAGES=OFF \
    -DCMAKE_VERBOSE_MAKEFILE=OFF; then
    echo "Failed to configure tests"
    exit 1
fi

if ! cmake --build "$android_build_dir" -j "$n_jobs"; then
    echo "Failed to build tests"
    exit 1
fi

echo "Discovering test executables..."
all_test_executables=($(find "$android_build_dir" -maxdepth 1 -name "test_*" -type f | sort))

if [ ${#all_test_executables[@]} -eq 0 ]; then
    echo "No test executables found"
    exit 1
fi

test_executables=()
for test_exe in "${all_test_executables[@]}"; do
    test_name=$(basename "$test_exe" | sed 's/^test_//')
    if [ -z "${CACTUS_TEST_SUITE:-}" ] || [ "$test_name" = "$CACTUS_TEST_SUITE" ]; then
        test_executables+=("$test_exe")
    fi
done

if [ ${#test_executables[@]} -eq 0 ]; then
    echo "No test executables match filter: $CACTUS_TEST_SUITE"
    exit 1
fi

echo "Found ${#test_executables[@]} test executable(s)"

echo ""
echo "Step 4: Deploying to device..."

model_dir=$(basename "$CACTUS_TEST_MODEL")
transcription_dir=$(basename "$CACTUS_TEST_TRANSCRIPTION_MODEL")
assets_src="$PROJECT_ROOT/cactus-engine/tests/assets"

device_test_dir="/data/local/tmp/cactus_tests"
device_model_dir="/data/local/tmp/cactus_models"
device_assets_dir="/data/local/tmp/cactus_assets"

adb -s "$DEVICE_ID" shell "rm -rf $device_test_dir $device_model_dir $device_assets_dir"
adb -s "$DEVICE_ID" shell "mkdir -p $device_test_dir $device_model_dir $device_assets_dir"

echo "Pushing model weights..."
if ! adb -s "$DEVICE_ID" push "$CACTUS_TEST_MODEL" "$device_model_dir/" >/dev/null; then
    echo "Failed to push model weights"
    exit 1
fi

echo "Pushing transcription model..."
if ! adb -s "$DEVICE_ID" push "$CACTUS_TEST_TRANSCRIPTION_MODEL" "$device_model_dir/" >/dev/null; then
    echo "Failed to push transcription model"
    exit 1
fi

if [ -d "$assets_src" ]; then
    echo "Pushing test assets..."
    if ! adb -s "$DEVICE_ID" push "$assets_src" "$device_assets_dir/" >/dev/null; then
        echo "Failed to push test assets"
        exit 1
    fi
fi

echo "Pushing test executables..."
for test_exe in "${test_executables[@]}"; do
    test_name=$(basename "$test_exe")
    if ! adb -s "$DEVICE_ID" push "$test_exe" "$device_test_dir/" >/dev/null; then
        echo "Failed to push $test_name"
        exit 1
    fi
    adb -s "$DEVICE_ID" shell "chmod +x $device_test_dir/$test_name"
done

echo ""
echo "Step 5: Running tests..."
echo "------------------------"
echo "Using model path:               $device_model_dir/$model_dir"
echo "Using transcription model path: $device_model_dir/$transcription_dir"
echo "Using assets path:              $device_assets_dir/assets"

FAILED=0
for test_exe in "${test_executables[@]}"; do
    test_name=$(basename "$test_exe")
    echo ""
    echo "Running $test_name..."

    if ! adb -s "$DEVICE_ID" shell "cd $device_test_dir && \
        export CACTUS_TEST_MODEL=$device_model_dir/$model_dir && \
        export CACTUS_TEST_TRANSCRIPTION_MODEL=$device_model_dir/$transcription_dir && \
        export CACTUS_TEST_ASSETS=$device_assets_dir/assets && \
        export CACTUS_INDEX_PATH=$device_assets_dir/assets && \
        export CACTUS_NO_CLOUD_TELE=${CACTUS_NO_CLOUD_TELE:-1} && \
        export CACTUS_TEST_BACKEND=${CACTUS_TEST_BACKEND:-auto} && \
        ./$test_name"; then
        FAILED=1
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All tests passed."
else
    echo "Some tests failed."
fi

exit $FAILED
