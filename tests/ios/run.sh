#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACTUS_CURL_ROOT="${CACTUS_CURL_ROOT:-$PROJECT_ROOT/libs/curl}"
export CACTUS_CURL_ROOT

MODEL_NAME="$1"
TRANSCRIBE_MODEL_NAME="$2"
VAD_MODEL_NAME="$3"
RUN_ASR="${CACTUS_RUN_ASR:-0}"
ASR_AUDIO_SOURCE="${CACTUS_ASR_AUDIO_SOURCE:-}"
ASR_AUDIO_FILE="${CACTUS_ASR_AUDIO_FILE:-}"

echo "Running Cactus tests on iOS..."
echo "============================"

if [ ! -d "/Applications/Xcode.app" ]; then
    echo "Xcode not installed"
    echo "Install Xcode from the Mac App Store"
    exit 1
fi

if ! xcode-select -p; then
    echo "Xcode Command Line Tools not installed"
    echo "Install with: xcode-select --install"
    exit 1
fi

if ! /usr/bin/xcrun --version; then
    echo "Xcode license not accepted"
    echo "Accept the license with: sudo xcodebuild -license accept"
    exit 1
fi

if ! command -v xcodebuild; then
    echo "xcodebuild not found"
    exit 1
fi

echo ""
echo "Step 1: Selecting iOS device..."

simulators=$(xcrun simctl list devices available | grep -E "^\s+(iPhone|iPad)" | grep -v "unavailable" | sed 's/^[[:space:]]*//' | while read line; do
    uuid=$(echo "$line" | grep -oE '\([A-F0-9-]{36}\)' | head -1 | tr -d '()')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | sed -E 's/ \([^)]*\)//g' | xargs)
        echo "${name}|simulator|${uuid}"
    fi
done)

xctrace_output=$(xcrun xctrace list devices 2>&1)

physical_devices=$(echo "$xctrace_output" | awk '
    /== Devices ==/ { in_online=1; in_offline=0; next }
    /== Devices Offline ==/ { in_online=0; in_offline=1; next }
    /== Simulators ==/ { exit }
    /00008[A-F0-9]{3}-[A-F0-9]{16}/ {
        if (in_online || in_offline) {
            status = in_offline ? "offline" : ""
            print $0 "|" status
        }
    }
' | while read line; do
    uuid=$(echo "$line" | grep -oE '00008[A-F0-9]{3}-[A-F0-9]{16}')
    status=$(echo "$line" | awk -F'|' '{print $2}')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | awk -F'|' '{print $1}' | sed -E 's/ \([0-9]+\.[0-9]+.*$//' | xargs)
        ios_version=$(echo "$line" | awk -F'|' '{print $1}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        echo "${name} (iOS ${ios_version})|device|${uuid}|${status}"
    fi
done)

all_devices=$(printf "%s\n%s\n" "$physical_devices" "$simulators" | grep -v '^$')

if [ -z "$all_devices" ]; then
    echo "No devices or simulators found"
    echo "To use a simulator:"
    echo "  - Install iOS simulators through Xcode"
    echo "To use a physical device:"
    echo "  - Connect via USB or network"
    echo "  - Enable Developer Mode in Settings"
    exit 1
fi

physical_count=$(echo "$all_devices" | grep -c '|device|' || true)
device_num=0

if [ "$physical_count" -gt 0 ]; then
    echo "Devices:"
    while IFS='|' read -r name type uuid status; do
        if [ "$type" = "device" ]; then
            device_num=$((device_num + 1))
            if [ "$status" = "offline" ]; then
                printf "  %2d. %s [offline]\n" "$device_num" "$name"
            else
                printf "  %2d. %s\n" "$device_num" "$name"
            fi
        fi
    done <<< "$all_devices"
    echo ""
fi

echo "Simulators:"
while IFS='|' read -r name type uuid status; do
    if [ "$type" = "simulator" ]; then
        device_num=$((device_num + 1))
        printf "  %2d. %s\n" "$device_num" "$name"
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
device_uuid=$(echo "$selected_line" | cut -d'|' -f3)
device_status=$(echo "$selected_line" | cut -d'|' -f4)

if [ -z "$device_uuid" ]; then
    echo "Could not parse device information"
    exit 1
fi

echo ""
if [ "$device_type" = "simulator" ]; then
    echo "Selected: $device_name (Simulator)"
else
    if [ "$device_status" = "offline" ]; then
        echo "Selected: $device_name (Device - Offline)"
        echo "Warning: This device is currently offline"
        echo "Ensure the device is:"
        echo "  - Connected via USB or network"
        echo "  - Unlocked and trusted"
        echo "  - Has Developer Mode enabled"
        echo ""
        read -p "Continue anyway? (y/N): " continue_choice
        if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
            echo "Aborted"
            exit 0
        fi
    else
        echo "Selected: $device_name (Device)"
    fi

    if ! security find-identity -v -p codesigning | grep -q "Apple Development"; then
        echo "No development certificates found"
        echo "To fix this:"
        echo "  1. Open Xcode > Settings > Accounts"
        echo "  2. Add your Apple ID"
        echo "  3. Download development certificates"
        exit 1
    fi
fi

echo ""
echo "Step 2: Building Cactus library for iOS..."

if ! BUILD_STATIC=true BUILD_XCFRAMEWORK=false "$PROJECT_ROOT/apple/build.sh"; then
    echo "Failed to build Cactus library"
    exit 1
fi

echo ""
echo "Step 3: Configuring Xcode project..."

xcodeproj_path="$SCRIPT_DIR/CactusTest/CactusTest.xcodeproj"
tests_root="$(cd "$SCRIPT_DIR/.." && pwd)"
cactus_root="$PROJECT_ROOT/cactus"

project_file="$xcodeproj_path/project.pbxproj"
template_file="$xcodeproj_path/project.pbxproj.template"
echo "Copying project template..."
cp "$template_file" "$project_file"

bundle_id="com.cactus.test.${USER}"
echo "Using Bundle ID: $bundle_id"

if [ "$device_type" = "device" ]; then
    development_team=$(security find-certificate -a -c "Apple Development" -p | openssl x509 -noout -subject | grep -oE 'OU=[A-Z0-9]{10}' | head -1 | cut -d= -f2)
    if [ -z "$development_team" ]; then
        echo "Could not extract Team ID from certificate"
        exit 1
    fi
    echo "Using Team ID: $development_team"
fi

if ! command -v ruby; then
    echo "Ruby not found"
    exit 1
fi

if ! gem list xcodeproj -i; then
    echo "Installing xcodeproj gem..."
    if ! gem install xcodeproj; then
        echo "Failed to install xcodeproj gem"
        echo "To fix this:"
        echo "  brew install rbenv ruby-build"
        echo "  rbenv init"
        echo "  rbenv install 2.6.10"
        echo "  rbenv global 2.6.10"
        exit 1
    fi
fi

export PROJECT_ROOT TESTS_ROOT="$tests_root" CACTUS_ROOT="$cactus_root" XCODEPROJ_PATH="$xcodeproj_path" BUNDLE_ID="$bundle_id" DEVELOPMENT_TEAM="$development_team" DEVICE_TYPE="$device_type" CACTUS_CURL_ROOT="$CACTUS_CURL_ROOT"
if ! ruby "$SCRIPT_DIR/configure_xcode.rb"; then
    echo "Failed to configure Xcode project"
    exit 1
fi

echo ""
echo "Step 4: Building iOS test application..."

if [ "$device_type" = "simulator" ]; then
    ios_sim_sdk_path=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if [ -z "$ios_sim_sdk_path" ] || [ ! -d "$ios_sim_sdk_path" ]; then
        echo "iOS Simulator SDK not found"
        exit 1
    fi

    if ! xcodebuild -project "$xcodeproj_path" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS Simulator,id=$device_uuid" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$ios_sim_sdk_path" \
         PRODUCT_BUNDLE_IDENTIFIER="$bundle_id" \
         build; then
        echo "Build failed"
        exit 1
    fi

    app_path="$SCRIPT_DIR/build/Build/Products/Release-iphonesimulator/CactusTest.app"
else
    ios_sdk_path=$(xcrun --sdk iphoneos --show-sdk-path)
    if [ -z "$ios_sdk_path" ] || [ ! -d "$ios_sdk_path" ]; then
        echo "iOS SDK not found"
        exit 1
    fi

    if ! xcodebuild -project "$xcodeproj_path" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS,id=$device_uuid" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         -allowProvisioningUpdates \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$ios_sdk_path" \
         PRODUCT_BUNDLE_IDENTIFIER="$bundle_id" \
         CODE_SIGN_STYLE="Automatic" \
         build; then
        echo "Build failed"
        exit 1
    fi

    app_path="$SCRIPT_DIR/build/Build/Products/Release-iphoneos/CactusTest.app"
fi

model_dir=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
transcribe_model_dir=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
vad_model_dir=$(echo "$VAD_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
model_src="$PROJECT_ROOT/weights/$model_dir"
transcribe_model_src="$PROJECT_ROOT/weights/$transcribe_model_dir"
vad_model_src="$PROJECT_ROOT/weights/$vad_model_dir"
assets_src="$PROJECT_ROOT/tests/assets"

if [ ! -d "$model_src" ] || [ ! -f "$model_src/config.txt" ]; then
    echo "Error: model weights missing or invalid at $model_src"
    exit 1
fi
if [ ! -d "$transcribe_model_src" ] || [ ! -f "$transcribe_model_src/config.txt" ]; then
    echo "Error: transcribe model weights missing or invalid at $transcribe_model_src"
    exit 1
fi
if [ ! -d "$vad_model_src" ] || [ ! -f "$vad_model_src/config.txt" ]; then
    echo "Error: VAD model weights missing or invalid at $vad_model_src"
    exit 1
fi

echo "Copying model weights to app bundle..."
rm -rf "$app_path/$model_dir" "$app_path/$transcribe_model_dir" "$app_path/$vad_model_dir"
if ! cp -R "$model_src" "$app_path/"; then
    echo "Error: Could not copy model weights from $model_src"
    exit 1
fi
if ! cp -R "$transcribe_model_src" "$app_path/"; then
    echo "Error: Could not copy transcribe model weights from $transcribe_model_src"
    exit 1
fi
if ! cp -R "$vad_model_src" "$app_path/"; then
    echo "Error: Could not copy VAD model weights from $vad_model_src"
    exit 1
fi

# Whisper/Moonshine init expects a bundled "<transcribe_model>/vad" directory.
transcribe_lower=$(echo "$transcribe_model_dir" | tr '[:upper:]' '[:lower:]')
if [[ "$transcribe_lower" == *"whisper"* || "$transcribe_lower" == *"moonshine"* || "$transcribe_lower" == *"parakeet"* ]]; then
    if [ ! -f "$app_path/$transcribe_model_dir/vad/config.txt" ]; then
        echo "Transcribe model missing bundled VAD; injecting from $vad_model_src"
        mkdir -p "$app_path/$transcribe_model_dir/vad"
        if ! rsync -a "$vad_model_src/" "$app_path/$transcribe_model_dir/vad/"; then
            echo "Error: Failed to inject VAD into transcribe model bundle"
            exit 1
        fi
    fi
    if [ ! -f "$app_path/$transcribe_model_dir/vad/config.txt" ]; then
        echo "Error: bundled transcribe VAD is missing config.txt after packaging"
        exit 1
    fi
fi

echo "Copying test assets to app bundle..."
if ! cp -R "$assets_src" "$app_path/"; then
    echo "Error: Could not copy test assets from $assets_src"
    exit 1
fi

if [ "$RUN_ASR" = "1" ]; then
    if [ -z "$ASR_AUDIO_SOURCE" ] || [ ! -f "$ASR_AUDIO_SOURCE" ]; then
        echo "Error: CACTUS_ASR_AUDIO_SOURCE must point to an existing audio file when CACTUS_RUN_ASR=1"
        exit 1
    fi
    if [ -z "$ASR_AUDIO_FILE" ]; then
        ASR_AUDIO_FILE=$(basename "$ASR_AUDIO_SOURCE")
    fi
    echo "Copying ASR audio file to app bundle..."
    if ! cp "$ASR_AUDIO_SOURCE" "$app_path/$ASR_AUDIO_FILE"; then
        echo "Error: Could not copy ASR audio file into app bundle"
        exit 1
    fi
fi

echo ""
echo "Step 5: Running tests..."
echo "------------------------"

if [ "$device_type" = "simulator" ]; then
    echo "Installing on: $device_name"

    xcrun simctl boot "$device_uuid" || true

    if ! xcrun simctl install "$device_uuid" "$app_path"; then
        echo "Failed to install app on simulator"
        exit 1
    fi

    if [ "$RUN_ASR" = "1" ]; then
        echo "Launching ASR transcription..."
    else
        echo "Launching tests..."
    fi
    echo "Using model path: $model_dir"
    echo "Using transcribe model path: $transcribe_model_dir"
    echo "Using assets path: assets"
    echo "Using index path: assets"

    sim_env=(
        "SIMCTL_CHILD_CACTUS_TEST_MODEL=$model_dir"
        "SIMCTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL=$transcribe_model_dir"
        "SIMCTL_CHILD_CACTUS_TEST_VAD_MODEL=$vad_model_dir"
        "SIMCTL_CHILD_CACTUS_TEST_ASSETS=assets"
        "SIMCTL_CHILD_CACTUS_INDEX_PATH=assets"
        "SIMCTL_CHILD_CACTUS_NO_CLOUD_TELE=$CACTUS_NO_CLOUD_TELE"
    )
    if [ "$RUN_ASR" = "1" ]; then
        sim_env+=("SIMCTL_CHILD_CACTUS_RUN_ASR=1")
        sim_env+=("SIMCTL_CHILD_CACTUS_ASR_AUDIO_FILE=$ASR_AUDIO_FILE")
    fi
    if [ -n "$CACTUS_CLOUD_KEY" ]; then
        sim_env+=("SIMCTL_CHILD_CACTUS_CLOUD_KEY=$CACTUS_CLOUD_KEY")
    fi
    if [ -n "$CACTUS_CLOUD_STRICT_SSL" ]; then
        sim_env+=("SIMCTL_CHILD_CACTUS_CLOUD_STRICT_SSL=$CACTUS_CLOUD_STRICT_SSL")
    fi
    if [ -n "$CACTUS_CLOUD_HANDOFF_THRESHOLD" ]; then
        sim_env+=("SIMCTL_CHILD_CACTUS_CLOUD_HANDOFF_THRESHOLD=$CACTUS_CLOUD_HANDOFF_THRESHOLD")
    fi

    env "${sim_env[@]}" xcrun simctl launch --console-pty "$device_uuid" "$bundle_id"
else
    echo "Installing on: $device_name"

    if ! xcrun devicectl device install app --device "$device_uuid" "$app_path"; then
        echo "Failed to install app on device"
        echo "Common issues:"
        echo "  - Device not trusted"
        echo "  - Code signing failed"
        echo "  - Device locked or screen off"
        exit 1
    fi

    if [ "$RUN_ASR" = "1" ]; then
        echo "Launching ASR transcription..."
    else
        echo "Launching tests..."
    fi
    echo "(Logs will be fetched from device after completion)"
    echo "Using model path: $model_dir"
    echo "Using transcribe model path: $transcribe_model_dir"
    echo "Using assets path: assets"
    echo "Using index path: assets"

    device_env=(
        "DEVICECTL_CHILD_CACTUS_TEST_MODEL=$model_dir"
        "DEVICECTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL=$transcribe_model_dir"
        "DEVICECTL_CHILD_CACTUS_TEST_VAD_MODEL=$vad_model_dir"
        "DEVICECTL_CHILD_CACTUS_TEST_ASSETS=assets"
        "DEVICECTL_CHILD_CACTUS_INDEX_PATH=assets"
        "DEVICECTL_CHILD_CACTUS_NO_CLOUD_TELE=$CACTUS_NO_CLOUD_TELE"
    )
    if [ "$RUN_ASR" = "1" ]; then
        device_env+=("DEVICECTL_CHILD_CACTUS_RUN_ASR=1")
        device_env+=("DEVICECTL_CHILD_CACTUS_ASR_AUDIO_FILE=$ASR_AUDIO_FILE")
    fi
    if [ -n "$CACTUS_CLOUD_KEY" ]; then
        device_env+=("DEVICECTL_CHILD_CACTUS_CLOUD_KEY=$CACTUS_CLOUD_KEY")
    fi
    if [ -n "$CACTUS_CLOUD_STRICT_SSL" ]; then
        device_env+=("DEVICECTL_CHILD_CACTUS_CLOUD_STRICT_SSL=$CACTUS_CLOUD_STRICT_SSL")
    fi
    if [ -n "$CACTUS_CLOUD_HANDOFF_THRESHOLD" ]; then
        device_env+=("DEVICECTL_CHILD_CACTUS_CLOUD_HANDOFF_THRESHOLD=$CACTUS_CLOUD_HANDOFF_THRESHOLD")
    fi

    launch_output=$(env "${device_env[@]}" \
    xcrun devicectl device process launch --device "$device_uuid" "$bundle_id" 2>&1) || true

    echo "$launch_output"

    max_wait=300
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if xcrun devicectl device info processes --device "$device_uuid" | grep -q "CactusTest.app/CactusTest"; then
            sleep 2
            elapsed=$((elapsed + 2))
        else
            break
        fi
    done

    if [ $elapsed -ge $max_wait ]; then
        echo "Warning: Test execution timeout reached (${max_wait}s)"
    fi

    sleep 1

    echo "Fetching logs from device..."

    temp_log_dir=$(mktemp -d)
    temp_log_file="$temp_log_dir/cactus_test.log"

    if xcrun devicectl device copy from \
        --device "$device_uuid" \
        --source "Documents/cactus_test.log" \
        --destination "$temp_log_file" \
        --domain-type appDataContainer \
        --domain-identifier "$bundle_id"; then

        if [ -f "$temp_log_file" ]; then
            echo ""
            cat "$temp_log_file"
        else
            echo "Warning: Could not find downloaded log file"
        fi

        rm -rf "$temp_log_dir"
    else
        echo "Warning: Could not fetch log file from device"
    fi

    if echo "$launch_output" | grep -q "FBSOpenApplicationErrorDomain error 3"; then
        echo ""
        echo "App launch failed: Untrusted Developer"
        echo "To trust the developer profile on your device:"
        echo "  1. Open Settings > General > VPN & Device Management"
        echo "  2. Under Developer App, tap your Apple ID"
        echo "  3. Tap Trust and confirm"
        echo "Then run this script again."
        echo ""
    fi
fi
