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

echo "Running Cactus tests on iOS..."
echo "============================"

if [ ! -d "/Applications/Xcode.app" ]; then
    echo "Xcode not installed"
    echo "Install Xcode from the Mac App Store"
    exit 1
fi

if ! xcode-select -p &>/dev/null; then
    echo "Xcode Command Line Tools not installed"
    echo "Install with: xcode-select --install"
    exit 1
fi

if ! /usr/bin/xcrun --version &>/dev/null; then
    echo "Xcode license not accepted"
    echo "Accept the license with: sudo xcodebuild -license accept"
    exit 1
fi

if ! command -v xcodebuild &>/dev/null; then
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

    matching_identity=$(security find-identity -p codesigning 2>/dev/null | grep "Apple Development" | head -1 || true)
    valid_identity=$(security find-identity -v -p codesigning 2>/dev/null | grep "Apple Development" | head -1 || true)

    if [ -z "$matching_identity" ]; then
        echo "No development certificates found"
        echo "To fix this:"
        echo "  1. Open Xcode > Settings > Accounts"
        echo "  2. Add your Apple ID"
        echo "  3. Download development certificates"
        exit 1
    fi

    if [ -z "$valid_identity" ]; then
        echo "Warning: Apple Development identity found, but macOS did not report it as a valid identity."
        echo "Proceeding and letting xcodebuild attempt automatic signing anyway."
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
tests_root="$PROJECT_ROOT/cactus-engine/tests"

project_file="$xcodeproj_path/project.pbxproj"
template_file="$xcodeproj_path/project.pbxproj.template"
echo "Copying project template..."
cp "$template_file" "$project_file"

bundle_id="com.cactus.test.${USER}"
echo "Using Bundle ID: $bundle_id"

if [ "$device_type" = "device" ]; then
    if [ -n "$DEVELOPMENT_TEAM" ]; then
        development_team="$DEVELOPMENT_TEAM"
    else
        teams=""
        while IFS= read -r line; do
            team_id=$(echo "$line" | grep -oE 'OU=[A-Z0-9]+' | cut -d= -f2)
            cn=$(echo "$line" | grep -oE 'CN=[^,]+' | cut -d= -f2)
            [ -n "$team_id" ] && teams=$(printf "%s\n%s|%s" "$teams" "$team_id" "$cn")
        done <<< "$(security find-identity -v -p codesigning 2>/dev/null | grep "Apple Development" | sed 's/.*"\(.*\)".*/\1/' | while read name; do security find-certificate -c "$name" -p 2>/dev/null | openssl x509 -noout -subject 2>/dev/null; done | sort -u)"
        teams=$(echo "$teams" | grep -v '^$')

        if [ -z "$teams" ]; then
            echo "No Apple Development certificates found"
            echo "Add one in Xcode > Settings > Accounts"
            exit 1
        fi

        team_count=$(echo "$teams" | wc -l | tr -d ' ')
        if [ "$team_count" -eq 1 ]; then
            development_team=$(echo "$teams" | head -1 | cut -d'|' -f1)
        else
            echo ""
            echo "Development Teams:"
            team_num=0
            while IFS='|' read -r tid tname; do
                team_num=$((team_num + 1))
                printf "  %2d. %s (%s)\n" "$team_num" "$tname" "$tid"
            done <<< "$teams"
            echo ""
            read -p "Select team (1-$team_num): " team_selection
            if ! [[ "$team_selection" =~ ^[0-9]+$ ]] || [ "$team_selection" -lt 1 ] || [ "$team_selection" -gt "$team_num" ]; then
                echo "Invalid selection"
                exit 1
            fi
            development_team=$(echo "$teams" | sed -n "${team_selection}p" | cut -d'|' -f1)
        fi
    fi
    if [ -z "$development_team" ]; then
        echo "Could not determine Team ID"
        exit 1
    fi
    echo "Using Team ID: $development_team"
fi

if ! command -v ruby &>/dev/null; then
    echo "Ruby not found"
    exit 1
fi

if ! gem list xcodeproj -i &>/dev/null; then
    echo "Installing xcodeproj gem..."
    if ! gem install xcodeproj; then
        echo "Failed to install xcodeproj gem"
        exit 1
    fi
fi

export PROJECT_ROOT TESTS_ROOT="$tests_root" XCODEPROJ_PATH="$xcodeproj_path" BUNDLE_ID="$bundle_id" DEVELOPMENT_TEAM="$development_team" DEVICE_TYPE="$device_type" CACTUS_CURL_ROOT="$CACTUS_CURL_ROOT"
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
         CODE_SIGN_ENTITLEMENTS="$SCRIPT_DIR/CactusTest/CactusTest/CactusTest.entitlements" \
         build; then
        echo "Build failed"
        exit 1
    fi

    app_path="$SCRIPT_DIR/build/Build/Products/Release-iphoneos/CactusTest.app"
fi

echo ""
echo "Step 5: Bundling model weights and assets..."

model_dir=$(basename "$CACTUS_TEST_MODEL")
transcription_dir=$(basename "$CACTUS_TEST_TRANSCRIPTION_MODEL")
assets_src="$PROJECT_ROOT/cactus-engine/tests/assets"

echo "Copying model weights to app bundle..."
rm -rf "$app_path/$model_dir"
if ! cp -R "$CACTUS_TEST_MODEL" "$app_path/"; then
    echo "Error: Could not copy model weights from $CACTUS_TEST_MODEL"
    exit 1
fi

echo "Copying transcription model to app bundle..."
rm -rf "$app_path/$transcription_dir"
if ! cp -R "$CACTUS_TEST_TRANSCRIPTION_MODEL" "$app_path/"; then
    echo "Error: Could not copy transcription model from $CACTUS_TEST_TRANSCRIPTION_MODEL"
    exit 1
fi

if [ -d "$assets_src" ]; then
    echo "Copying test assets to app bundle..."
    rm -rf "$app_path/assets"
    if ! cp -R "$assets_src" "$app_path/"; then
        echo "Error: Could not copy test assets from $assets_src"
        exit 1
    fi
fi

echo ""
echo "Step 6: Running tests..."
echo "------------------------"

if [ "$device_type" = "simulator" ]; then
    echo "Installing on: $device_name"

    xcrun simctl boot "$device_uuid" || true

    if ! xcrun simctl install "$device_uuid" "$app_path"; then
        echo "Failed to install app on simulator"
        exit 1
    fi

    echo "Launching tests..."
    echo "Using model path:               $model_dir"
    echo "Using transcription model path: $transcription_dir"
    echo "Using assets path:              assets"

    sim_env=(
        "SIMCTL_CHILD_CACTUS_TEST_MODEL=$model_dir"
        "SIMCTL_CHILD_CACTUS_TEST_TRANSCRIPTION_MODEL=$transcription_dir"
        "SIMCTL_CHILD_CACTUS_TEST_ASSETS=assets"
        "SIMCTL_CHILD_CACTUS_INDEX_PATH=assets"
        "SIMCTL_CHILD_CACTUS_NO_CLOUD_TELE=${CACTUS_NO_CLOUD_TELE:-1}"
        "SIMCTL_CHILD_CACTUS_TEST_ONLY=${CACTUS_TEST_SUITE:-}"
        "SIMCTL_CHILD_CACTUS_TEST_BACKEND=${CACTUS_TEST_BACKEND:-auto}"
    )

    env "${sim_env[@]}" xcrun simctl launch --console-pty --terminate-running-process "$device_uuid" "$bundle_id"

    data_container=$(xcrun simctl get_app_container "$device_uuid" "$bundle_id" data 2>/dev/null || true)
    exitcode_file="$data_container/Documents/cactus_test.exitcode"
    if [ -n "$data_container" ] && [ -f "$exitcode_file" ]; then
        SUITE_EXIT=$(tr -d '[:space:]' < "$exitcode_file")
        [[ "$SUITE_EXIT" =~ ^[0-9]+$ ]] || SUITE_EXIT=1
    else
        echo "Could not retrieve exit-code marker from simulator"
        SUITE_EXIT=1
    fi
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

    echo "Launching tests..."
    echo "(Logs will be fetched from device after completion)"
    echo "Using model path:               $model_dir"
    echo "Using transcription model path: $transcription_dir"
    echo "Using assets path:              assets"

    device_env=(
        "DEVICECTL_CHILD_CACTUS_TEST_MODEL=$model_dir"
        "DEVICECTL_CHILD_CACTUS_TEST_TRANSCRIPTION_MODEL=$transcription_dir"
        "DEVICECTL_CHILD_CACTUS_TEST_ASSETS=assets"
        "DEVICECTL_CHILD_CACTUS_INDEX_PATH=assets"
        "DEVICECTL_CHILD_CACTUS_NO_CLOUD_TELE=${CACTUS_NO_CLOUD_TELE:-1}"
        "DEVICECTL_CHILD_CACTUS_TEST_ONLY=${CACTUS_TEST_SUITE:-}"
        "DEVICECTL_CHILD_CACTUS_TEST_BACKEND=${CACTUS_TEST_BACKEND:-auto}"
    )

    env "${device_env[@]}" \
        xcrun devicectl device process launch --device "$device_uuid" "$bundle_id" 2>&1
    SUITE_EXIT=$?

    echo "Waiting for tests to complete..."
    max_wait=300
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if xcrun devicectl device info processes --device "$device_uuid" 2>/dev/null | grep -q "CactusTest.app/CactusTest"; then
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

    echo ""
    echo "Fetching logs from device..."
    log_dir=$(mktemp -d)
    xcrun devicectl device copy from --device "$device_uuid" \
        --domain-identifier "$bundle_id" \
        --domain-type appDataContainer \
        --source Documents/cactus_test.log \
        --destination "$log_dir/cactus_test.log" 2>/dev/null || true
    xcrun devicectl device copy from --device "$device_uuid" \
        --domain-identifier "$bundle_id" \
        --domain-type appDataContainer \
        --source Documents/cactus_test.exitcode \
        --destination "$log_dir/exitcode" 2>/dev/null || true

    if [ -f "$log_dir/cactus_test.log" ]; then
        echo "=== Device Test Output ==="
        cat "$log_dir/cactus_test.log"
        echo "=== End Device Test Output ==="
    else
        echo "Could not retrieve test logs from device"
    fi

    if [ -f "$log_dir/exitcode" ]; then
        SUITE_EXIT=$(tr -d '[:space:]' < "$log_dir/exitcode")
        [[ "$SUITE_EXIT" =~ ^[0-9]+$ ]] || SUITE_EXIT=1
    else
        echo "Could not retrieve exit-code marker from device"
        SUITE_EXIT=1
    fi
    rm -rf "$log_dir"
fi

echo ""
if [ "${SUITE_EXIT:-1}" -eq 0 ]; then
    echo "All tests passed."
else
    echo "Some tests failed."
fi
exit "${SUITE_EXIT:-1}"
