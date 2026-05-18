# Cactus Battery Test App

Minimal SwiftUI app: two buttons (GPU / NPU). Tap one to run cactus prefill repeatedly on that backend until you tap again. Shows iteration count, last prefill_tps, battery level, and battery drop over elapsed time.

## Files

- `CactusBatteryTestApp.swift` — single-file SwiftUI app (App + ContentView + Runner)
- `CactusBridge.h` — Objective-C bridging header that exposes the cactus C API to Swift

## Setup in Xcode (one-time)

1. **Create a new Xcode project**
   - File → New → Project → iOS → App
   - Interface: SwiftUI, Language: Swift
   - Name it whatever, e.g. `CactusBatteryTest`
   - Save it inside this directory or wherever convenient

2. **Drop in the source files**
   - Delete the auto-generated `ContentView.swift` and `<Name>App.swift`
   - Drag `CactusBatteryTestApp.swift` and `CactusBridge.h` into the project (check "Copy items if needed")

3. **Set the bridging header**
   - Project → Build Settings → "Objective-C Bridging Header"
   - Set to `$(SRCROOT)/CactusBridge.h` (or whatever path)

4. **Link `libcactus-device.a`**
   - Build Phases → Link Binary With Libraries → +
   - Add Other → Add Files → pick `apple/libcactus-device.a` from this repo
   - Also add system frameworks: `Metal`, `MetalPerformanceShaders`, `Accelerate`, `CoreML`, `Foundation`, `Security`, `SystemConfiguration`, `CFNetwork`
   - Add `libc++.tbd` for C++ stdlib

5. **Provide a model**
   The app looks for the model in this order:
   - `Bundle.main.path(forResource: "model", ofType: nil)` — if you add a folder called `model` to the app bundle (drag the model directory into Xcode, choose "Create folder references" so it stays a folder), this is used
   - `ProcessInfo.processInfo.environment["CACTUS_MODEL"]` — set in Xcode scheme → Run → Arguments → Environment Variables
   - `~/Documents/model` on the device — if you push via Files app

   The model directory is the cactus-format weights folder (e.g. `weights/gemma-4-e2b-it/`).

6. **Build target**
   - Set scheme to your physical iPhone
   - Build and run (⌘R)

## How to run a battery test

1. Charge phone to 100%, unplug
2. Disable background app refresh, set airplane mode, lock screen brightness at 50%
3. Open the app
4. Tap **GPU** — runs prefill on MPS continuously
5. Wait 15-30 minutes, note final battery level on the in-app display (`Δ` row)
6. Tap **STOP GPU** to stop
7. Reset (charge back, etc.) and repeat with **NPU**

The on-screen `Δ` shows e.g. `2.3% drop over 945s` — that's your drain rate.

## Notes

- `cactus_prefill` is called with `max_tokens=1` and a fixed long prompt (~250 tokens). Each iteration is a full prefill of the prompt; decode is suppressed.
- The `Runner` calls `cactus_npu_set_enabled` / `cactus_mps_set_enabled` before each run to ensure the right backend handles the work.
- Both buttons are mutually exclusive — one disables while the other is running.
- For more accurate per-op energy (joules/token), use Xcode → Open Developer Tool → Instruments → Energy Log instead of (or alongside) the in-app battery readout.

## Caveats

- The v2 model loader currently throws "Scalar operations only support FP16 precision" during init for current Gemma 4 E2B weights. Until that's fixed in the v2 main branch, `cactus_init` will fail. The app will display "failed to load model" and you can't run the test. This is a v2 bug, not the GPU code; see project notes.
- `apple/libcactus-device.a` must be the MPS-enabled build. If unsure, run `nm libcactus-device.a | grep cactus_mps_set_enabled` — it should show a `T` symbol.
