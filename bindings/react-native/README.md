# React Native Bindings

Native bridge modules over the cactus C API for iOS and Android.

## Integration

```bash
cactus build --apple
cactus build --android
```

1. Add the bridge files from this folder to your React Native app
2. Add the raw bindings they depend on: [`bindings/kotlin/`](/bindings/kotlin/) (Android), [`bindings/swift/`](/bindings/swift/) (Apple)
3. Register the Android package in `MainApplication.kt`:
   ```kotlin
   override fun getPackages() = PackageList(this).packages.apply {
       add(com.cactus.reactnative.CactusPackage())
   }
   ```

## Usage

```ts
import Cactus from './index';

const handle = await Cactus.init('/path/to/model', null, false);
const result = await Cactus.complete(handle, messagesJson, null, null, null);
await Cactus.destroy(handle);
```
