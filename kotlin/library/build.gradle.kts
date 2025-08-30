import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.kotlinSerialization)
    id("co.touchlab.kmmbridge") version "1.0.1"
    `maven-publish`
}

group = "com.cactus"
version = "0.2.4"

kotlin {
    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
    ).forEach { iosTarget ->
        iosTarget.binaries.framework {
            baseName = "cactus"
            isStatic = true
        }
        
        iosTarget.compilations.getByName("main") {
            cinterops {
                val cactus by creating {
                    defFile(project.file("src/iosMain/cinterop/cactus.def"))
                    packageName("com.cactus.native")
                    
                    val archPath = when (iosTarget.name) {
                        "iosArm64" -> "ios-arm64"
                        "iosX64" -> "ios-arm64_x86_64-simulator"
                        "iosSimulatorArm64" -> "ios-arm64_x86_64-simulator"
                        else -> "ios-arm64"
                    }
                    
                    val frameworkPath = project.file("src/commonMain/resources/ios/cactus.xcframework/$archPath")
                    includeDirs(project.file("$frameworkPath/cactus.framework/Headers"))
                    
                    compilerOpts("-framework", "Accelerate", "-framework", "Foundation",
                               "-framework", "Metal", "-framework", "MetalKit",
                               "-framework", "cactus", "-F", frameworkPath.absolutePath)
                }
                
                val vosk by creating {
                    defFile(project.file("src/nativeInterop/cinterop/vosk.def"))
                    packageName("com.vosk.native")
                    
                    val voskArchPath = when (iosTarget.name) {
                        "iosArm64" -> "ios-arm64_armv7_armv7s"
                        "iosX64" -> "ios-arm64_x86_64-simulator"
                        "iosSimulatorArm64" -> "ios-arm64_x86_64-simulator"
                        else -> "ios-arm64_armv7_armv7s"
                    }
                    
                    val voskLibPath = project.file("src/commonMain/resources/ios/libvosk.xcframework/$voskArchPath")
                    val voskHeaderPath = project.file("src/nativeInterop/cinterop")
                    
                    includeDirs(voskHeaderPath)
                    compilerOpts("-I", voskHeaderPath.absolutePath)
                    
                    // Link the static library
                    extraOpts("-libraryPath", voskLibPath.absolutePath)
                }
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.1")
            }
        }
        val androidMain by getting {
            dependencies {
                // Use the AAR artifact so that JNA's native .so files are bundled in the APK
                implementation("net.java.dev.jna:jna:5.13.0@aar")
                implementation("androidx.core:core-ktx:1.12.0")
                implementation("androidx.activity:activity-compose:1.8.2")
                // VOSK offline speech recognition
                implementation("com.alphacephei:vosk-android:0.3.47@aar")
            }
        }
    }
}

tasks.register("embedNativeLibrary") {
    dependsOn("assembleCactusReleaseXCFramework")
    doLast {
        val xcframeworkDir = file("build/XCFrameworks/release/cactus.xcframework")
        
        listOf(
            "ios-arm64",
            "ios-arm64_x86_64-simulator"
        ).forEach { archPath ->
            val sourceFramework = file("src/commonMain/resources/ios/cactus.xcframework/$archPath/cactus.framework/cactus")
            val targetFramework = file("$xcframeworkDir/$archPath/cactus.framework/cactus")
            
            if (sourceFramework.exists() && targetFramework.exists()) {
                copy {
                    from(sourceFramework)
                    into(targetFramework.parent)
                    rename { targetFramework.name }
                }
                
                val sourceFrameworkDir = file("src/commonMain/resources/ios/cactus.xcframework/$archPath/cactus.framework")
                val targetFrameworkDir = file("$xcframeworkDir/$archPath/cactus.framework")
                
                copy {
                    from(sourceFrameworkDir) {
                        include("*.metallib")
                    }
                    into(targetFrameworkDir)
                }
                
                println("Embedded native library and Metal shaders for $archPath")
            }
            
            // Copy Vosk model files to the framework
            val targetFrameworkDir = file("$xcframeworkDir/$archPath/cactus.framework")
            copy {
                from("src/commonMain/resources/ios") {
                    include("vosk-model-small-en-us-0.15/**")
                    include("vosk-model-spk-0.4/**")
                    include("10001-90210-01803.wav")
                }
                into(targetFrameworkDir)
            }
            
            println("Embedded Vosk models for $archPath")
        }
    }
}

tasks.named("publishToMavenLocal") {
    dependsOn("embedNativeLibrary")
}

kmmbridge {
    mavenPublishArtifacts()
    spm()
}

android {
    namespace = "com.cactus.library"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/commonMain/resources/android/jniLibs")
        }
    }
}

publishing {
    publications {
        withType<MavenPublication> {
            pom {
                name.set("Cactus")
                description.set("Run AI locally in your apps")
                url.set("https://github.com/cactus-compute/cactus/")
            }
        }
    }
}
