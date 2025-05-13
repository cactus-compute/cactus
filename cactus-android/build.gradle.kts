plugins {
    id("com.android.library") version "8.4.1"
    id("org.jetbrains.kotlin.android") version "1.9.23"
    id("maven-publish")
    signing
}


android {
    namespace = "com.cactus.android"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += listOf(
                "arm64-v8a",
                "x86_64"
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
        }
        debug {
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/CMakeLists.txt") 
            version = "3.22.1"
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
        singleVariant("debug") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

dependencies {
}

publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "io.github.cactus-compute"
            artifactId = "cactus-android"
            version = "0.0.7" 

            afterEvaluate {
                from(components["release"]) 
            }

            pom {
                name.set("Cactus Android Library") 
                description.set(file("README.md").readText())
                url.set("https://github.com/cactus-compute/cactus") 

                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://github.com/cactus-compute/cactus/blob/main/LICENSE")
                    }
                }
                developers {
                    developer {
                        id.set("cactus-compute")
                        name.set("Cactus Compute, Inc.")
                        email.set("founders@cactuscompute.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/cactus-compute/cactus.git")
                    developerConnection.set("scm:git:ssh://github.com/cactus-compute/cactus.git")
                    url.set("https://github.com/cactus-compute/cactus/tree/main/android")
                }
            }
        }
    }
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/cactus-compute/cactus") 
            credentials {
                username = project.findProperty("gpr.user")?.toString()
                password = project.findProperty("gpr.key")?.toString()
            }
        }
        maven {
            name = "OSSRH"
            url = uri(
                if (version.toString().endsWith("SNAPSHOT")) {
                    "https://s01.oss.sonatype.org/content/repositories/snapshots/"
                } else {
                    "https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/"
                }
            )
            credentials {
                username = project.findProperty("ossrhUsername")?.toString()
                password = project.findProperty("ossrhPassword")?.toString()
            }
        }
    }
}

signing {
    useInMemoryPgpKeys(
        project.findProperty("signing.keyId")?.toString(),
        project.findProperty("signing.pgpKey")?.toString(),
        project.findProperty("signing.password")?.toString()
    )
    sign(publishing.publications["release"])
}