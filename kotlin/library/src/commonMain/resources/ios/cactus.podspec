Pod::Spec.new do |spec|
  spec.name          = "cactus"
  spec.version       = "0.2.0"
  spec.summary       = "Cactus native framework for iOS"
  spec.description   = "Native C++ framework for running AI models locally on iOS"
  spec.homepage      = "https://github.com/cactus-compute/cactus"
  spec.license       = { :type => "MIT" }
  spec.author        = { "Cactus" => "contact@cactus.ai" }
  spec.ios.deployment_target = "13.0"
  spec.swift_version = "5.0"
  spec.vendored_frameworks = "cactus.xcframework", "libvosk.xcframework"
  spec.frameworks = "Accelerate", "Foundation", "Metal", "MetalKit"
  spec.source = { :path => "." }
  spec.preserve_paths = "cactus.xcframework", "libvosk.xcframework"
  spec.public_header_files = "cactus.xcframework/*/cactus.framework/Headers/*.h"
end 