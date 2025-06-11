#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LEFTHOOK=0 

[ -d node_modules ] && rm -rf node_modules
[ -d lib ] && rm -rf lib 

cd "$ROOT_DIR/cactus-react" 

echo "Copying iOS frameworks to React Native project..."
rm -rf ios/cactus.xcframework
cp -R "$ROOT_DIR/cactus-ios"/cactus.xcframework ios/
cp -R "$ROOT_DIR/cactus-ios"/CMakeLists.txt ios/

echo "Building React Native package..." 
yarn 
yarn build 

echo "React Native package built successfully!" 