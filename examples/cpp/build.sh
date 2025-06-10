mkdir -p build
cd build
cmake ..
make

ln -sf ../../../cactus/ggml-llama.metallib default.metallib

./test_vision_language