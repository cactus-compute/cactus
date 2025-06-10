mkdir -p build
cd build
cmake ..
make

ln -sf ../../../cactus/ggml-llama.metallib default.metallib

./cactus_llm