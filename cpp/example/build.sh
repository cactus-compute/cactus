mkdir -p build
cd build
cmake ..

echo "Building the project..."
make

ln -sf ../../../cpp/ggml-llama.metallib default.metallib

./cactus_llm chat