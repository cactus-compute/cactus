add_executable(test_rms_norm test_rms_norm.mm)
target_link_libraries(test_rms_norm PRIVATE cactus-kernels-gpu)
target_compile_options(test_rms_norm PRIVATE -fobjc-arc)

add_executable(test_matmul_int4 test_matmul_int4.mm)
target_link_libraries(test_matmul_int4 PRIVATE cactus-kernels-gpu)
target_compile_options(test_matmul_int4 PRIVATE -fobjc-arc)

add_executable(test_matmul_int4_simple test_matmul_int4_simple.mm)
target_link_libraries(test_matmul_int4_simple PRIVATE cactus-kernels-gpu)
target_compile_options(test_matmul_int4_simple PRIVATE -fobjc-arc)
