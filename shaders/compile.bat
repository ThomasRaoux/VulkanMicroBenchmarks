@echo on
glslangValidator.exe -V copy_vec4.comp -o copy_vec4.spv
glslangValidator.exe -V copy_scalar_4.comp -o copy_scalar_4.spv
glslangValidator.exe -V copy_scalar_1.comp -o copy_scalar_1.spv
glslangValidator.exe -V copy_vec2.comp -o copy_vec2.spv
glslangValidator.exe -V copy_vec4_2.comp -o copy_vec4_2.spv
glslangValidator.exe -V matmul_scalar.comp -o matmul_scalar.spv
glslangValidator.exe -V matmul_scalar_tiled.comp -o matmul_scalar_tiled.spv
glslangValidator.exe -V matmul_vector.comp -o matmul_vector.spv --target-env spirv1.3
glslangValidator.exe -V matmul_vector_2.comp -o matmul_vector_2.spv --target-env spirv1.3
glslangValidator.exe -V matmul_vector_3.comp -o matmul_vector_3.spv --target-env spirv1.3
glslangValidator.exe -V matmul_vector_4.comp -o matmul_vector_4.spv --target-env spirv1.3