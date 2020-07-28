# Instantiate shaders with multiple different tile size. 
# We cannot always rely on driver to fold specialized constant and unroll loops.
# Therefore we use pre-processor to set the tile size and use spirv-opt to unroll loops.
tiles=( 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 )

for i in "${tiles[@]}"
do
for j in "${tiles[@]}"
do
glslangValidator -V matmul_scalar_tiled.comp -DTILE_M=$i -DTILE_N=$j -o matmul_scalar_tiled$i-$j.spv
spirv-opt matmul_scalar_tiled$i-$j.spv -O -o matmul_scalar_tiled$i-$j.spv
done
done

tiles2=( 64 96 128 192 )
tiles4=( 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 72 80 88 96 104 112 120 128 )


for i in "${tiles4[@]}"
do
for j in "${tiles2[@]}"
do
glslangValidator -V matmul_4x8.comp -DTILE_M=$i -DTILE_N=$j -o matmul_4x8$i-$j.spv
spirv-opt matmul_4x8$i-$j.spv -O -o matmul_4x8$i-$j.spv
done
done

sizes=( 1 2 4 8 16 32 64 128 256 )
sizes2=( 1 2 4 8 16 32 64 128 )

for i in "${sizes[@]}"
do
for j in "${sizes2[@]}"
do
glslangValidator -V matmul_4x8block.comp -DX=$i -DY=$j -o matmul_4x8block$i-$j.spv
spirv-opt matmul_4x8block$i-$j.spv -O -o matmul_4x8block$i-$j.spv
done
done

tiles2=( 64 96 128 160 192 )

for i in "${tiles4[@]}"
do
for j in "${tiles2[@]}"
do
glslangValidator -V matmul_4x8x4.comp -DTILE_M=$i -DTILE_N=$j -o matmul_4x8x4$i-$j.spv
spirv-opt matmul_4x8x4$i-$j.spv -O -o matmul_4x8x4$i-$j.spv
done
done

tiles=( 16 32 48 64 80 96 112 128 )

for i in "${tiles[@]}"
do
for j in "${tiles2[@]}"
do
glslangValidator -V matmul_4x8x4_slm.comp -DTILE_M=$i -DTILE_N=$j -o matmul_4x8x4_slm$i-$j.spv --target-env spirv1.3
spirv-opt matmul_4x8x4_slm$i-$j.spv -O -o matmul_4x8x4_slm$i-$j.spv
done
done

tiles=( 8 16 32 64 128 )

for i in "${tiles[@]}"
do
for j in "${tiles[@]}"
do
glslangValidator -V matmul_vector.comp -DTILE_M=$i -DTILE_N=$j -o matmul_vector$i-$j.spv --target-env spirv1.3
spirv-opt matmul_vector$i-$j.spv -O -o matmul_vector$i-$j.spv
done
done