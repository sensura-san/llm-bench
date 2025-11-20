cd llama.cpp
cmake -S . -B build -DDGML_BLAS=ON -DDGML_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF
cmake --build build -j $(nproc)
