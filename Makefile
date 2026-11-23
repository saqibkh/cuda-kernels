# CPU emulator build needs only g++ (C++20). GPU build needs the CUDA toolkit.
CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= sm_75            # T4 (Colab). A100: sm_80, L4: sm_89, H100: sm_90
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -pthread -Iinclude -Ikernels
NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -Iinclude -Ikernels -lineinfo

.PHONY: test-cpu gpu test-gpu bench clean
test-cpu: build/test_cpu          ## correctness on the CPU emulator (no GPU needed)
	./build/test_cpu
build/test_cpu: tests/test_kernels.cu kernels/*.cuh include/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -x c++ $< -o $@

gpu: build/test_gpu build/bench   ## compile for a real GPU
build/test_gpu: tests/test_kernels.cu kernels/*.cuh include/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $< -o $@
build/bench: bench/bench.cu kernels/*.cuh include/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas
test-gpu: build/test_gpu
	./build/test_gpu
bench: build/bench
	./build/bench
clean:
	rm -rf build
