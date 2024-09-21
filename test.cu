#include <chrono>
#include <cstdint>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#define CUDA_THREAD_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define CUDA_THREAD_COUNT (gridDim.x * blockDim.x)

#define ASSERT_CUDA_ERROR() \
	do { \
		cudaDeviceSynchronize(); \
		if (cudaPeekAtLastError() != cudaSuccess) \
		{ \
			std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
			std::exit(EXIT_FAILURE); \
		} \
	} while (0)

constexpr std::uint32_t n_in = 128;
constexpr std::uint32_t n_out = 40*n_in;
constexpr std::uint32_t batch_sz = 4096;
constexpr std::uint32_t n_epochs = 100;
static_assert(n_out % 40 == 0);
static_assert(n_in == n_out / 40);

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN>
static __global__ void forward_fixed_layer(
	const float *__restrict__ _weights,
	const float *__restrict__ in,
	float *__restrict__ out,
	const std::uint32_t batch_sz)
{
	__shared__ float shrd_in[N_IN];
	const float *weights = _weights + CUDA_THREAD_INDEX * N_IN;
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		float result = out[CUDA_THREAD_INDEX];

		__syncthreads();
		shrd_in[threadIdx.x] = in[threadIdx.x];
		__syncthreads();

		#pragma unroll
		for (std::uint32_t j = 0; j < N_IN; j++)
		{
			result += weights[j] * shrd_in[j];
		}

		// TODO(petr): Remove the hardcoding of this activation function.
		result = tanhf(result);
		out[CUDA_THREAD_INDEX] = result;
		in += N_IN;
		out += CUDA_THREAD_COUNT;
	}
}

int main(void)
{
	float *biases;
	float *weights;
	float *in;
	float *out;
	cudaMalloc(&biases, n_out * sizeof(*biases));
	cudaMalloc(&weights, n_out * n_in * sizeof(*weights));
	cudaMalloc(&in, n_in * batch_sz * sizeof(*in));
	cudaMalloc(&out, n_out * batch_sz * sizeof(*out));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < n_epochs; epoch++)
	{
		// TODO(petr): This isn't correct!
		cudaMemcpyAsync(out, biases, n_out * sizeof(*biases), cudaMemcpyDeviceToDevice);
		forward_fixed_layer<n_in><<<40, n_out / 40>>>(weights, in, out, batch_sz);
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
