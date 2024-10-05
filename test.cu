#include <chrono>
#include <cstdint>
#include <iostream>

#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;

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

static __device__ __forceinline__ float _activation_stub(float x) { return x; }

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float) = _activation_stub>
static __global__ void forward_fixed_layer(
	const float *__restrict__ biases,
	const float *__restrict__ weights,
	const float *__restrict__ in,
	float *__restrict__ out,
	const std::uint32_t batch_sz)
{
	__shared__ float shrd_in[N_IN];
	const float bias = biases[CUDA_THREAD_INDEX];
	weights += CUDA_THREAD_INDEX * N_IN;
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		float result = bias;

		__syncthreads();
		shrd_in[threadIdx.x] = IN_ACTIVATION(in[threadIdx.x]);
		__syncthreads();

		#pragma unroll
		for (std::uint32_t j = 0; j < N_IN; j++)
		{
			result += weights[j] * shrd_in[j];
		}

		out[CUDA_THREAD_INDEX] = result;
		in += N_IN;
		out += CUDA_THREAD_COUNT;
	}
}

//
// N_OUT represents the size of the current layer, while the number of threads
// corresponds to the size of the previous layer.
// Ensure that N_OUT * 4 is less than the available shared memory capacity.
// N_OUT has to be equal to the number of threads per block.
//
template <std::uint32_t N_OUT, float(*IN_ACTIVATION_GRADIENT)(float) = _activation_stub>
static __global__ void backward_fixed_layer(
	const float *__restrict__ weights,
	const float *__restrict__ gradients,
	float *__restrict__ in_gradients,
	const float *__restrict__ in,
	const std::uint32_t batch_sz)
{
	weights += CUDA_THREAD_INDEX;
	__shared__ float shrd_gradients[N_OUT];
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		float in_gradient = 0.0f;

		__syncthreads();
		shrd_gradients[threadIdx.x] = gradients[threadIdx.x];
		__syncthreads();

		#pragma unroll
		for (std::uint32_t j = 0; j < N_OUT; j++)
		{
			in_gradient += weights[j * CUDA_THREAD_COUNT] * shrd_gradients[j];
		}

		in_gradient *= IN_ACTIVATION_GRADIENT(in[CUDA_THREAD_INDEX]);
		in_gradients[CUDA_THREAD_INDEX] = in_gradient;

		in += CUDA_THREAD_COUNT;
		in_gradients += CUDA_THREAD_COUNT;
		gradients += N_OUT;
	}
}

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float) = _activation_stub>
static __global__ void update_fixed_layer(
	float *__restrict__ biases,
	float *__restrict__ weights,
	float *__restrict__ gradients,
	const float *__restrict__ in,
	const float learning_rate,
	const std::uint32_t batch_sz)
{
	float bias = biases[CUDA_THREAD_INDEX];
	weights += CUDA_THREAD_INDEX * N_IN;
	__shared__ float shrd_in[N_IN];
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		// TODO(petr): I should probably average the gradients, possibly by dividing them by batch_sz.
		// This could also be achieved by doing the same in the cost calculation.
		float gradient = gradients[CUDA_THREAD_INDEX] * learning_rate;

		__syncthreads();
		shrd_in[threadIdx.x] = IN_ACTIVATION(in[threadIdx.x]);
		__syncthreads();

		#pragma unroll
		for (std::uint32_t j = 0; j < N_IN; j++)
		{
			weights[j] -= shrd_in[j] * gradient;
		}

		bias -= gradient;
		in += N_IN;
		gradients += CUDA_THREAD_COUNT;
	}
	biases[CUDA_THREAD_INDEX] = bias;
}

// TODO(petr): Compute the entire layer, not just a fixed one.
// TODO(petr): Paralelize batch if the layer is too small.
// TODO(petr): Utilize tensor cores for the machine learning.

int main(void)
{
	constexpr std::uint32_t n_in = 128;
	constexpr std::uint32_t n_out = 40*n_in;
	constexpr std::uint32_t batch_sz = 4096;
	constexpr std::uint32_t n_epochs = 100;
	static_assert(n_out % 40 == 0);
	static_assert(n_in == n_out / 40);

	float *biases;
	float *weights;
	float *in;
	float *out;
	float *gradients;
	cudaMalloc(&biases, n_out * sizeof(*biases));
	cudaMalloc(&weights, n_out * n_in * sizeof(*weights));
	cudaMalloc(&in, n_in * batch_sz * sizeof(*in));
	cudaMalloc(&out, n_out * batch_sz * sizeof(*out));
	cudaMalloc(&gradients, n_out * batch_sz * sizeof(*gradients));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < n_epochs; epoch++)
	{
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
