#include <cassert>
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

// TODO(petr): It would probably make sense to calculate the loss inside this function.
// n_out * batch_sz needs to fit inside std::uint32_t.
template <float(*IN_ACTIVATION)(float) = _activation_stub>
static __global__ void backward_mse(
	const float *__restrict__ out,
	const float *__restrict__ expected,
	float *__restrict__ gradients,
	const std::uint32_t n_out,
	const std::uint32_t batch_sz)
{
	for (std::uint32_t i = CUDA_THREAD_INDEX; i < n_out * batch_sz; i += CUDA_THREAD_COUNT)
	{
		gradients[i] = 2.0f * (IN_ACTIVATION(out[i]) - expected[i]) / batch_sz;
	}
}

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float)>
static __global__ void forward_fixed_layer(
	const float *__restrict__ biases,
	const float *__restrict__ weights,
	const float *__restrict__ in,
	const std::uint32_t in_stride,
	float *__restrict__ out,
	const std::uint32_t n_out,
	const std::uint32_t out_stride,
	const std::uint32_t batch_sz,
	const bool use_biases)
{
	if (CUDA_THREAD_INDEX < n_out)
	{
		__shared__ float shrd_in[N_IN];
		weights += CUDA_THREAD_INDEX * in_stride;
		for (std::uint32_t i = 0; i < batch_sz; i++)
		{
			float result = use_biases
				? biases[CUDA_THREAD_INDEX]
				: out[CUDA_THREAD_INDEX];

			__syncthreads();
			shrd_in[threadIdx.x] = IN_ACTIVATION(in[threadIdx.x]);
			__syncthreads();

			#pragma unroll
			for (std::uint32_t j = 0; j < N_IN; j++)
			{
				result += weights[j] * shrd_in[j];
			}

			out[CUDA_THREAD_INDEX] = result;
			in += in_stride;
			out += out_stride;
		}
	}
}

//
// Each input and the weights of every neuron need to be aligned to N_IN floats.
//
template <float(*IN_ACTIVATION)(float) = _activation_stub>
static void forward_layer(
	const float *__restrict__ biases,
	const float *__restrict__ weights,
	const float *__restrict__ in,
	const std::uint32_t n_in,
	float *__restrict__ out,
	const std::uint32_t n_out,
	const std::uint32_t batch_sz)
{
	constexpr std::uint32_t N_IN = 128;
	assert(n_in % N_IN == 0);
	for (std::uint32_t i = 0; i < n_out; i += 40*N_IN)
	{
		for (std::uint32_t j = 0; j < n_in; j += N_IN)
		{
			forward_fixed_layer<N_IN, IN_ACTIVATION><<<40, N_IN>>>(
				&biases[i],
				&weights[i * n_in + j],
				&in[i * n_in + j],
				(n_in + N_IN - 1) / N_IN * N_IN,
				&out[i],
				n_out - i,
				n_out,
				batch_sz,
				!j);
		}
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

// TODO(petr): Utilize tensor cores for the machine learning.
// TODO(petr): Try to compute the entire MLP within a single kernel.

int main(void)
{
	std::uint32_t n_in = 128;
	assert(n_in % 128 == 0);
	std::uint32_t n_out = 40*n_in;
	std::uint32_t batch_sz = 4096;
	std::uint32_t n_epochs = 100;

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
		forward_layer<>(biases, weights, in, n_in, out, n_out, batch_sz);
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
