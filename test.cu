#include <chrono>
#include <cstdint>
#include <iostream>

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

constexpr std::uint32_t n_in = 128;
constexpr std::uint32_t n_out = 40*n_in;
constexpr std::uint32_t batch_sz = 4096;
constexpr std::uint32_t n_epochs = 100;
static_assert(n_out % 40 == 0);
static_assert(n_in == n_out / 40);

static __device__ __forceinline__ float _activation_stub(float x) { return x; }

/*
// TODO(petr): This function isn't optimized yet.
template <float(*ACTIVATION)(float) = _activation_stub, float(*ACTIVATION_DELTA)(float) = _activation_stub>
static __global__ void backward_mse(
	const float *__restrict__ out,
	const float *__restrict__ expected,
	float *__restrict__ _delta,
	std::uint32_t batch_sz)
{
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		// TODO(petr): Try to divide by the batch_sz.
		float delta = 2.0f * (ACTIVATION(out[CUDA_THREAD_INDEX]) - expected[CUDA_THREAD_INDEX]);
		delta *= ACTIVATION_DELTA(out[CUDA_THREAD_INDEX]);

		_delta[CUDA_THREAD_INDEX] = delta;
		out += CUDA_THREAD_COUNT;
		_delta += CUDA_THREAD_COUNT;
	}
}
*/

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float) = _activation_stub>
static __global__ void forward_fixed_layer(
	const float *__restrict__ biases,
	const float *__restrict__ _weights,
	const float *__restrict__ in,
	float *__restrict__ out,
	const std::uint32_t batch_sz)
{
	__shared__ float shrd_in[N_IN];
	const float bias = biases[CUDA_THREAD_INDEX];
	const float *weights = _weights + CUDA_THREAD_INDEX * N_IN;
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

/*
template <std::uint32_t N_IN, float(*ACTIVATION_DELTA)(float) = _activation_stub>
static __global__ void backward_fixed_layer(
	const float *__restrict__ out_delta,
	float *__restrict__ delta,
	const std::uint32_t batch_sz)
{
}
*/

/*
static __global__ void update_fixed_layer()
{
	// TODO(petr): I have to either update the weights and biases after this or I have to compute the delta for the previous layer before updating.
	// TODO(petr): I will have to make a separate kernel for this.

	__shared__ float shrd_in[N_IN];
	float bias = biases[CUDA_THREAD_INDEX];
	float *weights = _weights + CUDA_THREAD_INDEX * N_IN;
	for (std::uint32_t i = 0; i < batch_sz; i++)
	{
		// TODO(petr): Delta could probably be calculated in the previous step

		const float delta = _delta[CUDA_THREAD_INDEX] * learning_rate;
		bias -= delta;

		__syncthreads();
		shrd_in[threadIdx.x] = IN_ACTIVATION(in[threadIdx.x]);
		__syncthreads();

		#pragma unroll
		for (std::uint32_t j = 0; j < N_IN; j++)
		{
			weights[j] -= shrd_in[j] * delta;
		}

		// TODO(petr): Be aware that I can't use in and _delta after this
		in += N_IN;
		_delta += CUDA_THREAD_COUNT;
	}
	biases[CUDA_THREAD_INDEX] = bias;
}
*/

// TODO(petr): Paralelize batch if the layer is too small.
// TODO(petr): Try to store the bias as an additional weight for potential better performance.
// TODO(petr): Compute the entire layer, not just a fixed one.
// TODO(petr): Some activation functions need the entire weighted output - Make them a separate layer.
// TODO(petr): Utilize tensor cores for the machine learning.

int main(void)
{
	float *biases;
	float *weights;
	float *in;
	float *out;
	float *delta;
	cudaMalloc(&biases, n_out * sizeof(*biases));
	cudaMalloc(&weights, n_out * n_in * sizeof(*weights));
	cudaMalloc(&in, n_in * batch_sz * sizeof(*in));
	cudaMalloc(&out, n_out * batch_sz * sizeof(*out));
	cudaMalloc(&delta, n_out * batch_sz * sizeof(*delta));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < n_epochs; epoch++)
	{
		forward_fixed_layer<n_in><<<40, n_out / 40>>>(biases, weights, in, out, batch_sz);
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
