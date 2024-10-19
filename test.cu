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
	const std::uint32_t in_stride,
	const std::uint32_t n_out,
	const std::uint32_t out_stride,
	const float learning_rate,
	const std::uint32_t batch_sz)
{
	if (CUDA_THREAD_INDEX < n_out)
	{
		float bias = biases[CUDA_THREAD_INDEX];
		weights += CUDA_THREAD_INDEX * in_stride;
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
			in += in_stride;
			gradients += out_stride;
		}
		biases[CUDA_THREAD_INDEX] = bias;
	}
}

// TODO(petr): Utilize tensor cores for the machine learning.
// TODO(petr): Try to compute the entire MLP within a single kernel.

class Layer
{
public:
	Layer(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t batch_sz)
		: n_in(n_in)
		, n_out(n_out)
		, batch_sz(batch_sz)
	{
		std::uint32_t aligned_n_in = (n_in + 127) & ~127;
//		std::uint32_t aligned_n_out = (n_out + 127) & ~127;

		cudaMalloc(&this->biases, n_out * sizeof(*this->biases));
		cudaMalloc(&this->weights, n_out * aligned_n_in * sizeof(*this->weights));
		cudaMalloc(&this->out, n_out * batch_sz * sizeof(*this->out));
		cudaMalloc(&this->gradients, n_out * this->batch_sz * sizeof(*this->gradients));
	}

	~Layer()
	{
		cudaFree(&this->biases);
		cudaFree(&this->weights);
		cudaFree(&this->out);
		cudaFree(&this->gradients);
	}

	// TODO(petr): Pass the input activation as a function argument, instead of a template argument.
	template <float(*IN_ACTIVATION)(float) = _activation_stub>
	const float *forward(const float *__restrict__ in)
	{
		constexpr std::uint32_t N_IN = 128;
		std::uint32_t aligned_n_in = (n_in + 127) & ~127;

		for (std::uint32_t i = 0; i < this->n_out; i += 40*N_IN)
		{
			for (std::uint32_t j = 0; j < aligned_n_in; j += N_IN)
			{
				forward_fixed_layer<N_IN, IN_ACTIVATION><<<40, N_IN>>>(
					&this->biases[i],
					&this->weights[i * aligned_n_in + j],
					&in[i * aligned_n_in + j],
					aligned_n_in,
					&this->out[i],
					this->n_out - i,
					this->n_out,
					this->batch_sz,
					!j);
			}
		}

		return this->out;
	}

	template <float(*IN_ACTIVATION_GRADIENT)(float) = _activation_stub>
	void backward(const float *__restrict__ in, const float learning_rate)
	{
		// TODO(petr): Call backward_fixed_layer

		constexpr std::uint32_t N_IN = 128;
		std::uint32_t aligned_n_in = (n_in + 127) & ~127;

		for (std::uint32_t i = 0; i < this->n_out; i += 40*N_IN)
		{
			for (std::uint32_t j = 0; j < aligned_n_in; j += N_IN)
			{
				update_fixed_layer<N_IN, IN_ACTIVATION_GRADIENT><<<40, N_IN>>>(
					&this->biases[i],
					&this->weights[i * aligned_n_in + j],
					&this->gradients[i],
					&in[i * aligned_n_in + j],
					aligned_n_in,
					this->n_out - i,
					this->n_out,
					learning_rate,
					this->batch_sz);
			}
		}
	}

private:
	std::uint32_t n_in;
	std::uint32_t n_out;
	std::uint32_t batch_sz;
	float *biases;
	float *weights;
	float *out;
	float *gradients;
};

int main(void)
{
	std::uint32_t n_in = 128;
	std::uint32_t n_out = 40*n_in;
	std::uint32_t batch_sz = 4096;
	std::uint32_t n_epochs = 100;
	Layer layer(n_in, n_out, batch_sz);

	// TODO(petr): Make the input allocation more convenient.
	float *in;
	cudaMalloc(&in, ((n_in + 127) & ~127) * batch_sz * sizeof(*in));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < n_epochs; epoch++)
	{
//		layer.forward<>(in);
		layer.backward<>(in, 0.1f);
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
