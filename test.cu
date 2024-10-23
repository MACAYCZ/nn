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
	const std::uint32_t in_off,
	float *__restrict__ out,
	const std::uint32_t out_stride,
	const std::uint32_t out_off,
	const std::uint32_t batch_sz)
{
	__shared__ float shrd_in[N_IN];
	weights += (out_off + CUDA_THREAD_INDEX) * in_stride + in_off;
	out += out_off + CUDA_THREAD_INDEX;

	// TODO(petr): Loop through all neurons, instead of performing an out-of-bounds check.
	if (out_off + CUDA_THREAD_INDEX < out_stride)
	{
		const float bias = biases[out_off + CUDA_THREAD_INDEX];
		for (std::uint32_t i = 0; i < batch_sz; i++)
		{
			float result = in_off ? *out : bias;

			__syncthreads();
			shrd_in[threadIdx.x] = IN_ACTIVATION(in[threadIdx.x]);
			__syncthreads();

			#pragma unroll
			for (std::uint32_t j = 0; j < N_IN; j++)
			{
				result += weights[j] * shrd_in[j];
			}

			*out = result;
			in += in_stride;
			out += out_stride;
		}
	}
}

// TODO(petr): Utilize tensor cores for the machine learning.
// TODO(petr): Try to compute the entire MLP within a single kernel.
// TODO(petr): Try to use expression templates for constructing the neural network.

class Layer
{
public:
	Layer(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t batch_sz)
		: n_in(n_in)
		, n_out(n_out)
		, batch_sz(batch_sz)
	{
		std::uint32_t in_stride = (n_in + 127) & ~127;

		cudaMalloc(&this->biases, n_out * sizeof(*this->biases));
		cudaMalloc(&this->weights, n_out * in_stride * sizeof(*this->weights));
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
		std::uint32_t in_stride = (this->n_in + 127) & ~127;

		for (std::uint32_t out_off = 0; out_off < this->n_out; out_off += 40*N_IN)
		{
			for (std::uint32_t in_off = 0; in_off < in_stride; in_off += N_IN)
			{
				forward_fixed_layer<N_IN, IN_ACTIVATION><<<40, N_IN>>>(
					this->biases,
					this->weights,
					in,
					in_stride,
					in_off,
					this->out,
					this->n_out,
					out_off,
					this->batch_sz);
			}
		}

		ASSERT_CUDA_ERROR();
		return this->out;
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
		layer.forward<>(in);
	}
	cudaEventRecord(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
