#include "nn.hh"

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
	in += in_off;
	out += out_off + CUDA_THREAD_INDEX;

	// TODO(petr): Loop through all neurons, instead of performing an out-of-bounds check.
	if (out_off + CUDA_THREAD_INDEX < out_stride) // TODO(petr): It should compare against n_out, instead of out_stride.
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

template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float)>
static const float *forward_layer(
	const float *__restrict__ biases,
	const float *__restrict__ weights,
	const float *__restrict__ in,
	const std::uint32_t n_in,
	float *__restrict__ out,
	const std::uint32_t n_out,
	const std::uint32_t batch_sz)
{
	std::uint32_t in_stride = (n_in + 127) & ~127;

	for (std::uint32_t out_off = 0; out_off < n_out; out_off += 40*N_IN)
	{
		for (std::uint32_t in_off = 0; in_off < n_in; in_off += N_IN)
		{
			forward_fixed_layer<N_IN, IN_ACTIVATION><<<40, N_IN>>>(
				biases,
				weights,
				in,
				in_stride,
				in_off,
				out,
				n_out,
				out_off,
				batch_sz);
		}
	}

	return out;
}

//
// N_OUT represents the size of the current layer, while the number of threads
// corresponds to the size of the previous layer.
// Ensure that N_OUT * 4 is less than the available shared memory capacity.
// N_OUT has to be equal to the number of threads per block.
//
template <std::uint32_t N_OUT, float(*IN_ACTIVATION_GRADIENT)(float)>
static __global__ void backward_fixed_layer(
	const float *__restrict__ weights,
	const float *__restrict__ gradients,
	const float *__restrict__ in,
	float *__restrict__ in_gradients,
	const std::uint32_t in_stride,
	const std::uint32_t in_off,
	const std::uint32_t out_stride,
	const std::uint32_t out_off,
	const std::uint32_t batch_sz)
{
	// TODO(petr): Lower the number of registers used by each thread.
	__shared__ float shrd_gradients[N_OUT];
	weights += out_off * in_stride + in_off + CUDA_THREAD_INDEX;
	gradients += out_off;
	in += in_off + CUDA_THREAD_INDEX;
	in_gradients += in_off + CUDA_THREAD_INDEX;

	if (in_off + CUDA_THREAD_INDEX < in_stride) // TODO(petr): It should compare against n_in, instead of in_stride.
	{
		for (std::uint32_t i = 0; i < batch_sz; i++)
		{
			float in_gradient = in_off ? *in_gradients : 0.0f;

			__syncthreads();
			shrd_gradients[threadIdx.x] = gradients[threadIdx.x];
			__syncthreads();

			#pragma unroll
			for (std::uint32_t j = 0; j < N_OUT; j++)
			{
				in_gradient += weights[j * in_stride] * shrd_gradients[j];
			}

			in_gradient *= IN_ACTIVATION_GRADIENT(*in);
			*in_gradients = in_gradient;

			in += in_stride;
			in_gradients += in_stride;
			gradients += out_stride;
		}
	}
}

template <std::uint32_t N_OUT, float(*IN_ACTIVATION_GRADIENT)(float)>
static void backward_layer(
	const float *__restrict__ weights,
	const float *__restrict__ gradients,
	const float *__restrict__ in,
	float *__restrict__ in_gradients,
	const std::uint32_t n_in,
	const std::uint32_t n_out,
	const std::uint32_t batch_sz)
{
	std::uint32_t in_stride = (n_in + 127) & ~127;
	std::uint32_t out_stride = (n_out + 127) & ~127;

	for (std::uint32_t in_off = 0; in_off < n_in; in_off += 40*N_OUT)
	{
		for (std::uint32_t out_off = 0; out_off < n_out; out_off += N_OUT)
		{
			backward_fixed_layer<N_OUT, IN_ACTIVATION_GRADIENT><<<40, N_OUT>>>(
				weights,
				gradients,
				in,
				in_gradients,
				in_stride,
				in_off,
				out_stride,
				out_off,
				batch_sz);
		}
	}
}

//
// N_IN represents the size of the previous layer, while the number of threads
// corresponds to the size of the current layer.
// Ensure that N_IN * 4 is less than the available shared memory capacity.
// N_IN has to be equal to the number of threads per block.
//
template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float)>
static __global__ void update_fixed_layer(
	float *__restrict__ biases,
	float *__restrict__ weights,
	const float *__restrict__ gradients,
	const float *__restrict__ in,
	const std::uint32_t in_stride,
	const std::uint32_t in_off,
	const std::uint32_t out_stride,
	const std::uint32_t out_off,
	const float learning_rate,
	const std::uint32_t batch_sz)
{
	__shared__ float shrd_in[N_IN];
	weights += (out_off + CUDA_THREAD_INDEX) * in_stride + in_off;
	in += in_off;

	if (out_off + CUDA_THREAD_INDEX < out_stride)
	{
		float bias = biases[out_off + CUDA_THREAD_INDEX];
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
		biases[out_off + CUDA_THREAD_INDEX] = bias;
	}
}

template <std::uint32_t N_IN, float(*IN_ACTIVATION)(float)>
static void update_layer(
	float *__restrict__ biases,
	float *__restrict__ weights,
	const float *__restrict__ gradients,
	const float *__restrict__ in,
	const std::uint32_t n_in,
	const std::uint32_t n_out,
	const float learning_rate,
	const std::uint32_t batch_sz)
{
	std::uint32_t in_stride = (n_in + 127) & ~127;
	std::uint32_t out_stride = (n_in + 127) & ~127;

	for (std::uint32_t out_off = 0; out_off < n_out; out_off += 40*N_IN)
	{
		for (std::uint32_t in_off = 0; in_off < n_in; in_off += N_IN)
		{
			update_fixed_layer<N_IN, IN_ACTIVATION><<<40, N_IN>>>(
				biases,
				weights,
				gradients,
				in,
				in_stride,
				in_off,
				out_stride,
				out_off,
				learning_rate,
				batch_sz);
		}
	}
}

static __device__ __forceinline__ float _activation_stub(float x) { return x; }
static __device__ __forceinline__ float forward_tanh(float x)     { return tanhf(x); }
static __device__ __forceinline__ float backward_tanh(float x)    { float y = tanhf(x); return 1.0f - y*y; }
static __device__ __forceinline__ float forward_relu(float x)     { return x > 0.0f ? x : 0.0f; }
static __device__ __forceinline__ float backward_relu(float x)    { return x >= 0.0f ? 1.0f : 0.0f; }

Layer::Layer(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t batch_sz, Activation in_activation)
	: n_in(n_in)
	, n_out(n_out)
	, batch_sz(batch_sz)
	, in_activation(in_activation)
{
	std::uint32_t in_stride = (n_in + 127) & ~127;
	std::uint32_t out_stride = (n_out + 127) & ~127;

	// TODO(petr): Randomize weights and biases.
	// TODO(petr): Are they correctly allocated?
	cudaMalloc(&this->biases, n_out * sizeof(*this->biases));
	cudaMalloc(&this->weights, n_out * in_stride * sizeof(*this->weights));
	cudaMalloc(&this->out, n_out * this->batch_sz * sizeof(*this->out));
	cudaMalloc(&this->gradients, out_stride * this->batch_sz * sizeof(*this->gradients));
}

Layer::~Layer()
{
	cudaFree(&this->biases);
	cudaFree(&this->weights);
	cudaFree(&this->out);
	cudaFree(&this->gradients);
}

const float *Layer::forward(const float *__restrict__ in) const
{
	switch (this->in_activation)
	{
	case Activation::Tanh:
		return forward_layer<128, forward_tanh>(this->biases, this->weights, in, this->n_in, this->out, this->n_out, this->batch_sz);
	case Activation::ReLU:
		return forward_layer<128, forward_relu>(this->biases, this->weights, in, this->n_in, this->out, this->n_out, this->batch_sz);
	default:
		return forward_layer<128, _activation_stub>(this->biases, this->weights, in, this->n_in, this->out, this->n_out, this->batch_sz);
	}
}

void Layer::backward(const float *__restrict__ in, float *__restrict__ in_gradients) const
{
	switch (this->in_activation)
	{
	case Activation::Tanh:
		return backward_layer<128, backward_tanh>(this->weights, this->gradients, in, in_gradients, this->n_in, this->n_out, this->batch_sz);
	case Activation::ReLU:
		return backward_layer<128, backward_relu>(this->weights, this->gradients, in, in_gradients, this->n_in, this->n_out, this->batch_sz);
	default:
		return backward_layer<128, _activation_stub>(this->weights, this->gradients, in, in_gradients, this->n_in, this->n_out, this->batch_sz);
	}
}

void Layer::update(const float *__restrict__ in, float learning_rate)
{
	switch (this->in_activation)
	{
	case Activation::Tanh:
		return update_layer<128, forward_tanh>(this->biases, this->weights, this->gradients, in, this->n_in, this->n_out, learning_rate, this->batch_sz);
	case Activation::ReLU:
		return update_layer<128, forward_relu>(this->biases, this->weights, this->gradients, in, this->n_in, this->n_out, learning_rate, this->batch_sz);
	default:
		return update_layer<128, _activation_stub>(this->biases, this->weights, this->gradients, in, this->n_in, this->n_out, learning_rate, this->batch_sz);
	}
}
