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
		if (cudaPeekAtLastError()) { \
			std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

static __global__ void init_random_state(curandState *state, unsigned long seed)
{
	curand_init(seed, CUDA_THREAD_INDEX, 0, &state[CUDA_THREAD_INDEX]);
}

static __global__ void randomize_array(curandState *state, float *array, std::uint32_t size)
{
	for (std::uint32_t i = CUDA_THREAD_INDEX; i < size; i += CUDA_THREAD_COUNT)
	{
		array[i] = curand_uniform(&state[CUDA_THREAD_INDEX]);
	}
}

float *get_random_array(std::uint32_t size)
{
	static curandState *state = nullptr;
	constexpr std::size_t threads = 64;
	if (!state)
	{
		cudaMalloc(&state, threads * sizeof(*state));
//		unsigned long seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		unsigned long seed = 1234;
		init_random_state<<<1, threads>>>(state, seed);
	}
	float *result;
	cudaMalloc(&result, size * sizeof(*result));
	randomize_array<<<1, threads>>>(state, result, size);
	return result;
}

class Layer
{
public:
	Layer(std::uint32_t n_in, std::uint32_t n_out)
		: n_in(n_in)
		, n_out(n_out)
	{
		this->weights = get_random_array(n_out * n_in);
		this->biases = get_random_array(n_out);
	}

	~Layer()
	{
		cudaFree(this->weights);
		cudaFree(this->biases);
	}

private:
	std::uint32_t n_in;
	std::uint32_t n_out;
	float *weights;
	float *biases;
};

// TODO(petr): It probably won't be necessary to use 32-bit integers everywhere, since it would be better to have kernels that compute a fixed number of 'things.'.
// Asynchronously copy biases into the result array, and then compute the weights in parallel.

int main(void)
{
	Layer layer(10, 10);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < 10; epoch++)
	{
	}
	cudaEventRecord(stop);
	ASSERT_CUDA_ERROR();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
