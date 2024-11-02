#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>

#include "nn.hh"

// TODO(petr): Utilize tensor cores for the machine learning.
// TODO(petr): Try to compute the entire MLP within a single kernel.

int main(void)
{
	std::uint32_t n_in = 128;
	std::uint32_t n_out = 40*n_in;
	std::uint32_t batch_sz = 4096;
	std::uint32_t n_epochs = 100;
	Layer in_layer(28*28, n_in, batch_sz, Activation::None);
	Layer layer(n_in, n_out, batch_sz, Activation::None);

	// TODO(petr): Make the input allocation more convenient.
	float *in;
	cudaMalloc(&in, ((n_in + 127) & ~127) * batch_sz * sizeof(*in));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (std::size_t epoch = 0; epoch < n_epochs; epoch++)
	{
		layer.forward(in);
//		layer.backward(in_layer);
//		layer.update(in, 0.1f);
	}
	cudaEventRecord(stop);

	float milliseconds = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed time: " << milliseconds << "ms" << std::endl;

	return 0;
}
