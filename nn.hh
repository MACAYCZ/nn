#pragma once
#include <cstdint>

enum class Activation
{
	None,
	Tanh,
	ReLU,
};

class
#if defined(_WINDLL)
	__declspec(dllexport)
#endif // defined(_WINDLL)
	Layer
{
public:
	Layer(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t batch_sz, Activation in_activation);
	~Layer();

	const float *forward(const float *__restrict__ in) const;
	void backward(const float *__restrict__ in, float *__restrict__ in_gradients) const;
	void update(const float *__restrict__ in, float learning_rate);

private:
	Activation in_activation;
	std::uint32_t n_in;
	std::uint32_t n_out;
	std::uint32_t batch_sz;
	float *biases;
	float *weights;
	float *out;
	float *gradients;
};
