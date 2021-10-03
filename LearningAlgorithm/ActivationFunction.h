#pragma once

enum class ActFncID
{
	IDENTITY,
	RELU,
	SIGMOID,
	STEP
};

template <typename T>
class ActivationFunction
{
public:
	ActivationFunction() = default;
	virtual ~ActivationFunction() = default;

public:
	virtual T operator()(T x) const = 0;
	virtual explicit operator ActFncID() const noexcept = 0;
};