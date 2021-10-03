#pragma once

#include "ActivationFunction.h"

template <typename T>
class ReLU : public ActivationFunction<T>
{
public:
	T operator()(T x) const override
	{
		return x >= 0 ? x : 0;
	}
	explicit operator ActFncID() const noexcept override
	{
		return ActFncID::RELU;
	}
};
