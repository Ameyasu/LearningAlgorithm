#pragma once

#include "ActivationFunction.h"
#include <cmath>

class Sigmoid : public ActivationFunction<double>
{
public:
	double operator()(double x) const override
	{
		return 1.0 / (1.0 + exp(x));
	}
	explicit operator ActFncID() const noexcept override
	{
		return ActFncID::SIGMOID;
	}
};
