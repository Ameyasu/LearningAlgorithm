#pragma once

#include "ActivationFunction.h"

template <typename T>
class Identity : public ActivationFunction<T>
{
public:
	T operator()(T x) const override
	{
		return x;
	}
	explicit operator ActFncID() const noexcept override
	{
		return ActFncID::IDENTITY;
	}
};
