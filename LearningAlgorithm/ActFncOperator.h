#pragma once

#include "ActivationFunction.h"
#include "Identity.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "Step.h"

#include <string>
#include <cassert>

class ActFncOperator
{
public:
	static constexpr const char* toString(ActFncID id)
	{
		if (in(id))
			return TEXT[static_cast<int>(id)].data();
		throw;
	}

	static constexpr ActFncID fromString(const char* s)
	{
		for (int id = BEGIN; id < END; ++id)
		{
			if (s == TEXT[id])
				return ActFncID(id);
		}
		throw;
	}

	template<typename T>
	static constexpr ActivationFunction<T>* create(ActFncID id)
	{
		switch (id)
		{
		case ActFncID::IDENTITY:
			return new Identity<T>;
		case ActFncID::RELU:
			return new ReLU<T>;
		case ActFncID::SIGMOID:
			if constexpr (std::is_same_v<T, double>)
				return new Sigmoid;
			else
				break;
		case ActFncID::STEP:
			return new Step<T>;
		}
		throw;
	}

private:
	static constexpr std::string_view TEXT[] =
	{
		"Identity",
		"ReLU",
		"Sigmoid",
		"Step"
	};
	static constexpr int BEGIN = static_cast<int>(ActFncID::IDENTITY);
	static constexpr int END = static_cast<int>(ActFncID::STEP) + 1;
	static constexpr bool in(ActFncID id)
	{
		return ActFncID(BEGIN) <= id && id < ActFncID(END);
	}

private:
	ActFncOperator() = delete;
};
