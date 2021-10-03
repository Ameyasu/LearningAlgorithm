#pragma once

#include <random>

template<typename T>
class Random
{
public:
	Random()
		: m_engine(std::random_device()())
	{
	}

	T operator()(T min, T max)
	{
		if constexpr (std::is_same_v<T, int>)
			return std::uniform_int_distribution<>(min, max)(m_engine);
		else
			return std::uniform_real_distribution<>(min, max)(m_engine);
	}

private:
	std::mt19937 m_engine;
};
