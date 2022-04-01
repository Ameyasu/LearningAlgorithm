#pragma once

#include "Random.h"
#include <memory>
#include <algorithm>

/*
* template<typename Gene, typename Fitness>
* Gene    染色体の型 intかdouble
* Fitness 適応度の型 intかdouble
* 
* このクラスの生成時またはclear関数を呼んだ状態では
* まずreset関数を実行すること
* そうしない場合その他の関数を呼び出さないこと
* 
* このクラスにおいてnullptrの入出力は一切なし
*/
template<typename Gene, typename Fitness>
class GeneticAlgorithm
{
	static_assert(std::is_same_v<Gene, int> || std::is_same_v<Gene, double>, "GeneticAlgorithm template is only int or double");
	static_assert(std::is_same_v<Fitness, int> || std::is_same_v<Fitness, double>, "GeneticAlgorithm template is only int or double");

public:
	GeneticAlgorithm();
	~GeneticAlgorithm();

	GeneticAlgorithm(const GeneticAlgorithm&) = delete;
	GeneticAlgorithm& operator=(const GeneticAlgorithm&) = delete;

	GeneticAlgorithm(GeneticAlgorithm&&) = default;
	GeneticAlgorithm& operator=(GeneticAlgorithm&&) = default;

public:
	/*
	* クラス生成時と同じ状態にする
	* 確保したメモリがあれば全て解放する
	*/
	void clear();

	/*
	* パラメータに応じて必要なメモリを確保する
	* 既に確保してあったメモリは解放する
	* 
	* @param population         １世代当たりの個体数 (人口)
	* @param chromosomeLength   １個体当たりの染色体の長さ
	* @param chromosomeValueMin 染色体の内容として取り得る最小値 (含む)
	* @param chromosomeValueMax 染色体の内容として取り得る最大値 (含む)
	* @param eliteNum           次世代に持ち越すエリート個体数 必ず人口未満の値
	* @param generation         世代数 特別な理由がある場合のみ設定
	*/
	void reset(int population, int chromosomeLength, Gene chromosomeValueMin, Gene chromosomeValueMax, int eliteNum, int generation = 1);

	/*
	* 全個体の染色体の内容を設定する
	* 
	* @param individuals 全個体の染色体の配列 サイズ = 人口 * 染色体の長さ
	*     N個体目(0-based)の染色体 = individuals[(染色体の長さ * N) ～ ((染色体の長さ * (N + 1)) - 1)]
	*/
	void setIndividuals(const Gene* individuals);

	/*
	* 全個体の染色体の内容をランダムに設定する
	* 
	* @param min ランダムの最小値 (含む)
	* @param max ランダムの最大値 (含む)
	*/
	void setIndividualsRandom(Gene min, Gene max);

	/*
	* 全個体の適応度を設定する
	* 
	* ((適応度の最大値 - 適応度の最小値) * 人口) がオーバーフローしないように十分な余裕を持たせること
	* 
	* @param fitnesses 適応度の配列 サイズ = 人口
	*     N個体目(0-based)の適応度 = fitnesses[N]
	*/
	void evaluate(const Fitness* fitnesses);

	/*
	* evaluate関数で設定された適応度を元に次世代を生成する
	* 
	* エリート個体は (エリート個体数 >= 1) の場合
	* 適応度が高い順にインデックス0から配置される
	*/
	void generateNextGeneration();

	// @return 現在の世代数 (1-based)
	int getGeneration() const;

	// @return １世代当たりの個体数
	int getPopulation() const;

	// @return １個体当たりの染色体の長さ
	int getChromosomeLength() const;

	// @return 染色体の内容として取り得る最小値 (含む)
	Gene getChromosomeValueMin() const;

	// @return 染色体の内容として取り得る最大値 (含む)
	Gene getChromosomeValueMax() const;

	// @return 次世代に持ち越すエリート個体数
	int getEliteNum() const;
	
	// @return 全個体の染色体の配列
	const Gene* getIndividuals() const;

	// @return index番目(0-based)の個体の染色体の配列
	const Gene* getIndividual(int index) const;

	// @return 全個体の適応度の配列
	const Fitness* getFitnesses() const;

private:
	int m_generation;
	int m_population;
	int m_chromosomeLength;
	Gene m_chromosomeValueMin;
	Gene m_chromosomeValueMax;
	int m_eliteNum;
	std::unique_ptr<Gene[]> m_individuals;
	std::unique_ptr<Gene[]> m_individualsTmp;
	std::unique_ptr<Fitness[]> m_fitnesses;
	std::unique_ptr<int[]> m_sortIndex;
};




template<typename Gene, typename Fitness>
inline GeneticAlgorithm<Gene, Fitness>::GeneticAlgorithm()
	: m_generation(1)
	, m_population()
	, m_chromosomeLength()
	, m_chromosomeValueMin()
	, m_chromosomeValueMax()
	, m_eliteNum()
	, m_individuals()
	, m_individualsTmp()
	, m_fitnesses()
	, m_sortIndex()
{
}

template<typename Gene, typename Fitness>
inline GeneticAlgorithm<Gene, Fitness>::~GeneticAlgorithm()
{
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::clear()
{
	m_generation = 1;
	m_population = 0;
	m_chromosomeLength = 0;
	m_chromosomeValueMin = 0;
	m_chromosomeValueMax = 0;
	m_eliteNum = 0;
	m_individuals.reset();
	m_individualsTmp.reset();
	m_fitnesses.reset();
	m_sortIndex.reset();
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::reset(int population, int chromosomeLength, Gene chromosomeValueMin, Gene chromosomeValueMax, int eliteNum, int generation)
{
	m_generation = generation;
	m_population = population;
	m_chromosomeLength = chromosomeLength;
	m_chromosomeValueMin = chromosomeValueMin;
	m_chromosomeValueMax = chromosomeValueMax;
	m_eliteNum = eliteNum;
	m_individuals.reset(new Gene[population * chromosomeLength]);
	m_individualsTmp.reset(new Gene[population * chromosomeLength]);
	m_fitnesses.reset(new Gene[population]);
	m_sortIndex.reset(new int[population]);

	for (int i = 0; i < population; ++i)
		m_sortIndex[i] = i;
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::setIndividuals(const Gene* individuals)
{
	memcpy(m_individuals.get(), individuals, sizeof(Gene) * m_population * m_chromosomeLength);
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::setIndividualsRandom(Gene min, Gene max)
{
	auto random = Random<Gene>();

	int size = m_population * m_chromosomeLength;
	for (int i = 0; i < size; ++i)
		m_individuals[i] = random(min, max);
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::evaluate(const Fitness* fitnesses)
{
	memcpy(m_fitnesses.get(), fitnesses, sizeof(Fitness) * m_population);
}

template<typename Gene, typename Fitness>
inline void GeneticAlgorithm<Gene, Fitness>::generateNextGeneration()
{
	for (int i = 0; i < m_population; ++i)
		m_sortIndex[i] = i;

	// 適応度が大きい順にソートする
	std::sort(m_sortIndex.get(), m_sortIndex.get() + m_population, [this](int lhs, int rhs)
		{ return m_fitnesses[lhs] > m_fitnesses[rhs]; }
	);

	// 次世代にエリートを持ち越す
	for (int i = 0; i < m_eliteNum; ++i)
		memcpy(&m_individualsTmp[i * m_chromosomeLength], &m_individuals[m_sortIndex[i] * m_chromosomeLength], sizeof(Gene) * m_chromosomeLength);

	// 適応度の合計値計算 (ルーレット選択に使う)
	Fitness fitnessBase = (-m_fitnesses[m_sortIndex[m_population - 1]]) + 1;
	Fitness fitnessSum = 0;
	for (int i = m_eliteNum; i < m_population; ++i)
		fitnessSum += m_fitnesses[m_sortIndex[i]] + fitnessBase;

	auto rndF = Random<Fitness>();

	// 次世代の個体を１ループに付き１個体生成
	for (int i = m_eliteNum; i < m_population; ++i)
	{
		// ルーレット選択で２個体選択
		Gene* indv[2] = {};
		for (int j = 0; j < 2; ++j)
		{
			Fitness r = rndF(0, fitnessSum);
			for (int k = m_eliteNum; k < m_population; ++k)
			{
				r -= m_fitnesses[m_sortIndex[k]] + fitnessBase;
				if (r <= 0)
				{
					indv[j] = &m_individuals[m_sortIndex[k] * m_chromosomeLength];
					break;
				}
			}
		}

		auto rndG = Random<Gene>();
		Gene* secondIndv = &m_individualsTmp[i * m_chromosomeLength];

		// ブレンド交叉 (BLX-α)
		for (int j = 0; j < m_chromosomeLength; ++j)
		{
			Gene diff = std::abs(indv[0][j] - indv[1][j]) / 2;
			if constexpr (std::is_same_v<Gene, int>)
			{
				if (diff == 0)
					diff = 1;
			}
			else
			{
				if (diff < std::numeric_limits<double>::epsilon() * 2.0)
					diff = std::numeric_limits<double>::epsilon() * 2.0;
			}
			Gene min = std::max(m_chromosomeValueMin, std::min(indv[0][j], indv[1][j]) - diff);
			Gene max = std::min(m_chromosomeValueMax, std::max(indv[0][j], indv[1][j]) + diff);
			secondIndv[j] = rndG(min, max);
		}
	}

	// 生成した次世代を現世代とする
	m_individuals.swap(m_individualsTmp);
	++m_generation;
}

template<typename Gene, typename Fitness>
inline int GeneticAlgorithm<Gene, Fitness>::getGeneration() const
{
	return m_generation;
}

template<typename Gene, typename Fitness>
inline int GeneticAlgorithm<Gene, Fitness>::getPopulation() const
{
	return m_population;
}

template<typename Gene, typename Fitness>
inline int GeneticAlgorithm<Gene, Fitness>::getChromosomeLength() const
{
	return m_chromosomeLength;
}

template<typename Gene, typename Fitness>
inline Gene GeneticAlgorithm<Gene, Fitness>::getChromosomeValueMin() const
{
	return m_chromosomeValueMin;
}

template<typename Gene, typename Fitness>
inline Gene GeneticAlgorithm<Gene, Fitness>::getChromosomeValueMax() const
{
	return m_chromosomeValueMax;
}

template<typename Gene, typename Fitness>
inline int GeneticAlgorithm<Gene, Fitness>::getEliteNum() const
{
	return m_eliteNum;
}

template<typename Gene, typename Fitness>
inline const Gene* GeneticAlgorithm<Gene, Fitness>::getIndividuals() const
{
	return m_individuals.get();
}

template<typename Gene, typename Fitness>
inline const Gene* GeneticAlgorithm<Gene, Fitness>::getIndividual(int index) const
{
	return &m_individuals[index * m_chromosomeLength];
}

template<typename Gene, typename Fitness>
inline const Fitness* GeneticAlgorithm<Gene, Fitness>::getFitnesses() const
{
	return m_fitnesses.get();
}
