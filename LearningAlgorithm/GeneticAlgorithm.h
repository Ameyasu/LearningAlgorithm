#pragma once

#include "Random.h"
#include <memory>
#include <algorithm>

/*
* template<typename Gene, typename Fitness>
* Gene    ���F�̂̌^ int��double
* Fitness �K���x�̌^ int��double
* 
* ���̃N���X�̐������܂���clear�֐����Ă񂾏�Ԃł�
* �܂�reset�֐������s���邱��
* �������Ȃ��ꍇ���̑��̊֐����Ăяo���Ȃ�����
* 
* ���̃N���X�ɂ�����nullptr�̓��o�͈͂�؂Ȃ�
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
	* �N���X�������Ɠ�����Ԃɂ���
	* �m�ۂ���������������ΑS�ĉ������
	*/
	void clear();

	/*
	* �p�����[�^�ɉ����ĕK�v�ȃ��������m�ۂ���
	* ���Ɋm�ۂ��Ă������������͉������
	* 
	* @param population         �P���㓖����̌̐� (�l��)
	* @param chromosomeLength   �P�̓�����̐��F�̂̒���
	* @param chromosomeValueMin ���F�̂̓��e�Ƃ��Ď�蓾��ŏ��l (�܂�)
	* @param chromosomeValueMax ���F�̂̓��e�Ƃ��Ď�蓾��ő�l (�܂�)
	* @param eliteNum           ������Ɏ����z���G���[�g�̐� �K���l�������̒l
	* @param generation         ���㐔 ���ʂȗ��R������ꍇ�̂ݐݒ�
	*/
	void reset(int population, int chromosomeLength, Gene chromosomeValueMin, Gene chromosomeValueMax, int eliteNum, int generation = 1);

	/*
	* �S�̂̐��F�̂̓��e��ݒ肷��
	* 
	* @param individuals �S�̂̐��F�̂̔z�� �T�C�Y = �l�� * ���F�̂̒���
	*     N�̖�(0-based)�̐��F�� = individuals[(���F�̂̒��� * N) �` ((���F�̂̒��� * (N + 1)) - 1)]
	*/
	void setIndividuals(const Gene* individuals);

	/*
	* �S�̂̐��F�̂̓��e�������_���ɐݒ肷��
	* 
	* @param min �����_���̍ŏ��l (�܂�)
	* @param max �����_���̍ő�l (�܂�)
	*/
	void setIndividualsRandom(Gene min, Gene max);

	/*
	* �S�̂̓K���x��ݒ肷��
	* 
	* ((�K���x�̍ő�l - �K���x�̍ŏ��l) * �l��) ���I�[�o�[�t���[���Ȃ��悤�ɏ\���ȗ]�T���������邱��
	* 
	* @param fitnesses �K���x�̔z�� �T�C�Y = �l��
	*     N�̖�(0-based)�̓K���x = fitnesses[N]
	*/
	void evaluate(const Fitness* fitnesses);

	/*
	* evaluate�֐��Őݒ肳�ꂽ�K���x�����Ɏ�����𐶐�����
	* 
	* �G���[�g�̂� (�G���[�g�̐� >= 1) �̏ꍇ
	* �K���x���������ɃC���f�b�N�X0����z�u�����
	*/
	void generateNextGeneration();

	// @return ���݂̐��㐔 (1-based)
	int getGeneration() const;

	// @return �P���㓖����̌̐�
	int getPopulation() const;

	// @return �P�̓�����̐��F�̂̒���
	int getChromosomeLength() const;

	// @return ���F�̂̓��e�Ƃ��Ď�蓾��ŏ��l (�܂�)
	Gene getChromosomeValueMin() const;

	// @return ���F�̂̓��e�Ƃ��Ď�蓾��ő�l (�܂�)
	Gene getChromosomeValueMax() const;

	// @return ������Ɏ����z���G���[�g�̐�
	int getEliteNum() const;
	
	// @return �S�̂̐��F�̂̔z��
	const Gene* getIndividuals() const;

	// @return index�Ԗ�(0-based)�̌̂̐��F�̂̔z��
	const Gene* getIndividual(int index) const;

	// @return �S�̂̓K���x�̔z��
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

	// �K���x���傫�����Ƀ\�[�g����
	std::sort(m_sortIndex.get(), m_sortIndex.get() + m_population, [this](int lhs, int rhs)
		{ return m_fitnesses[lhs] > m_fitnesses[rhs]; }
	);

	// ������ɃG���[�g�������z��
	for (int i = 0; i < m_eliteNum; ++i)
		memcpy(&m_individualsTmp[i * m_chromosomeLength], &m_individuals[m_sortIndex[i] * m_chromosomeLength], sizeof(Gene) * m_chromosomeLength);

	// �K���x�̍��v�l�v�Z (���[���b�g�I���Ɏg��)
	Fitness fitnessBase = (-m_fitnesses[m_sortIndex[m_population - 1]]) + 1;
	Fitness fitnessSum = 0;
	for (int i = m_eliteNum; i < m_population; ++i)
		fitnessSum += m_fitnesses[m_sortIndex[i]] + fitnessBase;

	auto rndF = Random<Fitness>();

	// ������̌̂��P���[�v�ɕt���P�̐���
	for (int i = m_eliteNum; i < m_population; ++i)
	{
		// ���[���b�g�I���łQ�̑I��
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

		// �u�����h���� (BLX-��)
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

	// ���������������������Ƃ���
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
