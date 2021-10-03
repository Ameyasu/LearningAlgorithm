#pragma once

#include "ActFncOperator.h"
#include "Random.h"
#include <memory>

/*
* template<typename T>
* T ���́E�o�́E�d�݂̌^ int��double
* 
* ���̃N���X�̐������܂���clear�֐����Ă񂾏�Ԃł�
* �܂��@�`�D�̊֐������Ɏ��s���邱��
* �������Ȃ��ꍇ���̑��̊֐����Ăяo���Ȃ�����
* 
* ���̃N���X�ɂ�����nullptr�̓��o�͈͂�؂Ȃ�
*/
template<typename T>
class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();

	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;

	NeuralNetwork(NeuralNetwork&&) = default;
	NeuralNetwork& operator=(NeuralNetwork&&) = default;

public:
	/*
	* �N���X�������Ɠ�����Ԃɂ���
	* �m�ۂ���������������ΑS�ĉ������
	*/
	void clear();

	/*
	* �@���͑w�̐ݒ�
	* 
	* �o�C�A�X�m�[�h����
	* 
	* @param size �m�[�h�� (�o�C�A�X�m�[�h���܂܂Ȃ�)
	*/
	void setInputLayer(int size);

	/*
	* �A���ԑw�̐��̐ݒ�
	* 
	* @param num ���ԑw�̐� �P�ȏ�
	*/
	void setHiddenLayerNum(int num);

	/*
	* �B���ԑw�̐ݒ�
	* 
	* �A�Őݒ肵���񐔂������̊֐����J��Ԃ��Ă�
	* �ĂԂ��Ƃɓ��͑w�ɋ߂������珇�ɐݒ肷��
	* �o�C�A�X�m�[�h����
	* 
	* @param size   �m�[�h�� (�o�C�A�X�m�[�h���܂܂Ȃ�)
	* @param actFnc �������֐���ID
	*/
	void setHiddenLayer(int size, ActFncID actFncID);

	/*
	* �C�o�͑w�̐ݒ�
	* 
	* �o�C�A�X�m�[�h�Ȃ�
	* �d�݂̃T�C�Y�͂����Ōv�Z����A�d�݂����������
	*
	* @param size   �m�[�h��
	* @param actFnc �������֐���ID
	*/
	void setOutputLayer(int size, ActFncID actFncID);

	/*
	* �D�d�݂̐ݒ�
	* 
	* ���ꂩ�D'�̂ǂ��炩���Ă�
	* 
	* @param weight �d�݂̔z�� �T�C�Y = getWeightSize�֐�
	*/
	void setWeight(const T* weight);

	/*
	* �D'�d�݂������_���ɐݒ�
	* 
	* ���ꂩ�D�̂ǂ��炩���Ă�
	* 
	* @param min �����_���̍ŏ��l (�܂�)
	* @param max �����_���̍ő�l (�܂�)
	*/
	void setWeightRandom(T min, T max);

	/*
	* ���`�d���ē��͂���o�͂𓾂�
	* 
	* @param input ���͔z�� �T�C�Y = getInputLayerSize�֐�
	* @return �o�͔z�� �T�C�Y = getOutputLayerSize�֐�
	*/
	const T* forwardPropagation(const T* input);

	/*
	* �덷�t�`�d���ďd�݂𒲐�����
	* 
	* @param input  ���`�d�ɂ�������͔z��
	* @param output ���͂ɑ΂��Ė]�ޏo�͔z��
	*/
	void backpropagation(const T* input, const T* output);

	// @return ���͑w�̃m�[�h��
	int getInputLayerSize() const;

	// @return ���ԑw�̐�
	int getHiddenLayerNum() const;

	// @return index�Ԗ�(0-based)�̒��ԑw�̃m�[�h��
	int getHiddenLayerSize(int index) const;

	// @return index�Ԗ�(0-based)�̒��ԑw�̊������֐�
	ActFncID getHiddenLayerActFncID(int index) const;

	// @return �o�͑w�̃m�[�h��
	int getOutputLayerSize() const;

	// @return �o�͑w�̊������֐�
	ActFncID getOutputLayerActFncID() const;

	/*
	* �@�`�C��ݒ肵�ď��߂ē���
	* 
	* @return �d�݂̃T�C�Y
	*/
	int getWeightSize() const;

	// @return �d�݂̔z��
	const T* getWeight() const;

private:
	struct Layer
	{
		int size = 0;
		std::unique_ptr<T[]> layer;
		std::unique_ptr<ActivationFunction<T>> actFnc;

		void clear()
		{
			size = 0;
			layer.reset();
			actFnc.reset();
		}
	};

	Layer m_inputLayer;
	int m_hiddenLayerNum;
	std::unique_ptr<Layer[]> m_hiddenLayer;
	Layer m_outputLayer;
	int m_weightSize;
	std::unique_ptr<T[]> m_weight;
};




template<typename T>
inline NeuralNetwork<T>::NeuralNetwork()
	: m_inputLayer()
	, m_hiddenLayerNum()
	, m_hiddenLayer()
	, m_outputLayer()
	, m_weightSize()
	, m_weight()
{
}

template<typename T>
inline NeuralNetwork<T>::~NeuralNetwork()
{
}

template<typename T>
inline void NeuralNetwork<T>::clear()
{
	m_inputLayer.clear();
	m_hiddenLayerNum = 0;
	m_hiddenLayer.reset();
	m_outputLayer.clear();
	m_weightSize = 0;
	m_weight.reset();
}

template<typename T>
inline void NeuralNetwork<T>::setInputLayer(int size)
{
	m_inputLayer.size = size;
	m_inputLayer.layer.reset(new T[size + 1]);
	m_inputLayer.layer[size] = 1;
}

template<typename T>
inline void NeuralNetwork<T>::setHiddenLayerNum(int num)
{
	m_hiddenLayerNum = 0;
	m_hiddenLayer.reset(new Layer[num]);
}

template<typename T>
inline void NeuralNetwork<T>::setHiddenLayer(int size, ActFncID actFncID)
{
	auto& hiddenLayer = m_hiddenLayer[m_hiddenLayerNum++];
	hiddenLayer.size = size;
	hiddenLayer.layer.reset(new T[size + 1]);
	hiddenLayer.layer[size] = 1;
	hiddenLayer.actFnc.reset(ActFncOperator::create<T>(actFncID));
}

template<typename T>
inline void NeuralNetwork<T>::setOutputLayer(int size, ActFncID actFncID)
{
	m_outputLayer.size = size;
	m_outputLayer.layer.reset(new T[size]);
	m_outputLayer.actFnc.reset(ActFncOperator::create<T>(actFncID));

	// ���͑w�ƒ��ԑw�̏d�݃T�C�Y
	m_weightSize = (m_inputLayer.size + 1) * m_hiddenLayer[0].size;

	// ���ԑw���m�̏d�݃T�C�Y
	for (int i = 0; i < m_hiddenLayerNum - 1; ++i)
		m_weightSize += (m_hiddenLayer[i].size + 1) * m_hiddenLayer[i + 1].size;
	
	// ���ԑw�Əo�͑w�̏d�݃T�C�Y
	m_weightSize += (m_hiddenLayer[m_hiddenLayerNum - 1].size + 1) * m_outputLayer.size;

	m_weight.reset(new T[m_weightSize]);
}

template<typename T>
inline void NeuralNetwork<T>::setWeight(const T* weight)
{
	memcpy(m_weight.get(), weight, sizeof(T) * m_weightSize);
}

template<typename T>
inline void NeuralNetwork<T>::setWeightRandom(T min, T max)
{
	auto random = Random<T>();
	for (int i = 0; i < m_weightSize; ++i)
		m_weight[i] = random(min, max);
}

template<typename T>
inline const T* NeuralNetwork<T>::forwardPropagation(const T* input)
{
	for (int i = 0; i < m_inputLayer.size; ++i)
		m_inputLayer.layer[i] = input[i];

	int weightIndex = 0;

	// ���͑w�ƒ��ԑw
	for (int h = 0; h < m_hiddenLayer[0].size; ++h)
	{
		T sum = 0;
		for (int i = 0; i < m_inputLayer.size + 1; ++i)
		{
			sum += m_inputLayer.layer[i] * m_weight[weightIndex++];
		}
		m_hiddenLayer[0].layer[h] = (*m_hiddenLayer[0].actFnc)(sum);
	}

	// ���ԑw���m
	for (int n = 0; n < m_hiddenLayerNum - 1; ++n)
	{
		for (int h2 = 0; h2 < m_hiddenLayer[n + 1].size; ++h2)
		{
			T sum = 0;
			for (int h = 0; h < m_hiddenLayer[n].size + 1; ++h)
			{
				sum += m_hiddenLayer[n].layer[h] * m_weight[weightIndex++];
			}
			m_hiddenLayer[n + 1].layer[h2] = (*m_hiddenLayer[n + 1].actFnc)(sum);
		}
	}

	// ���ԑw�Əo�͑w
	for (int o = 0; o < m_outputLayer.size; ++o)
	{
		T sum = 0;
		for (int h = 0; h < m_hiddenLayer[m_hiddenLayerNum - 1].size + 1; ++h)
		{
			sum += m_hiddenLayer[m_hiddenLayerNum - 1].layer[h] * m_weight[weightIndex++];
		}
		m_outputLayer.layer[o] = (*m_outputLayer.actFnc)(sum);
	}

	return m_outputLayer.layer.get();
}

template<typename T>
inline void NeuralNetwork<T>::backpropagation(const T* input, const T* output)
{
}

template<typename T>
int NeuralNetwork<T>::getInputLayerSize() const
{
	return m_inputLayer.size;
}

template<typename T>
inline int NeuralNetwork<T>::getHiddenLayerNum() const
{
	return m_hiddenLayerNum;
}

template<typename T>
inline int NeuralNetwork<T>::getHiddenLayerSize(int index) const
{
	return m_hiddenLayer[index].size;
}

template<typename T>
inline ActFncID NeuralNetwork<T>::getHiddenLayerActFncID(int index) const
{
	return static_cast<ActFncID>(*m_hiddenLayer[index].actFnc);
}

template<typename T>
inline int NeuralNetwork<T>::getOutputLayerSize() const
{
	return m_outputLayer.size;
}

template<typename T>
inline ActFncID NeuralNetwork<T>::getOutputLayerActFncID() const
{
	return static_cast<ActFncID>(*m_outputLayer.actFnc);
}

template<typename T>
inline int NeuralNetwork<T>::getWeightSize() const
{
	return m_weightSize;
}

template<typename T>
inline const T* NeuralNetwork<T>::getWeight() const
{
	return m_weight.get();
}

