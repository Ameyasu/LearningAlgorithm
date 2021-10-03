#pragma once

#include "ActFncOperator.h"
#include "Random.h"
#include <memory>

/*
* template<typename T>
* T 入力・出力・重みの型 intかdouble
* 
* このクラスの生成時またはclear関数を呼んだ状態では
* まず①～⑤の関数を順に実行すること
* そうしない場合その他の関数を呼び出さないこと
* 
* このクラスにおいてnullptrの入出力は一切なし
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
	* クラス生成時と同じ状態にする
	* 確保したメモリがあれば全て解放する
	*/
	void clear();

	/*
	* ①入力層の設定
	* 
	* バイアスノードあり
	* 
	* @param size ノード数 (バイアスノードを含まない)
	*/
	void setInputLayer(int size);

	/*
	* ②中間層の数の設定
	* 
	* @param num 中間層の数 １以上
	*/
	void setHiddenLayerNum(int num);

	/*
	* ③中間層の設定
	* 
	* ②で設定した回数だけこの関数を繰り返し呼ぶ
	* 呼ぶごとに入力層に近い方から順に設定する
	* バイアスノードあり
	* 
	* @param size   ノード数 (バイアスノードを含まない)
	* @param actFnc 活性化関数のID
	*/
	void setHiddenLayer(int size, ActFncID actFncID);

	/*
	* ④出力層の設定
	* 
	* バイアスノードなし
	* 重みのサイズはここで計算され、重みが生成される
	*
	* @param size   ノード数
	* @param actFnc 活性化関数のID
	*/
	void setOutputLayer(int size, ActFncID actFncID);

	/*
	* ⑤重みの設定
	* 
	* これか⑤'のどちらかを呼ぶ
	* 
	* @param weight 重みの配列 サイズ = getWeightSize関数
	*/
	void setWeight(const T* weight);

	/*
	* ⑤'重みをランダムに設定
	* 
	* これか⑤のどちらかを呼ぶ
	* 
	* @param min ランダムの最小値 (含む)
	* @param max ランダムの最大値 (含む)
	*/
	void setWeightRandom(T min, T max);

	/*
	* 順伝播して入力から出力を得る
	* 
	* @param input 入力配列 サイズ = getInputLayerSize関数
	* @return 出力配列 サイズ = getOutputLayerSize関数
	*/
	const T* forwardPropagation(const T* input);

	/*
	* 誤差逆伝播して重みを調整する
	* 
	* @param input  順伝播における入力配列
	* @param output 入力に対して望む出力配列
	*/
	void backpropagation(const T* input, const T* output);

	// @return 入力層のノード数
	int getInputLayerSize() const;

	// @return 中間層の数
	int getHiddenLayerNum() const;

	// @return index番目(0-based)の中間層のノード数
	int getHiddenLayerSize(int index) const;

	// @return index番目(0-based)の中間層の活性化関数
	ActFncID getHiddenLayerActFncID(int index) const;

	// @return 出力層のノード数
	int getOutputLayerSize() const;

	// @return 出力層の活性化関数
	ActFncID getOutputLayerActFncID() const;

	/*
	* ①～④を設定して初めて働く
	* 
	* @return 重みのサイズ
	*/
	int getWeightSize() const;

	// @return 重みの配列
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

	// 入力層と中間層の重みサイズ
	m_weightSize = (m_inputLayer.size + 1) * m_hiddenLayer[0].size;

	// 中間層同士の重みサイズ
	for (int i = 0; i < m_hiddenLayerNum - 1; ++i)
		m_weightSize += (m_hiddenLayer[i].size + 1) * m_hiddenLayer[i + 1].size;
	
	// 中間層と出力層の重みサイズ
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

	// 入力層と中間層
	for (int h = 0; h < m_hiddenLayer[0].size; ++h)
	{
		T sum = 0;
		for (int i = 0; i < m_inputLayer.size + 1; ++i)
		{
			sum += m_inputLayer.layer[i] * m_weight[weightIndex++];
		}
		m_hiddenLayer[0].layer[h] = (*m_hiddenLayer[0].actFnc)(sum);
	}

	// 中間層同士
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

	// 中間層と出力層
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

