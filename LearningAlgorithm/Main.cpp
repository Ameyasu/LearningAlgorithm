#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"
#include "ActFncOperator.h"
#include "LAFileIO.h"

#include <iostream>
#include <iomanip>

template<typename T>
void printNeuralNetwork(const NeuralNetwork<T>& nn)
{
	std::cout << "入力層のノード数 = " << nn.getInputLayerSize() << std::endl;
	std::cout << "中間層の数 = " << nn.getHiddenLayerNum() << std::endl;
	for (int i = 0; i < nn.getHiddenLayerNum(); ++i)
	{
		std::cout << "中間層(" << i << "番目)のノード数 = " << nn.getHiddenLayerSize(i) << std::endl;
		std::cout << "中間層(" << i << "番目)の活性化関数 = " << ActFncOperator::toString(nn.getHiddenLayerActFncID(i)) << std::endl;
	}
	std::cout << "出力層のノード数 = " << nn.getOutputLayerSize() << std::endl;
	std::cout << "出力層の活性化関数 = " << ActFncOperator::toString(nn.getOutputLayerActFncID()) << std::endl;
	std::cout << "重みのサイズ = " << nn.getWeightSize() << std::endl;
	std::cout << "重み = {";
	for (int i = 0; i < nn.getWeightSize(); ++i)
		std::cout << nn.getWeight()[i] << ", ";
	std::cout << "\b\b}" << std::endl;
}

template<typename Gene, typename Fitness>
void printGeneticAlgorithm(const GeneticAlgorithm<Gene, Fitness>& ga)
{
	std::cout << "世代 = " << ga.getGeneration() << std::endl;
	std::cout << "人口 = " << ga.getPopulation() << std::endl;
	std::cout << "染色体の長さ = " << ga.getChromosomeLength() << std::endl;
	std::cout << "染色体の最小値 (含む) = " << ga.getChromosomeValueMin() << std::endl;
	std::cout << "染色体の最大値 (含む) = " << ga.getChromosomeValueMax() << std::endl;
	std::cout << "エリート数 = " << ga.getEliteNum() << std::endl;
	std::cout << "個体 = {" << std::endl;
	for (int i = 0; i < ga.getPopulation(); ++i)
	{
		std::cout << "  index(" << std::setw(2) << i << ") = {";
		for (int j = 0; j < ga.getChromosomeLength(); ++j)
			std::cout << std::setw(2) << ga.getIndividual(i)[j] << ", ";
		std::cout << "\b\b}" << std::endl;
	}
	std::cout << "}" << std::endl;
}

int main(void)
{
	/*
	* 以下の構造のニューラルネットワークを生成する
	* 
	* ○…ノード　●…バイアスノード
	* 入力層　中間層　出力層
	* 　○　　　○　　　○
	* 　○　　　○
	* 　●　　　●
	* 　　　　 ReLU 　 Step
	*/
	NeuralNetwork<int> nn;
	nn.setInputLayer(2);
	nn.setHiddenLayerNum(1);
	nn.setHiddenLayer(2, ActFncID::RELU);
	nn.setOutputLayer(1, ActFncID::STEP);

	/*
	* その他構造の作り方の例
	*
	* ○…ノード　●…バイアスノード
	* 入力層　中間層　中間層　中間層　出力層
	* 　○　　　○　　　○　　　○　　　○
	* 　○　　　○　　　○　　　●　　　○
	* 　○　　　●　　　○
	* 　●　　　　　　　●
	* 　　　　 ReLU 　 Step 　Sigmoid　Identity
	*/
	//NeuralNetwork<double> nn2;
	//nn2.setInputLayer(3);
	//nn2.setHiddenLayerNum(3);
	//nn2.setHiddenLayer(2, ActFncID::RELU);
	//nn2.setHiddenLayer(3, ActFncID::STEP);
	//nn2.setHiddenLayer(1, ActFncID::SIGMOID);
	//nn2.setOutputLayer(2, ActFncID::IDENTITY);

	// NN初期情報表示
	std::cout << "+===+===+===+ NN初期化 +===+===+===+" << std::endl;
	printNeuralNetwork(nn);

	// 遺伝的アルゴリズムクラス生成
	static constexpr int POPULATION = 20;
	static constexpr int CHRMSM_MIN = -9;
	static constexpr int CHRMSM_MAX = 9;
	static constexpr int ELITE_NUM = 1;
	GeneticAlgorithm<int, int> ga;
	ga.reset(POPULATION, nn.getWeightSize(), CHRMSM_MIN, CHRMSM_MAX, ELITE_NUM);
	ga.setIndividualsRandom(ga.getChromosomeValueMin(), ga.getChromosomeValueMax());
	
	// GA初期情報表示
	std::cout << std::endl << "+===+===+===+ GA初期化 +===+===+===+" << std::endl;
	printGeneticAlgorithm(ga);

	// 最適化開始前にキー入力でストップ
	{
		std::cout << "GAでNNの重みの最適化開始" << std::endl;
		std::cout << "プログラムを続行するには何かキーを入力... >";
		std::string s;
		std::cin >> s;
	}

	std::cout << std::endl << "+===+===+===+ 最適化 +===+===+===+" << std::endl;
	/*
	* NNでXOR回路を表現するように最適化をする
	* 入力(0, 0) → 出力(0)
	* 入力(0, 1) → 出力(1)
	* 入力(1, 0) → 出力(1)
	* 入力(1, 1) → 出力(0)
	*/
	int input[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	int idealOutput[] = {0, 1, 1, 0};

	// 全個体それぞれの適応度
	int fitness[POPULATION] = {};

	while (true)
	{
		// GAの全個体でNNの入出力を確認し、適応度を計算する
		for (int pop = 0; pop < POPULATION; ++pop)
		{
			// pop番目の個体の染色体をNNの重みに設定
			nn.setWeight(ga.getIndividual(pop));
			fitness[pop] = 0;
			for (int i = 0; i < 4; ++i)
			{
				// NN順伝播
				const int* output = nn.forwardPropagation(input[i]);

				// 誤差が多いほど適応度が低くなるように計算
				fitness[pop] += -std::abs(output[0] - idealOutput[i]);
			}
		}

		// 計算した適応度を設定
		ga.evaluate(fitness);

		// 適応度を元に次世代を生成する
		ga.generateNextGeneration();

		// 全個体の染色体を表示
		std::cout << "個体 = {" << std::endl;
		for (int i = 0; i < ga.getPopulation(); ++i)
		{
			std::cout << "  index(" << std::setw(2) << i << ") = {";
			for (int j = 0; j < ga.getChromosomeLength(); ++j)
				std::cout << std::setw(2) << ga.getIndividual(i)[j] << ", ";
			std::cout << "\b\b}" << std::endl;
		}
		std::cout << "}" << std::endl;

		// 以下、エリート個体が最適化されたか確認する

		// エリート個体をNNにセット
		nn.setWeight(ga.getIndividual(0));

		// 誤差の計算
		int e = 0;
		for (int i = 0; i < 4; ++i)
		{
			// NN順伝播
			const int* output = nn.forwardPropagation(input[i]);

			// 誤差を計算
			int ee = output[0] - idealOutput[i];
			e += std::abs(ee);

			// 入出力と誤差を表示
			std::cout << "(" << input[i][0] << ", " << input[i][1] << ") = " << output[0] << ", 誤差 = ";
			if (ee == 0)
				std::cout << " 0" << std::endl;
			else
				std::cout << std::showpos << ee << std::noshowpos << std::endl;
		}
		std::cout << "誤差の絶対値の合計 = " << e << std::endl;

		// 誤差が0になれば終了
		if (e == 0)
		{
			std::cout << std::endl << "+===+===+===+ 最適化完了 +===+===+===+" << std::endl;
			std::cout << "世代 = " << ga.getGeneration() << std::endl;
			break;
		}

		// 100世代進むごとにプログラムの続行を尋ねる
		if (ga.getGeneration() % 100 == 0)
		{
			std::cout << std::endl << "+===+===+===+ プログラムを続行しますか？ +===+===+===+" << std::endl;
			std::cout << "世代 = " << ga.getGeneration() << std::endl;
			std::cout << "はい = y, いいえ = n >";
			std::string s;
			std::cin >> s;
			if (s == "n")
				break;
		}
	}

	// NeuralNetworkクラスをファイル入出力
	{
		LAFileIO::outputNeuralNetwork("dataNN.dat", nn);

		NeuralNetwork<int> nn2;
		LAFileIO::inputNeuralNetwork("dataNN.dat", nn2);

		std::cout << std::endl << "+===+===+===+ ファイルからNN取得 +===+===+===+" << std::endl;
		printNeuralNetwork(nn2);
	}

	// GeneticAlgorithmクラスをファイル入出力
	{
		LAFileIO::outputGeneticAlgorithm("dataGA.dat", ga);

		GeneticAlgorithm<int, int> ga2;
		LAFileIO::inputGeneticAlgorithm("dataGA.dat", ga2);

		std::cout << std::endl << "+===+===+===+ ファイルからGA取得 +===+===+===+" << std::endl;
		printGeneticAlgorithm(ga2);
	}

	return 0;
}