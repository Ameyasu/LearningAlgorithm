#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"
#include "ActFncOperator.h"
#include "LAFileIO.h"

#include <iostream>
#include <iomanip>

template<typename T>
void printNeuralNetwork(const NeuralNetwork<T>& nn)
{
	std::cout << "���͑w�̃m�[�h�� = " << nn.getInputLayerSize() << std::endl;
	std::cout << "���ԑw�̐� = " << nn.getHiddenLayerNum() << std::endl;
	for (int i = 0; i < nn.getHiddenLayerNum(); ++i)
	{
		std::cout << "���ԑw(" << i << "�Ԗ�)�̃m�[�h�� = " << nn.getHiddenLayerSize(i) << std::endl;
		std::cout << "���ԑw(" << i << "�Ԗ�)�̊������֐� = " << ActFncOperator::toString(nn.getHiddenLayerActFncID(i)) << std::endl;
	}
	std::cout << "�o�͑w�̃m�[�h�� = " << nn.getOutputLayerSize() << std::endl;
	std::cout << "�o�͑w�̊������֐� = " << ActFncOperator::toString(nn.getOutputLayerActFncID()) << std::endl;
	std::cout << "�d�݂̃T�C�Y = " << nn.getWeightSize() << std::endl;
	std::cout << "�d�� = {";
	for (int i = 0; i < nn.getWeightSize(); ++i)
		std::cout << nn.getWeight()[i] << ", ";
	std::cout << "\b\b}" << std::endl;
}

template<typename Gene, typename Fitness>
void printGeneticAlgorithm(const GeneticAlgorithm<Gene, Fitness>& ga)
{
	std::cout << "���� = " << ga.getGeneration() << std::endl;
	std::cout << "�l�� = " << ga.getPopulation() << std::endl;
	std::cout << "���F�̂̒��� = " << ga.getChromosomeLength() << std::endl;
	std::cout << "���F�̂̍ŏ��l (�܂�) = " << ga.getChromosomeValueMin() << std::endl;
	std::cout << "���F�̂̍ő�l (�܂�) = " << ga.getChromosomeValueMax() << std::endl;
	std::cout << "�G���[�g�� = " << ga.getEliteNum() << std::endl;
	std::cout << "�� = {" << std::endl;
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
	* �ȉ��̍\���̃j���[�����l�b�g���[�N�𐶐�����
	* 
	* ���c�m�[�h�@���c�o�C�A�X�m�[�h
	* ���͑w�@���ԑw�@�o�͑w
	* �@���@�@�@���@�@�@��
	* �@���@�@�@��
	* �@���@�@�@��
	* �@�@�@�@ ReLU �@ Step
	*/
	NeuralNetwork<int> nn;
	nn.setInputLayer(2);
	nn.setHiddenLayerNum(1);
	nn.setHiddenLayer(2, ActFncID::RELU);
	nn.setOutputLayer(1, ActFncID::STEP);

	/*
	* ���̑��\���̍����̗�
	*
	* ���c�m�[�h�@���c�o�C�A�X�m�[�h
	* ���͑w�@���ԑw�@���ԑw�@���ԑw�@�o�͑w
	* �@���@�@�@���@�@�@���@�@�@���@�@�@��
	* �@���@�@�@���@�@�@���@�@�@���@�@�@��
	* �@���@�@�@���@�@�@��
	* �@���@�@�@�@�@�@�@��
	* �@�@�@�@ ReLU �@ Step �@Sigmoid�@Identity
	*/
	//NeuralNetwork<double> nn2;
	//nn2.setInputLayer(3);
	//nn2.setHiddenLayerNum(3);
	//nn2.setHiddenLayer(2, ActFncID::RELU);
	//nn2.setHiddenLayer(3, ActFncID::STEP);
	//nn2.setHiddenLayer(1, ActFncID::SIGMOID);
	//nn2.setOutputLayer(2, ActFncID::IDENTITY);

	// NN�������\��
	std::cout << "+===+===+===+ NN������ +===+===+===+" << std::endl;
	printNeuralNetwork(nn);

	// ��`�I�A���S���Y���N���X����
	static constexpr int POPULATION = 20;
	static constexpr int CHRMSM_MIN = -9;
	static constexpr int CHRMSM_MAX = 9;
	static constexpr int ELITE_NUM = 1;
	GeneticAlgorithm<int, int> ga;
	ga.reset(POPULATION, nn.getWeightSize(), CHRMSM_MIN, CHRMSM_MAX, ELITE_NUM);
	ga.setIndividualsRandom(ga.getChromosomeValueMin(), ga.getChromosomeValueMax());
	
	// GA�������\��
	std::cout << std::endl << "+===+===+===+ GA������ +===+===+===+" << std::endl;
	printGeneticAlgorithm(ga);

	// �œK���J�n�O�ɃL�[���͂ŃX�g�b�v
	{
		std::cout << "GA��NN�̏d�݂̍œK���J�n" << std::endl;
		std::cout << "�v���O�����𑱍s����ɂ͉����L�[�����... >";
		std::string s;
		std::cin >> s;
	}

	std::cout << std::endl << "+===+===+===+ �œK�� +===+===+===+" << std::endl;
	/*
	* NN��XOR��H��\������悤�ɍœK��������
	* ����(0, 0) �� �o��(0)
	* ����(0, 1) �� �o��(1)
	* ����(1, 0) �� �o��(1)
	* ����(1, 1) �� �o��(0)
	*/
	int input[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	int idealOutput[] = {0, 1, 1, 0};

	// �S�̂��ꂼ��̓K���x
	int fitness[POPULATION] = {};

	while (true)
	{
		// GA�̑S�̂�NN�̓��o�͂��m�F���A�K���x���v�Z����
		for (int pop = 0; pop < POPULATION; ++pop)
		{
			// pop�Ԗڂ̌̂̐��F�̂�NN�̏d�݂ɐݒ�
			nn.setWeight(ga.getIndividual(pop));
			fitness[pop] = 0;
			for (int i = 0; i < 4; ++i)
			{
				// NN���`�d
				const int* output = nn.forwardPropagation(input[i]);

				// �덷�������قǓK���x���Ⴍ�Ȃ�悤�Ɍv�Z
				fitness[pop] += -std::abs(output[0] - idealOutput[i]);
			}
		}

		// �v�Z�����K���x��ݒ�
		ga.evaluate(fitness);

		// �K���x�����Ɏ�����𐶐�����
		ga.generateNextGeneration();

		// �S�̂̐��F�̂�\��
		std::cout << "�� = {" << std::endl;
		for (int i = 0; i < ga.getPopulation(); ++i)
		{
			std::cout << "  index(" << std::setw(2) << i << ") = {";
			for (int j = 0; j < ga.getChromosomeLength(); ++j)
				std::cout << std::setw(2) << ga.getIndividual(i)[j] << ", ";
			std::cout << "\b\b}" << std::endl;
		}
		std::cout << "}" << std::endl;

		// �ȉ��A�G���[�g�̂��œK�����ꂽ���m�F����

		// �G���[�g�̂�NN�ɃZ�b�g
		nn.setWeight(ga.getIndividual(0));

		// �덷�̌v�Z
		int e = 0;
		for (int i = 0; i < 4; ++i)
		{
			// NN���`�d
			const int* output = nn.forwardPropagation(input[i]);

			// �덷���v�Z
			int ee = output[0] - idealOutput[i];
			e += std::abs(ee);

			// ���o�͂ƌ덷��\��
			std::cout << "(" << input[i][0] << ", " << input[i][1] << ") = " << output[0] << ", �덷 = ";
			if (ee == 0)
				std::cout << " 0" << std::endl;
			else
				std::cout << std::showpos << ee << std::noshowpos << std::endl;
		}
		std::cout << "�덷�̐�Βl�̍��v = " << e << std::endl;

		// �덷��0�ɂȂ�ΏI��
		if (e == 0)
		{
			std::cout << std::endl << "+===+===+===+ �œK������ +===+===+===+" << std::endl;
			std::cout << "���� = " << ga.getGeneration() << std::endl;
			break;
		}

		// 100����i�ނ��ƂɃv���O�����̑��s��q�˂�
		if (ga.getGeneration() % 100 == 0)
		{
			std::cout << std::endl << "+===+===+===+ �v���O�����𑱍s���܂����H +===+===+===+" << std::endl;
			std::cout << "���� = " << ga.getGeneration() << std::endl;
			std::cout << "�͂� = y, ������ = n >";
			std::string s;
			std::cin >> s;
			if (s == "n")
				break;
		}
	}

	// NeuralNetwork�N���X���t�@�C�����o��
	{
		LAFileIO::outputNeuralNetwork("dataNN.dat", nn);

		NeuralNetwork<int> nn2;
		LAFileIO::inputNeuralNetwork("dataNN.dat", nn2);

		std::cout << std::endl << "+===+===+===+ �t�@�C������NN�擾 +===+===+===+" << std::endl;
		printNeuralNetwork(nn2);
	}

	// GeneticAlgorithm�N���X���t�@�C�����o��
	{
		LAFileIO::outputGeneticAlgorithm("dataGA.dat", ga);

		GeneticAlgorithm<int, int> ga2;
		LAFileIO::inputGeneticAlgorithm("dataGA.dat", ga2);

		std::cout << std::endl << "+===+===+===+ �t�@�C������GA�擾 +===+===+===+" << std::endl;
		printGeneticAlgorithm(ga2);
	}

	return 0;
}