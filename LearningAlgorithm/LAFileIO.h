#pragma once

#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"
#include "ActivationFunction.h"
#include <string>
#include <fstream>
#include <memory>

class LAFileIO
{
public:
	template<typename T>
	static bool inputNeuralNetwork(std::string path, NeuralNetwork<T>& nn);

	template<typename T>
	static bool outputNeuralNetwork(std::string path, const NeuralNetwork<T>& nn);

	template<typename Gene, typename Fitness>
	static bool inputGeneticAlgorithm(std::string path, GeneticAlgorithm<Gene, Fitness>& ga);

	template<typename Gene, typename Fitness>
	static bool outputGeneticAlgorithm(std::string path, const GeneticAlgorithm<Gene, Fitness>& ga);

private:
	LAFileIO() = delete;
};




template<typename T>
bool LAFileIO::inputNeuralNetwork(std::string path, NeuralNetwork<T>& nn)
{
	std::ifstream ifs(path, std::ios::in | std::ios::binary);
	if (!ifs)
		return false;

	nn.clear();

	int inputLayerSize = 0;
	int hiddenLayerNum = 0;

	ifs.read(reinterpret_cast<char*>(&inputLayerSize), sizeof(inputLayerSize));
	ifs.read(reinterpret_cast<char*>(&hiddenLayerNum), sizeof(hiddenLayerNum));

	std::unique_ptr<int[]> hiddenLayerSize(new int[hiddenLayerNum]);
	std::unique_ptr<ActFncID[]> hiddenLayerActFncID(new ActFncID[hiddenLayerNum]);
	ifs.read(reinterpret_cast<char*>(hiddenLayerSize.get()), sizeof(hiddenLayerSize[0]) * hiddenLayerNum);
	ifs.read(reinterpret_cast<char*>(hiddenLayerActFncID.get()), sizeof(hiddenLayerActFncID[0]) * hiddenLayerNum);

	int outputLayerSize = 0;
	ActFncID outputLayerActFncID = ActFncID::IDENTITY;
	ifs.read(reinterpret_cast<char*>(&outputLayerSize), sizeof(outputLayerSize));
	ifs.read(reinterpret_cast<char*>(&outputLayerActFncID), sizeof(outputLayerActFncID));

	nn.setInputLayer(inputLayerSize);
	nn.setHiddenLayerNum(hiddenLayerNum);
	for (int i = 0; i < hiddenLayerNum; ++i)
		nn.setHiddenLayer(hiddenLayerSize[i], hiddenLayerActFncID[i]);
	nn.setOutputLayer(outputLayerSize, outputLayerActFncID);

	ifs.read(const_cast<char*>(reinterpret_cast<const char*>(nn.getWeight())), sizeof(nn.getWeight()[0]) * nn.getWeightSize());

	return true;
}

template<typename T>
bool LAFileIO::outputNeuralNetwork(std::string path, const NeuralNetwork<T>& nn)
{
	std::ofstream ofs(path, std::ios::out | std::ios::binary);
	if (!ofs)
		return false;

	int inputLayerSize = nn.getInputLayerSize();
	int hiddenLayerNum = nn.getHiddenLayerNum();
	std::unique_ptr<int[]> hiddenLayerSize(new int[hiddenLayerNum]);
	std::unique_ptr<ActFncID[]> hiddenLayerActFncID(new ActFncID[hiddenLayerNum]);
	for (int i = 0; i < hiddenLayerNum; ++i)
	{
		hiddenLayerSize[i] = nn.getHiddenLayerSize(i);
		hiddenLayerActFncID[i] = nn.getHiddenLayerActFncID(i);
	}
	int outputLayerSize = nn.getOutputLayerSize();
	ActFncID outputLayerActFncID = nn.getOutputLayerActFncID();

	ofs.write(reinterpret_cast<const char*>(&inputLayerSize), sizeof(inputLayerSize));
	ofs.write(reinterpret_cast<const char*>(&hiddenLayerNum), sizeof(hiddenLayerNum));
	ofs.write(reinterpret_cast<const char*>(hiddenLayerSize.get()), sizeof(hiddenLayerSize[0]) * hiddenLayerNum);
	ofs.write(reinterpret_cast<const char*>(hiddenLayerActFncID.get()), sizeof(hiddenLayerActFncID[0]) * hiddenLayerNum);
	ofs.write(reinterpret_cast<const char*>(&outputLayerSize), sizeof(outputLayerSize));
	ofs.write(reinterpret_cast<const char*>(&outputLayerActFncID), sizeof(outputLayerActFncID));
	ofs.write(reinterpret_cast<const char*>(nn.getWeight()), sizeof(nn.getWeight()[0]) * nn.getWeightSize());

	return true;
}

template<typename Gene, typename Fitness>
inline bool LAFileIO::inputGeneticAlgorithm(std::string path, GeneticAlgorithm<Gene, Fitness>& ga)
{
	std::ifstream ifs(path, std::ios::in | std::ios::binary);
	if (!ifs)
		return false;

	ga.clear();

	int generation = 0;
	int population = 0;
	int chromosomeLength = 0;
	Gene chromosomeValueMin = 0;
	Gene chromosomeValueMax = 0;
	int eliteNum = 0;

	ifs.read(reinterpret_cast<char*>(&generation), sizeof(generation));
	ifs.read(reinterpret_cast<char*>(&population), sizeof(population));
	ifs.read(reinterpret_cast<char*>(&chromosomeLength), sizeof(chromosomeLength));
	ifs.read(reinterpret_cast<char*>(&chromosomeValueMin), sizeof(chromosomeValueMin));
	ifs.read(reinterpret_cast<char*>(&chromosomeValueMax), sizeof(chromosomeValueMax));
	ifs.read(reinterpret_cast<char*>(&eliteNum), sizeof(eliteNum));

	ga.reset(population, chromosomeLength, chromosomeValueMin, chromosomeValueMax, eliteNum, generation);

	ifs.read(const_cast<char*>(reinterpret_cast<const char*>(ga.getIndividuals())), sizeof(ga.getIndividuals()[0]) * population * chromosomeLength);
	ifs.read(const_cast<char*>(reinterpret_cast<const char*>(ga.getFitnesses())), sizeof(ga.getFitnesses()[0]) * population);

	return true;
}

template<typename Gene, typename Fitness>
inline bool LAFileIO::outputGeneticAlgorithm(std::string path, const GeneticAlgorithm<Gene, Fitness>& ga)
{
	std::ofstream ofs(path, std::ios::out | std::ios::binary);
	if (!ofs)
		return false;

	int generation = ga.getGeneration();
	int population = ga.getPopulation();
	int chromosomeLength = ga.getChromosomeLength();
	Gene chromosomeValueMin = ga.getChromosomeValueMin();
	Gene chromosomeValueMax = ga.getChromosomeValueMax();
	int eliteNum = ga.getEliteNum();

	ofs.write(reinterpret_cast<const char*>(&generation), sizeof(generation));
	ofs.write(reinterpret_cast<const char*>(&population), sizeof(population));
	ofs.write(reinterpret_cast<const char*>(&chromosomeLength), sizeof(chromosomeLength));
	ofs.write(reinterpret_cast<const char*>(&chromosomeValueMin), sizeof(chromosomeValueMin));
	ofs.write(reinterpret_cast<const char*>(&chromosomeValueMax), sizeof(chromosomeValueMax));
	ofs.write(reinterpret_cast<const char*>(&eliteNum), sizeof(eliteNum));
	ofs.write(reinterpret_cast<const char*>(ga.getIndividuals()), sizeof(ga.getIndividuals()[0]) * population * chromosomeLength);
	ofs.write(reinterpret_cast<const char*>(ga.getFitnesses()), sizeof(ga.getFitnesses()[0]) * population);

	return true;
}
