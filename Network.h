#pragma once
#include "Matrix.h"
#include "utils.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <sstream>

class Network
{
public:
    Network(const Matrix &trainingData, uint32_t inputCols, uint32_t outputCols);

    uint32_t getDataSetSize() const;

    uint32_t getLayerCount() const;

    uint32_t getLayerNeurons(uint32_t layer) const;

    void addLayers(const std::vector<uint32_t> &neuronCounts);

    void addLayer(uint32_t neurons);

    void feedForwardTo(const Matrix &data, uint32_t layerIndex);

    double computeCost();

    void train(uint32_t epochs, double eps, double rate);

    void feedForward(const Matrix &inputData);

    Matrix predict(const Matrix &inputRow) const;

    void feedForwardRow(const Matrix &data);

private:
    std::vector<Matrix> inputs_;

    std::vector<Matrix> expectedOutputs_;

    std::vector<Matrix> weights_;

    std::vector<Matrix> biases_;

    uint32_t inputCols_;

    uint32_t outputCols_;
};
