#include "Network.h"

// GOAL: pass layers through weight and biases to create a network which eventually outputs the answer.

Network::Network(const Matrix &trainingData, uint32_t inputCols, uint32_t outputCols) : inputCols_(inputCols), outputCols_(outputCols)
{
    uint32_t dataRows = trainingData.getRowCount();
    uint32_t dataCols = trainingData.getColCount();
    if (dataCols != inputCols_ + outputCols_)
    {
        throw std::invalid_argument("Training data dimensions does not match!");
    }
    for (uint32_t k = 0; k < dataRows; ++k)
    {
        Matrix newInput{1, inputCols_};
        Matrix newOutput{1, outputCols_};
        for (uint32_t i = 0; i < inputCols_; ++i)
        {
            newInput(0, i) = trainingData(k, i);
        }
        for (uint32_t i = inputCols_; i < inputCols_ + outputCols_; ++i)
        {
            newOutput(0, i - inputCols_) = trainingData(k, i);
        }
        inputs_.push_back(std::move(newInput));
        expectedOutputs_.push_back(std::move(newOutput));
    }
}

uint32_t Network::getDataSetSize() const { return inputs_.size(); }

uint32_t Network::getLayerCount() const { return weights_.size(); }

uint32_t Network::getLayerNeurons(uint32_t layer) const
{
    if (layer >= getLayerCount())
    {
        throw std::out_of_range("Layer does not exist!");
    }
    return biases_[layer].getColCount();
}

void Network::addLayers(const std::vector<uint32_t> &neuronCounts)
{
    for (uint32_t i = 0; i < neuronCounts.size(); ++i)
    {
        addLayer(neuronCounts[i]);
    }
}

void Network::addLayer(uint32_t neuronCount)
{
    uint32_t layerCount = getLayerCount();
    uint32_t lastLayerNeurons = (layerCount == 0) ? inputCols_ : getLayerNeurons(layerCount - 1);
    weights_.push_back(utils::generateMatrix(lastLayerNeurons, neuronCount));
    biases_.push_back(utils::generateMatrix(1, neuronCount));
}

double Network::computeCost()
{
    double totalDiff = 0.0;
    uint32_t layerCount = getLayerCount();
    if (layerCount == 0)
    {
        throw std::invalid_argument("Cannot compute cost of a network with no layers!");
    }
    size_t dataSetSize = getDataSetSize();
    if (dataSetSize == 0)
    {
        return 0.0;
    }
    for (uint32_t dataIndex = 0; dataIndex < dataSetSize; ++dataIndex)
    {
        Matrix prediction = predict(inputs_[dataIndex]);
        Matrix diffMatrix = prediction - expectedOutputs_[dataIndex];
        for (uint32_t r = 0; r < diffMatrix.getRowCount(); ++r)
        {
            for (uint32_t c = 0; c < diffMatrix.getColCount(); ++c)
            {
                double diff = diffMatrix(r, c);
                totalDiff += diff * diff;
            }
        }
    }
    return totalDiff / dataSetSize;
}

void Network::train(uint32_t epochs, double eps, double rate)
{
    uint32_t layerCount = getLayerCount();
    if (layerCount == 0)
    {
        throw std::invalid_argument("Cannot train network with 0 layers");
    }
    for (uint32_t epoch = 0; epoch < epochs; ++epoch)
    {
        // --- 1. Create matrices to store the gradients ---
        std::vector<Matrix> weightGradients(layerCount);
        std::vector<Matrix> biasGradients(layerCount);
        for (uint32_t l = 0; l < layerCount; ++l)
        {
            weightGradients[l] = Matrix{weights_[l].getRowCount(), weights_[l].getColCount()};
            biasGradients[l] = Matrix{1, biases_[l].getColCount()};
        }

        // --- 1.5 GET ORIGINAL COST AT THE BEGINNING OF EPOCH ---
        double originalCost = computeCost();

        // --- 2. Calculate all gradients for each Weight and Bias ---
        for (uint32_t l = 0; l < layerCount; ++l)
        {
            for (uint32_t r = 0; r < weights_[l].getRowCount(); ++r)
            {
                for (uint32_t c = 0; c < weights_[l].getColCount(); ++c)
                {
                    double originalValue = weights_[l](r, c);
                    weights_[l](r, c) = originalValue + eps;
                    double newCost = computeCost();
                    weightGradients[l](r, c) = (newCost - originalCost) / eps;
                    weights_[l](r, c) = originalValue;
                }
            }
            for (uint32_t c = 0; c < biases_[l].getColCount(); ++c)
            {
                double originalValue = biases_[l](0, c);
                biases_[l](0, c) = originalValue + eps;
                double newCost = computeCost();
                biasGradients[l](0, c) = (newCost - originalCost) / eps;
                biases_[l](0, c) = originalValue;
            }
        }

        // --- 3. Apply all calculated gradients ---
        for (uint32_t l = 0; l < layerCount; ++l)
        {
            weights_[l] -= (weightGradients[l] * rate);
            biases_[l] -= (biasGradients[l] * rate);
        }
        if (epoch % 1000 == 0 || epoch == epochs - 1)
        {
            std::cout << "Epoch: " << epoch << " Cost: " << computeCost() << std::endl;
        }
    }
}

void Network::feedForward(const Matrix &inputData)
{
    if (inputData.getColCount() != inputCols_)
    {
        std::ostringstream oss;
        oss << "Input data has " << inputData.getColCount() << " columns, but network expects " << inputCols_;
        throw std::invalid_argument(oss.str());
    }
    Matrix dataRow{1, inputCols_};
    std::cout << "------------------------" << "\n";
    for (uint32_t r = 0; r < inputData.getRowCount(); ++r)
    {
        for (uint32_t c = 0; c < inputCols_; ++c)
        {
            dataRow(0, c) = inputData(r, c);
        }
        Matrix outputRow = predict(dataRow);
        for (uint32_t c = 0; c < inputCols_; ++c)
        {
            std::cout << dataRow(0, c) << " ";
        }
        for (uint32_t c = 0; c < outputCols_; ++c)
        {
            std::cout << "| " << outputRow(0, c) << " ";
        }
        std::cout << '\n';
    }
    std::cout << "------------------------" << std::endl;
}

Matrix Network::predict(const Matrix &inputRow) const
{
    if (inputRow.getColCount() != inputCols_ || inputRow.getRowCount() != 1)
    {
        std::ostringstream oss;
        oss << "Input for predict must be a 1x" << inputCols_ << " matrix, but got 1x" << inputRow.getColCount();
        throw std::invalid_argument(oss.str());
    }
    uint32_t layerCount = getLayerCount();
    if (layerCount == 0)
    {
        throw std::invalid_argument("Cannot predict: network has 0 layers.");
    }
    Matrix currentActivations = inputRow;
    for (uint32_t i = 0; i < layerCount - 1; ++i)
    {
        currentActivations = (currentActivations * weights_[i]) + biases_[i];
        // Change activation function for hidden layers here.
        currentActivations.apply(utils::relu);
    }
    currentActivations = (currentActivations * weights_[layerCount - 1]) + biases_[layerCount - 1];
    // Change activation function for output layer here.
    currentActivations.apply(utils::sigmoid);
    return currentActivations;
}
