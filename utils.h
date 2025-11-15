#pragma once

#include <cstdlib>
#include <cmath>

namespace utils
{
    inline double randd()
    {
        return ((double)rand() / (double)RAND_MAX);
    }
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline Matrix generateMatrix(uint32_t rows, uint32_t cols)
    {
        Matrix resultMatrix = Matrix{rows, cols};
        for (uint32_t r = 0; r < rows; ++r)
        {
            for (uint32_t c = 0; c < cols; ++c)
            {
                resultMatrix(r, c) = randd() * 2.0 - 1.0;
            }
        }
        return resultMatrix;
    }
    inline double relu(double x) { return (x > 0.0 ? x : 0.0); }
}
