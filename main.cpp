#include "Network.h"
#include "Matrix.h"
#include "utils.h"
#include <iostream>

int main()
{
    srand(time(nullptr));
    // Matrix trainingData{{{0, 0, 0}, // 0 OR 0 = 0, 0 OR 1 = 1, 1 OR 0 = 1, 1 OR 1 = 1
    //                      {0, 1, 1},
    //                      {1, 0, 1},
    //                      {1, 1, 0}}};

    Matrix trainingData{{{0.0000, 0.0000},   // x = 0.0
                         {0.1000, 0.3090},   // x = PI/10
                         {0.1667, 0.5000},   // x = PI/6
                         {0.2500, 0.7071},   // x = PI/4
                         {0.3333, 0.8660},   // x = PI/3
                         {0.5000, 1.0000},   // x = PI/2
                         {0.6667, 0.8660},   // x = 2*PI/3
                         {0.7500, 0.7071},   // x = 3*PI/4
                         {0.8333, 0.5000},   // x = 5*PI/6
                         {0.9000, 0.3090},   // x = 9*PI/10
                         {1.0000, 0.0000}}}; // x = PI

    Network testNet{trainingData, 1, 1};
    testNet.addLayers({6, 1});
    testNet.train(50000, 1e-5, 1e-1);

    Matrix trainingInput{{{0},
                          {0.1000},
                          {0.1667},
                          {0.2500}}};

    testNet.feedForward(trainingInput);
}
