#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED

#include "Eigen/Dense"

using namespace Eigen;

class NeuralNetwork {
    private:
        int debug;
        int inodes;
        int hnodes;
        int onodes;
        double lrate;//learning rate
        MatrixXd mWeightsI2H;//matrix of weights between input and hidden layers
        MatrixXd mWeightsH2O;//matrix of weights between hidden and output layers
        MatrixXd mHiddenOutputs;

    public:
        NeuralNetwork();//default constructor creates a 3x3 network with 0.3 lr
        NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes);
        ~NeuralNetwork();//destructor
        void displayWeights();
        void train(MatrixXd &mInputs, MatrixXd &mTargets);
        void query(MatrixXd &mInputs, MatrixXd &mOutputs);
        void setDebug(int d);
};

#endif // NEURALNETWORK_H_INCLUDED
