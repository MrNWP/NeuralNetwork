#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>

using namespace Eigen;

NeuralNetwork::NeuralNetwork(){
    inodes = 3;
    hnodes = 3;
    onodes = 3;
    lrate = 0.3;
    mWeightsI2H = MatrixXd::Random(hnodes, inodes) * 0.5;
    mWeightsH2O = MatrixXd::Random(onodes, hnodes) * 0.5;
    debug = 0;
}

NeuralNetwork::NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes){
    inodes = inputnodes;
    hnodes = hiddennodes;
    onodes = outputnodes;
    lrate = 1/(sqrt(inodes));
    mWeightsI2H = MatrixXd::Random(hnodes, inodes) * 0.5;
    mWeightsH2O = MatrixXd::Random(onodes, hnodes) * 0.5;
    //the MatrixXd::Random() function fills the matrix with random values -1:1
    //and we want them to be -0.5:0.5 so we squash them to fit within that range
    debug = 0;//in the range 0:3
    //the higher the debug level the more info gets printed to console
    //for 2 or 3 the output should be piped to a log file instead of console
}

NeuralNetwork::~NeuralNetwork(){
    //put cleanup stuff here

}

void NeuralNetwork::setDebug(int d){
    debug = d;
}

void NeuralNetwork::displayWeights(){
    std::cout << "Here are the I2H weights: \n" << mWeightsI2H << std::endl;
    std::cout << "Here are the H2O weights: \n" << mWeightsH2O << std::endl;
}

void NeuralNetwork::train(MatrixXd &mInputs, MatrixXd &mTargets){
    MatrixXd mOutputs;
    query(mInputs, mOutputs);
    if(debug) std::cout << "mTargets:\n" << mTargets << std::endl;
    MatrixXd mErrors = mTargets - mOutputs;
    if(debug) std::cout << "mErrors:\n" << mErrors << std::endl;
    MatrixXd mHiddenErrors = mWeightsH2O.transpose() * mErrors;
    if(debug > 1) std::cout << "mHiddenErrors:\n" << mHiddenErrors << std::endl;
    MatrixXd temp = mOutputs.unaryExpr([](double x) { return x*(1.0 - x); });
    if(debug) std::cout << "temp H2O:\n" << temp << std::endl;
    MatrixXd temp2 = lrate * (mErrors.cwiseProduct(temp));
    if(debug) std::cout << "temp2 H2O:\n" << temp2 << std::endl;
    MatrixXd mDelta = temp2 * mHiddenOutputs.transpose();
    if(debug > 1) std::cout << "mDelta H2O:\nrows: " << mDelta.rows() << " cols: " << mDelta.cols() << std::endl;
    if(debug > 1) std::cout << mDelta << std::endl;
    mWeightsH2O += mDelta;
    temp = mHiddenOutputs.unaryExpr([](double x) { return x*(1.0 - x); });
    if(debug > 1) std::cout << "temp I2H:\n" << temp << std::endl;
    temp2 = lrate * (mHiddenErrors.cwiseProduct(temp));
    if(debug > 1) std::cout << "temp2 I2H:\n" << temp2 << std::endl;
    mDelta = temp2 * mInputs.transpose();
    if(debug > 1) std::cout << "mDelta I2H:\nrows: " << mDelta.rows() << " cols: " << mDelta.cols() << std::endl;
    if(debug > 1) std::cout << "mDelta I2H:\n" << mDelta << std::endl;
    mWeightsI2H += mDelta;
    if(debug > 2) displayWeights();
}

void NeuralNetwork::query(MatrixXd &mInputs, MatrixXd &mOutputs){
    MatrixXd mHiddenInputs = mWeightsI2H * mInputs;
    mHiddenOutputs = mHiddenInputs.unaryExpr([](double x) { return 1.0/(1.0+pow(2.71828,-x)); });
    MatrixXd mFinalInputs = mWeightsH2O * mHiddenOutputs;
    mOutputs = mFinalInputs.unaryExpr([](double x) { return 1.0/(1.0+pow(2.71828,-x)); });
    //the inputs are weighted, and then squashed with a standard sigmoid function
    if(debug) std::cout << "mOutputs:\n" << mOutputs << std::endl;
}
