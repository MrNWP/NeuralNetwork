#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>

template<typename T>
void string_explode(const std::basic_string<T>& s, T c, std::vector<std::basic_string<T> >& v){
    //explodes the string s on the char c into the vector<string> v
    //example usage:
    //std::string s("23,45,18,92,54");
    //std::vector<std::string> v;
    //bsc_explode(s, ',', v); ~note the single quotes around the comma, not double quotes
    //also note that vector v will be filled with strings, not numeric values
    typename std::basic_string<T>::size_type i = 0;
    typename std::basic_string<T>::size_type j = s.find(c);
    while(j != std::basic_string<T>::npos){
        v.push_back(s.substr(i, j-i));
        i = ++j;
        j = s.find(c, j);
        if(j == std::basic_string<T>::npos) v.push_back(s.substr(i, s.length()));
    }
}

using namespace Eigen;

int main()
{
    NeuralNetwork myNet(784, 260, 10);
    //myNet.setDebug(1);
    MatrixXd targets;
    std::vector<std::string> lines;
    std::ifstream file("training.txt");
    std::string line;
    while ( std::getline(file, line) ) {
        if ( !line.empty() )
            lines.push_back(line);
    }
    file.close();
    for(int k = 0;k < 10; k++){
        int processedLines = 0;
        for(auto l : lines){
            //split the first item off to use for target
            auto posComma = l.find(",", 0);
            int iTarget = atoi((l.substr(0, posComma)).data());
            std::string t = l.substr(posComma+1,l.size());
            //load remaining items into matrix
            std::vector<std::string> vTargets;
            string_explode(t, ',', vTargets);
            MatrixXd mTrainingInputs(vTargets.size(), 1);
            uint iter = 0;
            for(auto g : vTargets){
                mTrainingInputs(iter, 0) = (atof(g.data())/255 * 0.99) + 0.01;
                ++iter;
            }
            //now load correct target matrix
            MatrixXd mTargets(10, 1);
            mTargets << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
            mTargets(iTarget, 0) = 0.99;
            //and finally, the actual training process
            myNet.train(mTrainingInputs, mTargets);
            ++processedLines;
            std::cout << "Finished training line " << processedLines << std::endl;
        }
    }
    //now let's test our network to see if training was successful
    std::vector<std::string> testlines;
    std::ifstream testfile("test.txt");
    std::string testline;
    while ( std::getline(testfile, testline) ) {
        if ( !testline.empty() )
            testlines.push_back(testline);
    }
    testfile.close();
    std::vector<int> results;
    for(int i = 0;i < 10;i++) results.push_back(i);
    for(auto l : testlines){
        auto posComma = l.find(",", 0);
        int iTarget = atoi((l.substr(0, posComma)).data());
        std::string t = l.substr(posComma+1,l.size());
        //load remaining items into matrix
        std::vector<std::string> vTargets;
        string_explode(t, ',', vTargets);
        MatrixXd mTestInputs(vTargets.size(), 1);
        uint iter = 0;
        for(auto g : vTargets){
            mTestInputs(iter, 0) = (atof(g.data())/255 * 0.99) + 0.01;
            ++iter;
        }
        MatrixXd mOutputs;
        myNet.query(mTestInputs, mOutputs);
        int result;
        double dMax = 0.0;
        for(int i = 0;i < 10;i++){
            if(mOutputs(i,0) > dMax){
                dMax = mOutputs(i,0);
                result = i;
            }
        }
        std::cout << "Target: " << iTarget << " Result: " << results[result] << std::endl;
    }
    return 0;
}
