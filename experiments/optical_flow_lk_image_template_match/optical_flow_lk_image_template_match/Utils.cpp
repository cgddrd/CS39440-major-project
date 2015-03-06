//
//  Utils.cpp
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 06/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include "Utils.h"

using namespace std;

Utils::Utils() {}

vector<double> Utils::filterOutliers(vector<double> inValues) {
    
    double mean = calcMean(inValues);
    double sdDev = calcStandardDeviation(inValues);
    
    double positiveThreshold = mean + (1 * sdDev);
    double negativeThreshold = mean - (1 * sdDev);
    
    vector <double> filteredResults;
    
    for(int i = 0; i < inValues.size(); i++){
        if (negativeThreshold <= inValues[i] && positiveThreshold >= inValues[i]) {
            
            filteredResults.push_back(inValues[i]);
        }
    }
    
    return filteredResults;
    
}

double Utils::calcStandardDeviation(vector<double> values) {
    
    // Standard deviation is simply the SQRT of the variance.
    return sqrt(calcVariance(values));
}

double Utils::calcVariance(vector<double> values) {
    
    
    // 1) Calculate the mean of the dataset.
    double mean = calcMean(values);
    
    double dev = 0.0, sdev = 0.0;
    
    // 2) For each value, calculate the difference from the mean, and square it.
    // 3) Add all of these squared differences together.
    for(int i = 0; i < values.size(); i++){
        dev = (values[i] - mean) * (values[i] - mean);
        sdev += dev;
    }
    
    // 4) Divide the sum of the squared differences by the size of the dataset. This value is the variance of the dataset.
    return sdev / (values.size());
}

double Utils::calcMean(vector<double> values) {
    
//    ofstream myfile;
//
//    myfile.open ("data.csv", ios::out | ios::trunc);
//    
//    for(int i = 0; i < values.size(); i++){
//        myfile << values[i] << "\n";
//    }
//    
//    myfile.close();
    
    double sum = accumulate(values.begin(), values.end(), 0.0);
    
    return sum / values.size();
    
}