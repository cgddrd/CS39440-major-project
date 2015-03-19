//
//  Utils.cpp
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 06/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include "Utils.h"

using namespace std;
using namespace cv;

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
    
    double sum = accumulate(values.begin(), values.end(), 0.0);
    
    return sum / values.size();
    
}

// CG - Arrowed Line drawing method. (PhilLab - http://stackoverflow.com/questions/10161351) (Built into OpenCV v3.0.0)
void Utils::arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, int shift, double tipLength)
{
    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
    line(img, pt1, pt2, color, thickness, line_type, shift);
    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
            cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}