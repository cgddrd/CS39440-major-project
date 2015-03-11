//
//  Utils.h
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 06/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#ifndef __optical_flow_lk_image_template_match__Utils__
#define __optical_flow_lk_image_template_match__Utils__

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

class Utils {
    
public:
    
    Utils();
    static double calcVariance(std::vector<double> values);
    static double calcMean(std::vector<double> values);
    static double calcStandardDeviation(std::vector<double> values);
    static std::vector<double> filterOutliers(std::vector<double> inValues);
    static void arrowedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);
    
};

#endif /* defined(__optical_flow_lk_image_template_match__Utils__) */
