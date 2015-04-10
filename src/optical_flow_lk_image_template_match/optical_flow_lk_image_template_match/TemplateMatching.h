//
//  EuclideanDistance.h
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 05/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#ifndef __optical_flow_lk_image_template_match__EuclideanDistance__
#define __optical_flow_lk_image_template_match__EuclideanDistance__

#include <stdio.h>
#include <iostream>
#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TemplateMatching {
    
public:
    
    TemplateMatching();
    static double calcEuclideanDistance(cv::Mat& patch1, cv::Mat& patch2);
    static double calcEuclideanDistanceNorm(cv::Mat patch1, cv::Mat patch2);
    static long double calcSSD(cv::Mat& patch1, cv::Mat& patch2);
    static long double calcCorrelation(cv::Mat& patch1, cv::Mat& patch2);
    static long double calcSSDNormalised(cv::Mat& patch1, cv::Mat& patch2);
    static long double calcNormalisationFactor(cv::Mat& patch1, cv::Mat& patch2);
    static long double calcCorrelationNorm(cv::Mat& patch1, cv::Mat& patch2);
};

#endif /* defined(__optical_flow_lk_image_template_match__EuclideanDistance__) */
