//
//  Utils.hpp
//  optical_flow_lk_two_images
//
//  Created by Connor Goddard on 21/02/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#ifndef optical_flow_lk_two_images_Utils_hpp
#define optical_flow_lk_two_images_Utils_hpp

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

class Utils
{
    
    public:
    
        Utils();
        static void arrowedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);
        static void printParams(cv::Algorithm* algorithm);
    
};

#endif
