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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TemplateMatching {
    
public:
    
    TemplateMatching();
    static void calcEuclideanDistance();
    static void calcSSD(cv::Mat& patch1, cv::Mat& patch2);
};

#endif /* defined(__optical_flow_lk_image_template_match__EuclideanDistance__) */
