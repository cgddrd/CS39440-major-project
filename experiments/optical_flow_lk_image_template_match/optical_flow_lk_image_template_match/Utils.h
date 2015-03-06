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

class Utils {
    
public:
    
    Utils();
    static double calcVariance(std::vector<double> values);
    static double calcMean(std::vector<double> values);
    static double calcStandardDeviation(std::vector<double> values);
    static std::vector<double> filterOutliers(std::vector<double> inValues);
    
};

#endif /* defined(__optical_flow_lk_image_template_match__Utils__) */
