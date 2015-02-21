//
//  Utils.cpp
//  optical_flow_lk_two_images
//
//  Created by Connor Goddard on 21/02/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include "Utils.hpp"

using namespace std;
using namespace cv;

Utils::Utils() {
    
    
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

void Utils::printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);
    
    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;
        
        switch (type) {
            case cv::Param::BOOLEAN:
                typeText = "bool";
                break;
            case cv::Param::INT:
                typeText = "int";
                break;
            case cv::Param::REAL:
                typeText = "real (double)";
                break;
            case cv::Param::STRING:
                typeText = "string";
                break;
            case cv::Param::MAT:
                typeText = "Mat";
                break;
            case cv::Param::ALGORITHM:
                typeText = "Algorithm";
                break;
            case cv::Param::MAT_VECTOR:
                typeText = "Mat vector";
                break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}
