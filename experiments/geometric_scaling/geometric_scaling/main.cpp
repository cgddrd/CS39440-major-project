//
//  main.cpp
//  geometric_scaling
//
//  Created by Connor Goddard on 12/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    
    Mat opticalFlow = Mat(500, 500, CV_32FC3);
    
    vector<Point> oldPoints {Point(50, 50), Point(150, 50), Point(50, 100), Point(150, 100)};
    
    vector<Point> newPoints;
    
    Point centrePoint(100, 75);
    
    for(std::vector<Point>::iterator it = oldPoints.begin(); it != oldPoints.end(); ++it) {
        
        circle(opticalFlow, *it, 1, Scalar(0,0,255));
        
        Point vec = *it - centrePoint;
        
        vec *= 1.5;
        
        newPoints.push_back(centrePoint + vec);
        
        circle(opticalFlow, (centrePoint + vec), 1, Scalar(0,255,0));
    }
    
    imshow("output", opticalFlow);
    
    waitKey();
    
    
    return 0;
}
