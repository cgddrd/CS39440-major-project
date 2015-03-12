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
    
    double windowScale = 5;
    
    int width = 100;
    int height = 50;

    Mat opticalFlow = Mat(height * windowScale, width * windowScale, CV_32FC3);
    
    int originX = (width * (windowScale / 2)) - (width / 2);
    int originY = (height * (windowScale / 2)) - (height / 2);
    
    Point topLeft(originX, originY);
    Point topRight(originX + width, originY);
    Point bottomLeft(originX, originY + height);
    Point bottomRight(originX + width, originY + height);
    Point centreLeft(originX, originY + (height / 2));
    Point centreCentre(originX + (width / 2), originY + (height / 2));
    Point centreRight(originX + width, originY + (height / 2));
    Point bottomCentre(originX + (width / 2), originY + height);
    Point topCentre(originX + (width / 2), originY);
    
    vector<Point> oldPoints {topLeft, topCentre, topRight, centreLeft, centreCentre, centreRight, bottomLeft, bottomCentre, bottomRight};

    double old_size = width;
    
    double scaleUnit = (old_size + 10) / old_size;
    
    double scalePercent = 2; //Double the original size.
    
    
    // Modified from original source: http://stackoverflow.com/questions/18791880/
    for(std::vector<Point>::iterator it = oldPoints.begin(); it != oldPoints.end(); ++it) {
        
        circle(opticalFlow, *it, 2, Scalar(0,0,255));
        
        // 1) Calculate vector from centre (X,Y) of rectangle to current (X,Y) point being processed (e.g. top-left corner).
        Point vec = *it - centreCentre;
        
        // 2) Scale this vector (e.g. diagnonal vector if centre -> top-left corner) either using scale factor, or single unit.
        Point scaleUnitResultPoint = vec * scaleUnit;
        Point scalePercentResultPoint = vec * scalePercent;
        
        // 3) Draw a circle at the point of the scaled (geometrically transformed) position.
        circle(opticalFlow, (centreCentre + scaleUnitResultPoint), 2, Scalar(0,255,0));
        circle(opticalFlow, (centreCentre + scalePercentResultPoint), 2, Scalar(255,128,0));
    }
    
    
    putText(opticalFlow, "Original", Point(10,15), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255));
    
    
    // Provide additional scope so destructor for ostringstream is called earlier than normal.
    {
        std::ostringstream str;
        str << "Scale Unit: " << scaleUnit;
        putText(opticalFlow, str.str(), Point(10,30), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
    }
    
    {
        std::ostringstream str;
        str << "Scale Percentage: " << scalePercent;
        putText(opticalFlow, str.str(), Point(10,45), CV_FONT_HERSHEY_PLAIN, 1, Scalar(255,128,0));
    }
    
    imshow("output", opticalFlow);
    
    waitKey();
    
    
    return 0;
}
