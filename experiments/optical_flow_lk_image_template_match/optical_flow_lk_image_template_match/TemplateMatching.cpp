//
//  TemplateMatching.cpp
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 05/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include "TemplateMatching.h"

using namespace std;
using namespace cv;

TemplateMatching::TemplateMatching() {}

double TemplateMatching::calcEuclideanDistanceNorm(Mat& patch1, Mat& patch2) {
    
    return norm(patch1, patch2, NORM_L2);
    
}

double TemplateMatching::calcEuclideanDistance(Mat& patch1, Mat& patch2) {
    
    // Euclidean distance returns the SQRT of the sum of squared differences.
    return sqrt(calcSSD(patch1, patch2));
}

double TemplateMatching::calcSSD(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    double ssd = 0;
    
    // Template patch (patch1) is a ROI from a larger image, therefore it currently CANNOT BE CONTINUOUS!
    // Mat test = patch1.clone();
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        int nRows = patch2.rows;
        int nCols = patch2.cols * patch2.channels();
        
        if (patch1.isContinuous() && patch2.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        
        int i,j;
        double diff = 0.0;
        uchar* p1;
        uchar* p2;
        int channelNo = 0;
        
        for( i = 0; i < nRows; ++i)
        {
            
            // Get a pointer to the VECTOR of COLUMNS for the current ROW.
            p1 = patch1.ptr<uchar>(i);
            p2 = patch2.ptr<uchar>(i);
            
            for ( j = 0; j < nCols; ++j)
            {
                
                //We only want to extract channels 0 and 1. (2 = Value (HSV) -> IGNORE!)
                if (channelNo <= 1) {
                    
                    diff = int(p1[j]) - int(p2[j]);
                    
                    ssd += (diff * diff);
                    
                    channelNo++;
                    
                } else {
                    
                    channelNo = 0;
                }
                
            }
        }
        
        
    } else {
        
        cout << "ERROR: Two images are not of the same size!" << endl;
    }
    
    return ssd;
    
}

double TemplateMatching::calcCorrelaton(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    double ssd = 0;
    
    // Template patch (patch1) is a ROI from a larger image, therefore it currently CANNOT BE CONTINUOUS!
    // Mat test = patch1.clone();
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        int nRows = patch2.rows;
        int nCols = patch2.cols * patch2.channels();
        
        if (patch1.isContinuous() && patch2.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        
        int i,j;
        double diff = 0.0;
        uchar* p1;
        uchar* p2;
        int channelNo = 0;
        
        for( i = 0; i < nRows; ++i)
        {
            
            // Get a pointer to the VECTOR of COLUMNS for the current ROW.
            p1 = patch1.ptr<uchar>(i);
            p2 = patch2.ptr<uchar>(i);
            
            for ( j = 0; j < nCols; ++j)
            {
                
                //We only want to extract channels 0 and 1. (2 = Value (HSV) -> IGNORE!)
                if (channelNo <= 1) {
                    
                    diff = int(p1[j]) * int(p2[j]);
                    
                    ssd += (diff * diff);
                    
                    channelNo++;
                    
                } else {
                    
                    channelNo = 0;
                }
                
            }
        }
        
        
    } else {
        
        cout << "ERROR: Two images are not of the same size!" << endl;
    }
    
    return ssd;
    
}