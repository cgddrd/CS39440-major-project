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

double TemplateMatching::calcEuclideanDistanceNorm(Mat patch1, Mat patch2) {
    
    Mat hsvChannelsImg1[3], hsvChannelsImg2[3];
    
    split(patch1, hsvChannelsImg1);
    split(patch2, hsvChannelsImg2);
    
    //Set VALUE channel to 0
    hsvChannelsImg1[2]=Mat::zeros(patch1.rows, patch1.cols, CV_8UC1);
    hsvChannelsImg2[2]=Mat::zeros(patch2.rows, patch2.cols, CV_8UC1);
    
    merge(hsvChannelsImg1,3,patch1);
    merge(hsvChannelsImg2,3,patch2);
    
    return norm(patch1, patch2, NORM_L2);
    
}

double TemplateMatching::calcEuclideanDistance(Mat& patch1, Mat& patch2) {
    
    // Euclidean distance returns the SQRT of the sum of squared differences.
    return sqrt(calcSSD(patch1, patch2));
}

long double TemplateMatching::calcSSDNormalised(Mat& patch1, Mat& patch2) {
    
    return (calcSSD(patch1, patch2) / calcNormalisationFactor(patch1, patch2));
    
}

long double TemplateMatching::calcCorrelationNorm(Mat& patch1, Mat& patch2) {
    
    long double correlation = calcCorrelation(patch1, patch2);
    
    long double norm = calcNormalisationFactor(patch1, patch2);
    
    return correlation / norm;
    
}

long double TemplateMatching::calcSSD(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    long double ssd = 0;
    
    // Template patch (patch1) is a ROI from a larger image, therefore it currently CANNOT BE CONTINUOUS!
    // Mat test = patch1.clone();
    
    //    imshow("patch1", patch1);
    //    imshow("patch2", patch2);
    //    waitKey();
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        int nRows = patch2.rows;
        int nCols = patch2.cols * patch2.channels();
        
        int i,j;
        long double diff = 0.0;
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

long double TemplateMatching::calcNormalisationFactor(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    long double norm = 0.0;
    long double pix1Sqrd = 0.0;
    long double pix2Sqrd = 0.0;
    
    // Template patch (patch1) is a ROI from a larger image, therefore it currently CANNOT BE CONTINUOUS!
    // Mat test = patch1.clone();
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        int nRows = patch2.rows;
        int nCols = patch2.cols * patch2.channels();
        
        int i,j;
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
                    
                    pix1Sqrd = (int(p1[j]) * int(p1[j]));
                    
                    pix2Sqrd = (int(p2[j]) * int(p2[j]));
                    
                    norm += (pix1Sqrd * pix2Sqrd);
                    
                    channelNo++;
                    
                } else {
                    
                    channelNo = 0;
                }
                
            }
        }
        
        
    } else {
        
        cout << "ERROR: Two images are not of the same size!" << endl;
    }
    
    return sqrt(norm);
    
}

long double TemplateMatching::calcCorrelation(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    long double ssd = 0;
    
    // Template patch (patch1) is a ROI from a larger image, therefore it currently CANNOT BE CONTINUOUS!
    // Mat test = patch1.clone();
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        int nRows = patch2.rows;
        int nCols = patch2.cols * patch2.channels();
        
        int i,j;
        long double diff = 0.0;
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
                    
                    
                    // WE MAY NOT NEED TO SQUARE THE RESULT FOR CORRELATION! - CHECK WITH FRED!!
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