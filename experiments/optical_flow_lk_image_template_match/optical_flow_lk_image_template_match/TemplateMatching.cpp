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

double TemplateMatching::calcSSDNormalised(Mat& patch1, Mat& patch2) {

    return (calcSSD(patch1, patch2) / calcNormalisationFactorLoop(patch1, patch2));
    
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

double TemplateMatching::calcNormalisationFactor(int pixel1, int pixel2) {
    
    
    double pix1Sqrd = (pixel1 * pixel1);
    
    double pix2Sqrd = (pixel2 * pixel2);
    
    double normResult = pix1Sqrd * pix2Sqrd;
    
    return sqrt(normResult);
    
}

double TemplateMatching::calcNormalisationFactorLoop(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    double norm = 0.0;
    
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
                    
                    double pix1Sqrd = (int(p1[j]) * int(p1[j]));
                    
                    double pix2Sqrd = (int(p2[j]) * int(p2[j]));
                    
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


// I THINK THIS IS INCORRECT, IT WE SHOULD ONLY SQRT THE TOTAL RESULT OF THE NORMALISATION VALUE, NOT SQRT EACH VALUE EVERY TIME. - KEPT HERE IN CASE I HAVE THIS WRONG!
double TemplateMatching::calcNormalisationFactorLoop2(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    double norm = 0.0;
    
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
                    
                    norm += calcNormalisationFactor(int(p1[j]), int(p2[j]));
                    
                    channelNo++;
                    
                } else {
                    
                    channelNo = 0;
                }
                
            }
        }
        
        
    } else {
        
        cout << "ERROR: Two images are not of the same size!" << endl;
    }
    
    return norm;
    
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