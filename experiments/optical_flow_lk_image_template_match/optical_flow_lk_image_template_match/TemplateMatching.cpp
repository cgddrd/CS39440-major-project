//
//  EuclideanDistance.cpp
//  optical_flow_lk_image_template_match
//
//  Created by Connor Goddard on 05/03/2015.
//  Copyright (c) 2015 Connor Goddard. All rights reserved.
//

#include "TemplateMatching.h"

using namespace std;
using namespace cv;

TemplateMatching::TemplateMatching() {}

void TemplateMatching::calcEuclideanDistance() {
    
    
    
}

void TemplateMatching::calcSSD(Mat& patch1, Mat& patch2) {
    
    // accept only char type matrices
    CV_Assert(patch1.depth() != sizeof(uchar));
    CV_Assert(patch2.depth() != sizeof(uchar));
    
    if (patch1.rows == patch2.rows && patch1.cols == patch2.cols) {
        
        
        
        // SLOWER/SAFER APPROACH USING ITERATOR!
        
        MatIterator_<Vec3b> it, end;
        double t = (double)getTickCount();
        for( it = patch1.begin<Vec3b>(), end = patch1.end<Vec3b>(); it != end; ++it)
        {
            
            // Set the B and G channels to 0.
            (*it)[0] = 0;
            (*it)[1] = 0;
            
        }
        
        t = ((double)getTickCount() - t)/getTickFrequency();
        cout << "Times passed in seconds: " << t << endl;
        
        
        // FASTEST APPROACH - USING C-STYLE POINTER!
        
        int nRows = patch2.rows;
        int nCols = patch2.cols;
        
        if (patch2.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        
        int i,j;
        Vec3b* p;
        
        t = (double)getTickCount();
        
        for( i = 0; i < nRows; ++i)
        {
            
            // Get a pointer to the VECTOR of COLUMNS for the current ROW.
            p = patch2.ptr<Vec3b>(i);
            
            for ( j = 0; j < nCols; ++j)
            {
                // Access the current COLUMN (j) and then within that we can access each of the channels: B (0), G (1), R(2).
                // Set the B and G channels to 0.
                p[j][0] = 0;
                p[j][1] = 0;
            }
        }
        
        t = ((double)getTickCount() - t)/getTickFrequency();
        cout << "Times passed in seconds2: " << t << endl;
        
        imshow("fsdfs", patch1);
        imshow("fsdfs", patch2);
        waitKey();
        
    } else {
        
        cout << "ERROR: Two images are not of the same size!" << endl;
    }
    
}


