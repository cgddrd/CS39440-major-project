#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> ScanImagePointer(Mat &inputMat, vector<Point2f> &points, int patchSize = 10);
static void arrowedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);
static float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);

static void help()
{
    printf("Usage: ./optical_flow_lk_image_template_match <image1> <image2>\n");
}

int main(int argc, char** argv) {
    
    if(argc != 3)
    {
        help();
        return -1;
    }
    
    static const int match_method = CV_TM_CCOEFF_NORMED;
    int patchSize = 30;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    
    resize(img1, img1, Size(img1.cols/4, img1.rows/4));
    resize(img2, img2, Size(img2.cols/4, img2.rows/4));
    
    Mat result, img1_gray, img2_gray;
    
    cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    
    //CG - Calculate a central column through the two images that has a width of 10% of the original images.
    double centre_point = img1_gray.cols / 2;
    double width_ten_percent = img1_gray.cols * 0.3;
    double half_width_ten_percent = width_ten_percent / 2;
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat roi = img1_gray( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img1_gray.rows) );
    Mat roi2 = img2_gray( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img2_gray.rows) );
    
    Mat opticalFlow = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    
    //Mat templ = img1_gray( Rect(centre_point - (windowsize / 2),y,windowsize,windowsize) );
    
    int result_cols =  roi.cols - patchSize + 1;
    int result_rows = roi.rows - patchSize + 1;
    
    result.create( result_cols, result_rows, CV_32FC1 );
    
    vector<Point2f> points;
    
    vector<Mat> test = ScanImagePointer(roi, points, patchSize);
    
    std::vector<Mat>::iterator i1;
    std::vector<Point2f>::iterator i2;
    
    int txtcount = 0;
    
    
    for( i1 = test.begin(), i2 = points.begin(); i1 < test.end() && i2 < points.end(); ++i1, ++i2 )
    {
        //doSomething( *i1, *i2 );
        
        //cout << "Mat: " << *i1 << ", Point X: " << int((*i2).x) << ", Point Y:" << int((*i2).y) << endl;
        
        
        Mat templ = *i1;
        
        matchTemplate( roi2, templ, result, match_method );
        normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
        
        // Localizing the best match with minMaxLoc
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        
        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        
        // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
        if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
        { matchLoc = minLoc; }
        else
        { matchLoc = maxLoc; }
        
        
        rectangle( img1, Point(centre_point - half_width_ten_percent, 0), Point(centre_point + half_width_ten_percent , img1.rows), Scalar(0, 0, 255), 2, 8, 0 );
        rectangle( img2, Point(centre_point - half_width_ten_percent, 0), Point(centre_point + half_width_ten_percent , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
        
        rectangle( img1, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point( ((*i2).x + (centre_point - half_width_ten_percent) + templ.cols) , (*i2).y + templ.rows ), Scalar(255, 0, 0), 2, 8, 0 );
        
        
        rectangle( img2, Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Point( matchLoc.x + (centre_point - half_width_ten_percent) + templ.cols , matchLoc.y + templ.rows ), Scalar(0, 255, 0), 2, 8, 0 );
        
        
        //rectangle( result, Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Point( matchLoc.x + (centre_point - half_width_ten_percent) + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    
        char str[10];
        char str2[10];
        
        //sprintf(str,"%d",txtcount);
        
        //putText(img1, str, Point((*i2).x + (centre_point - half_width_ten_percent) + (patchSize / 2), (*i2).y + (patchSize / 2)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        
      //  putText(img2, str, Point(matchLoc.x + (centre_point - half_width_ten_percent) + (patchSize / 2), matchLoc.y + (patchSize / 2)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        
        
        
        Point vectorOF(matchLoc.x - (*i2).x, matchLoc.y - (*i2).y);
        
        Point vectorYComponent((*i2).x - (*i2).x, matchLoc.y - (*i2).y);
        
        
        float angle = innerAngle((*i2).x + (centre_point - half_width_ten_percent), matchLoc.y, matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y,  (*i2).x + (centre_point - half_width_ten_percent), (*i2).y);
        
        
        float lengthOFVector = sqrt(pow(vectorOF.x, 2) + pow(vectorOF.y, 2));
        
        
        sprintf(str,"%3.1f",angle);
        
        sprintf(str2,"%3.1f",lengthOFVector);
        
        
        if (((*i2).y - matchLoc.y) > 0 || (((*i2).y - matchLoc.y) < (2 * patchSize * -1)) || angle < 45) {
            
//            arrowedLine(img2, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(0,0,255));
//            
//             arrowedLine(opticalFlow, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(0,0,255));
//            
//             arrowedLine(opticalFlow, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point((*i2).x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(255,0,0));
            
        } else {
            
            arrowedLine(img2, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(0,255,0));
            
            
            //CG - OF motion vector
            arrowedLine(opticalFlow, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point(matchLoc.x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(0,255,0));
            
            
            //CG - y component
            arrowedLine(opticalFlow, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), Point((*i2).x + (centre_point - half_width_ten_percent), matchLoc.y), Scalar(255,0,0));
            
            
           // double Angle = atan2(matchLoc.y - (*i2).y,matchLoc.x - (*i2).x) * 180.0 / CV_PI;
            
            //if(Angle<0) Angle=Angle+360;
        
            
            //double Angle = atan2(vectorOF.y,vectorOF.x) - atan2(vectorYComponent.y,vectorYComponent.x) * 180.0 / CV_PI;
            
           
            
          //  putText(img2, str, Point(matchLoc.x matchLoc.y + (patchSize)), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255));
            
            putText(opticalFlow, str, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
            
            putText(opticalFlow, str2, Point((*i2).x + (centre_point - half_width_ten_percent), (*i2).y - 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0));
            
        }
        
        
        
        txtcount++;
        
        
        // rectangle( img1, Point(centre_point - (templ.cols / 2), y), Point( centre_point + (templ.cols / 2) , y + templ.rows ), Scalar(0, 255, 0), 2, 8, 0 );
    }
    
//    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
//    //resize(result, result, Size(result.cols/2, result.rows/2));
//    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
//    resize(roi, roi, Size(roi.cols/2, roi.rows/2));
//    resize(roi2, roi2, Size(roi2.cols/2, roi2.rows/2));
    
    //resize(opticalFlow, opticalFlow, Size(opticalFlow.cols * 2, opticalFlow.rows * 2));
    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    //imshow("Normalised Result", result);

    imshow("ROI1", roi);
    imshow("ROI2", roi2);
    
    imshow("Result", img2);
    
    //CG - 'resultFrame' is already set to the second image ('img2') anyway.
    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
    
    imshow("Merged Result", img2);
    
    imshow("OPTICAL FLOW", opticalFlow);
    
    imshow("Original", img1);
    
    waitKey(0);
    
    return 0;
    
}

// CG - Arrowed Line drawing method. (PhilLab - http://stackoverflow.com/questions/10161351) (Built into OpenCV v3.0.0)
static void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, int shift, double tipLength)
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

static float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{
    
    float dist1 = sqrt(  (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) );
    float dist2 = sqrt(  (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) );
    
    float Ax, Ay;
    float Bx, By;
    float Cx, Cy;
    
    //find closest point to C
    //printf("dist = %lf %lf\n", dist1, dist2);
    
    Cx = cx1;
    Cy = cy1;
    if(dist1 < dist2)
    {
        Bx = px1;
        By = py1;
        Ax = px2;
        Ay = py2;
        
        
    }else{
        Bx = px2;
        By = py2;
        Ax = px1;
        Ay = py1;
    }
    
    
    float Q1 = Cx - Ax;
    float Q2 = Cy - Ay;
    float P1 = Bx - Ax;
    float P2 = By - Ay;  
    
    
    float A = acos( (P1*Q1 + P2*Q2) / ( sqrt(P1*P1+P2*P2) * sqrt(Q1*Q1+Q2*Q2) ) );
    
    A = A*180/M_PI;
    
    return A;
}




// CG - Here, we are passing 'inputMat' and 'points' by REFERENCE, NOT BY VALUE.
vector<Mat> ScanImagePointer(Mat &inputMat, vector<Point2f> &points, int patchSize)
{
    
    vector<Mat> mats;
    
 //   int count = 0;
    
    // accept only char type matrices
    CV_Assert(inputMat.depth() != sizeof(uchar));
    
    int channels = inputMat.channels();
    
    int nRows = inputMat.rows;
    
    //int nRows = 1;
    
    // CG - Multiply each row by the colour channels (i.e. 3 for BGR, and 1 for grayscale)
    int nCols = inputMat.cols * channels;
    
    int i,j;
    uchar* p;
    
   // int tempPatch = patchSize;
    
    //int newPatchSize = patchSize;
    
    for( i = ((patchSize/2) + 1); i < nRows - ((patchSize /2) + 1); i+=patchSize)
    {
        
//        if (count == 500) {
//            
//            break;
//            
//        }
        
        p = inputMat.ptr<uchar>(i);
        
      //  count++;
        
       // patchSize = tempPatch;
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for ( j = ((patchSize/2) + 1); j < nCols - ((patchSize/2) + 1); j+=patchSize)
        {
            
            //I( Rect(j - (patchSize / 2),i - (patchSize / 2),patchSize,patchSize) );
            
            
            
           int x = j - (patchSize / 2);
           
           int y = i - (patchSize / 2);
//            
//            cout << "X: " << x << ", J: " << j << ", Y: " << y << ", I: " << i << endl;
            
//            if (x > (patchSize / 2) && x < nCols - (patchSize / 2) && y > (patchSize / 2) && y < nRows - (patchSize / 2) && (x + patchSize) <= nCols) {
//            
//                 mats.push_back(inputMat(Rect(x,y,patchSize,patchSize)));
//                
//            }
            
//            x = x >= 0 ? x : 0;
//            
//            y = x >= 0 ? x : 0;
//            
//            x = (x+ patchSize) <= nCols ? x : nCols;
//            
//            x = (y+ patchSize) <= nRows ? x : nRows;
            
            
//            if (x < 0) {
//            
//                //Subtract x from patchSize (patchSize + (-x))
//                patchSize += x;
//                
//                x = 0;
//                
//            }
//            
//            if ((x + patchSize) > nCols) {
//                
//                patchSize -= (x + patchSize);
//                
//                x = nCols-patchSize;
//            }
//            
//            if (y < 0) {
//                
//                //Subtract x from patchSize (patchSize + (-x)
//                patchSize += y;
//                
//                y = 0;
//                
//            }
//            
//            if ((y + patchSize) > nRows) {
//                
//                patchSize -= (y + patchSize);
//                
//                y = nRows-patchSize;
//            }
//            
            //cout << "X: " << x << ", J: " << j << ", Y: " << y << ", I: " << i << endl;
            
           // if (x >= 0 && y >= 0 && (x + patchSize) <= nCols && (y + patchSize) <= nRows) {
            
           // if (patchSize > 0) {
                
                mats.push_back(inputMat(Rect(x,y,patchSize,patchSize)));
                
                //CG - Same as Point2f (typename alias)
                points.push_back(Point_<float>(x, y));
                
           // }
        
            
                
           // }
            
        
            
//            newPatchSize = patchSize > 0 ? patchSize : 1;
//            
//            patchSize = tempPatch;
            
        }
        
    }
    
    // CG - We divide by three so we are only returning the total count pixels (and not each of the CHANNELS within EACH PIXEL as well).
    return mats;
}