#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> ScanImagePointer(Mat &inputMat, vector<Point2f> &points, int patchSize = 10);
static void arrowedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);

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
    int patchSize = 40;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
    
    Mat result, img1_gray, img2_gray, localisedSearchWindow, templatePatch;
    
   // cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
   // cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    
    cvtColor(img1, img1_gray, cv::COLOR_BGR2HSV);
    cvtColor(img2, img2_gray, cv::COLOR_BGR2HSV);
    
  //  img1_gray = img1.clone();
    
   // img2_gray = img2.clone();
    
    
    //CG - Calculate a central column through the two images that has a width of 10% of the original images.
    double imageCentreX = img1_gray.cols / 2;
    double imageROIWidth = img1_gray.cols * 0.5;
    double imageROIHalfWidth = imageROIWidth / 2;
    double imgROIStartX = imageCentreX - imageROIHalfWidth;
    double imgROIEndX = imageCentreX + imageROIHalfWidth;
    
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat roi = img1_gray( Rect(imgROIStartX,0,imageROIWidth,img1_gray.rows) );
    Mat roi2 = img2_gray( Rect(imgROIStartX,0,imageROIWidth,img2_gray.rows) );
    
    Mat opticalFlow = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    
    vector<Point2f> points;
    
    vector<Mat> test = ScanImagePointer(roi, points, patchSize);
    
    std::vector<Mat>::iterator i1;
    std::vector<Point2f>::iterator i2;
    
    int txtcount = 0;
    
    for( i1 = test.begin(), i2 = points.begin(); i1 < test.end() && i2 < points.end(); ++i1, ++i2 )
    {
        
//        if (txtcount >=1) {
//            break;
//        }
        
        Point2f templateOrigin = (*i2);
        templatePatch = *i1;
        
        //GaussianBlur( templatePatch, templatePatch, Size(123, 123), 0, 0 );
        
        int localisedWindowX = templateOrigin.x;
        
        int localisedWindowY = templateOrigin.y;
        
        int localisedWindowWidth = templatePatch.cols;
        
        int localisedWindowHeight = roi2.rows - localisedWindowY;
        
        localisedSearchWindow = roi2(Rect(localisedWindowX, localisedWindowY, localisedWindowWidth, localisedWindowHeight) );
        
        //GaussianBlur( localisedSearchWindow, localisedSearchWindow, Size( 123,123 ), 0, 0 );
        
        int result_cols =  localisedSearchWindow.cols - templatePatch.cols + 1;
        int result_rows = localisedSearchWindow.rows - templatePatch.rows + 1;
        
        result.create( result_cols, result_rows, CV_32FC1 );
        
        matchTemplate( localisedSearchWindow, templatePatch, result, match_method );
        
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
        
        // CG - Draw main ROI windows.
        rectangle( img1, Point(imgROIStartX, 0), Point(imgROIEndX , img1.rows), Scalar(0, 0, 255), 2, 8, 0 );
        rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
        
        
        // CG - Draw location of best match patch found in localised search window (column)
        rectangle( localisedSearchWindow, Point(matchLoc.x, matchLoc.y), Point(matchLoc.x + templatePatch.cols , matchLoc.y + templatePatch.rows), Scalar(0, 0, 255), 2, 8, 0 );
        
        
        matchLoc.x += localisedWindowX;
        
        
        rectangle( img1, Point(imgROIStartX + templateOrigin.x, templateOrigin.y), Point( (imgROIStartX + templateOrigin.x + templatePatch.cols), templateOrigin.y + templatePatch.rows ), Scalar(255, 0, 0), 2, 8, 0 );
        
        circle(img1, Point(imgROIStartX + templateOrigin.x, templateOrigin.y),5, Scalar(255,255,255), CV_FILLED, 8,0);
        
        rectangle( img2, Point(imgROIStartX + matchLoc.x, matchLoc.y), Point(imgROIStartX + matchLoc.x + templatePatch.cols , matchLoc.y + templatePatch.rows), Scalar(0, 255, 0), 2, 8, 0 );
        
        circle(img2, Point(imgROIStartX + matchLoc.x, matchLoc.y),5, Scalar(255,255,255), CV_FILLED, 8,0);

        //CG - Green OF motion vector.
        Point vectorOF(matchLoc.x - templateOrigin.x, matchLoc.y - templateOrigin.y);
        
        //CG - Blue OF 'Y'-component vector.
        Point vectorYComponent(templateOrigin.x - templateOrigin.x, matchLoc.y - templateOrigin.y);
        
        //CG - Check that the vector direction is downwards. Ignore any upwards directions.
        if ((templateOrigin.y - matchLoc.y) < 0) {
            
            //CG - OF motion vector arrow
            arrowedLine(img2, Point(imgROIStartX + templateOrigin.x , templateOrigin.y), Point(imgROIStartX + matchLoc.x, matchLoc.y), Scalar(0,255,0));
            
            
            //CG - OF motion vector arrow
            arrowedLine(opticalFlow, Point(imgROIStartX + templateOrigin.x, templateOrigin.y), Point(imgROIStartX + matchLoc.x, matchLoc.y), Scalar(0,255,0));
            
            
            //CG - y component
            arrowedLine(opticalFlow, Point(imgROIStartX + templateOrigin.x, templateOrigin.y), Point(imgROIStartX + templateOrigin.x, matchLoc.y), Scalar(255,0,0));
            
        }
        

        txtcount++;
        
    }

    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    imshow("Result", img2);
    
    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
    
    imshow("Merged Result", img2);
    
    imshow("OPTICAL FLOW", opticalFlow);
    
    imshow("Original", img1);
    
    imshow("Search Window", localisedSearchWindow);
    
    imshow("Template", templatePatch);
    
    imshow("Normalised Result matrix", result);
    
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

// CG - Here, we are passing 'inputMat' and 'points' by REFERENCE, NOT BY VALUE.
vector<Mat> ScanImagePointer(Mat &inputMat, vector<Point2f> &points, int patchSize)
{
    
    vector<Mat> mats;

    CV_Assert(inputMat.depth() != sizeof(uchar));
    
   // int channels = inputMat.channels();
    
    int nRows = inputMat.rows;

    // CG - Multiply each row by the colour channels (i.e. 3 for BGR, and 1 for grayscale)
   // int nCols = inputMat.cols * channels;
    
    int nCols = inputMat.cols;
    
    int i,j;
    uchar* p;
    
    for( i = 0; i < nRows - patchSize; i+=2)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for ( j = 0; j < nCols - patchSize; j+=2)
        {
           mats.push_back(inputMat(Rect(j,i,patchSize,patchSize)));
                
           //CG - Same as Point2f (typename alias)
           points.push_back(Point_<float>(j, i));
            
        }
        
    }
    
    // CG - We divide by three so we are only returning the total count pixels (and not each of the CHANNELS within EACH PIXEL as well).
    return mats;
}