#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Utils.hpp"

using namespace cv;
using namespace std;

#define MAX_COUNT 500
#define PERCENTAGE_WIDTH 0.20

vector<Point2f> ScanImagePointer(Mat& I, int step = 50);

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
           "Using the SURF desriptor:\n"
           "\n"
           "Usage:\n matcher_simple <image1> <image2>\n");
}

int main(int argc, char** argv) {
    
    if(argc != 3)
    {
        help();
        return -1;
    }
    
    static const int match_method = CV_TM_CCOEFF_NORMED;
    int y = 500;
    int windowsize = 100;
//    const bool useRoi = true;
//    
//    static const bool useText = false;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    Mat result, frame, grayFrames, rgbFrames, prevGrayFrame, resultFrame, descriptors;
    
    Mat img_display;

    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    cvtColor(img1, prevGrayFrame, cv::COLOR_BGR2GRAY);
    cvtColor(img2, grayFrames, cv::COLOR_BGR2GRAY);
    
    grayFrames.copyTo( img_display );
    
    
    
    double centre_point = prevGrayFrame.cols / 2;
    
    Mat templ = prevGrayFrame( Rect(centre_point - (windowsize / 2),y,windowsize,windowsize) );
    
    int result_cols =  grayFrames.cols - templ.cols + 1;
    int result_rows = grayFrames.rows - templ.rows + 1;
    
    result.create( result_cols, result_rows, CV_32FC1 );
    
    matchTemplate( grayFrames, templ, result, match_method );
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
    
    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    
    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
    else
    { matchLoc = maxLoc; }
    
    rectangle( img2, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(255, 0, 0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    
    rectangle( img1, Point(centre_point - (templ.cols / 2), y), Point( centre_point + (templ.cols / 2) , y + templ.rows ), Scalar(0, 255, 0), 2, 8, 0 );
    
    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
    resize(result, result, Size(result.cols/2, result.rows/2));
    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    
    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    imshow("Original", img1);
    imshow("Normalised Result", result);
    
    imshow("Result", img2);
    
    //CG - 'resultFrame' is already set to the second image ('img2') anyway.
    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
    
    imshow("Merged Result", img2);
    
    waitKey(0);
    return 0;
    
    
    
//    
//    Mat frame, grayFrames, rgbFrames, prevGrayFrame, resultFrame, descriptors;
//    
//    Mat opticalFlow = Mat(img1.rows, img1.cols, CV_32FC3);
//    
//    vector<Point2f> points1;
//    vector<Point2f> points2;
//    vector<Point2f> displacement_points;
//    
//    //vector<KeyPoint> keypoints1, keypoints2;
//
//    vector<uchar> status;
//    vector<float> err;
//    
//    int i, k;
//    
//    TermCriteria termcrit(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03);
//    
//    //CG - Size of the search window when conducting Lucas-Kanade optical flow analysis.
//    Size winSize(200, 200);
//
//    //CG - Calculate a central column through the two images that has a width of 10% of the original images.
//    double centre_point = img1.cols / 2;
//    double width_ten_percent = img1.cols * PERCENTAGE_WIDTH;
//    double half_width_ten_percent = width_ten_percent / 2;
//    
//    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
//    Mat roi = img1( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img1.rows) );
//    Mat roi2 = img2( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img1.rows) );
//    
//    if (!useRoi) {
//        roi = img1;
//        roi2 = img2;
//    }
//    
//    //CG - Convert the first ROI to gray scale so we can perform Shi-Tomasi feature detection.
//    cvtColor(roi, prevGrayFrame, cv::COLOR_BGR2GRAY);
//    
//    img2.copyTo(resultFrame);
//    
//    cvtColor(roi2, grayFrames, cv::COLOR_BGR2GRAY);
//    
//    points1 = ScanImagePointer(prevGrayFrame, 40);
//    
//    //CG - Perform the actual sparse optical flow within the ROI extracted from the two images.
//    calcOpticalFlowPyrLK(roi, roi2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
//    
//    cout << "Optical Flow Difference:\n\n";
//    
//    for (i = k = 0; i < points2.size(); i++) {
//        
//        //CG - Get a string ready to display the vector number in the image.
//        char str[4];
//        sprintf(str,"%d",i);
//        
//        if (useRoi) {
//            
//            //CG - We need to move the X position of both the start and end points for each vector over so that it is displayed within the bounds of thr ROI extracted from the main large image.
//            points1[i].x += (centre_point - half_width_ten_percent);
//            points2[i].x += (centre_point - half_width_ten_percent);
//            
//        }
//
//        
//        //CG - Draw the vector number above the vector arrow on the image.
//        if (useText) {
//           
//            putText(resultFrame, str, points1[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255));
//            putText(opticalFlow, str, points1[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255));
//            
//        }
//        
//        
//        //CG - Push the 'X' coord from the starting position, and the 'Y' coord from the second position, so we can draw a stright line vector showing the displacement (BLUE LINE)
//        displacement_points.push_back(Point2f(points1[i].x, points2[i].y));
//        
//        
//        //CG - If the motion vector is going in the UP direction, draw red arrow.
//        if ((points1[i].y - points2[i].y) > 0) {
//            
//            // CG - Note: Scalar COLOUR values use the format (B,G,R)
//            Utils::arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,0,255));
//            
//            // CG - Draw blue arrow to indicate displacement.
//            Utils::arrowedLine(resultFrame, points1[i], displacement_points[i], Scalar(255,0,0));
//            
//            circle(resultFrame, points1[i], 2, Scalar(255, 0, 0), 1, 1, 0);
//            
//            Utils::arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,0,255));
//            
//            Utils::arrowedLine(opticalFlow, points1[i], displacement_points[i], Scalar(255,0,0));
//            
//            circle(resultFrame, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
//         
//        //CG - Otherwise the motion must be going DOWN, so draw a green arrow.
//        } else {
//            
//            Utils::arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,255,0));
//            
//            Utils::arrowedLine(resultFrame, points1[i], displacement_points[i], Scalar(255,0,0));
//            
//            circle(resultFrame, points1[i], 2, Scalar(0, 0, 0), 1, 1, 0);
//            
//            Utils::arrowedLine(opticalFlow, points1[i], points2[i], Scalar(0,255,0));
//            
//            Utils::arrowedLine(opticalFlow, points1[i], displacement_points[i], Scalar(255,0,0));
//            
//            circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
//        }
//        
//        cout << "Vector: " << i << " - X: " << int(points1[i].x - points2[i].x) << ", Y: " << int(points1[i].y - points2[i].y) << ", Norm: " << norm(points1[i]-points2[i]) << ", Displacement: " << norm(points1[i]-displacement_points[i]) << "\n";
//        
//    }
//    
//    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
//    double alpha = 0.4;
//    
//    //CG - 'resultFrame' is already set to the second image ('img2') anyway.
//    addWeighted(img1, alpha, resultFrame, 1.0 - alpha , 0.0, resultFrame);
//    
//    //CG - Resize the images so we can see them on the screen.
//    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
//    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
//    resize(resultFrame, resultFrame, Size(resultFrame.cols/2, resultFrame.rows/2));
//    resize(opticalFlow, opticalFlow, Size(opticalFlow.cols/2, opticalFlow.rows/2));
//    resize(roi, roi, Size(roi.cols/2, roi.rows/2));
//    resize(roi2, roi2, Size(roi2.cols/2, roi2.rows/2));
//    
//    //CG - Display the two raw input images side by side for GUI comparison.
//    Mat inputDisplay(img1.rows, img1.cols+img2.cols, CV_8UC3);
//    Mat left(inputDisplay, Rect(0, 0, img1.cols, img1.rows));
//    img1.copyTo(left);
//    Mat right(inputDisplay, Rect(img1.cols, 0, img2.cols, img2.rows));
//    img2.copyTo(right);
//    
//    //CG - Create windows for display.
//    namedWindow( "Input Images", WINDOW_NORMAL );
//
//    //CG - Show the images on screen.
//    imshow("Input Images", inputDisplay);
//    
//    imshow("Optical Flow Output (Blended Images)", resultFrame);
//    
//    imshow("Optical Flow Output (Raw)", opticalFlow);
//    
//    //CG - Wait for the user to press a key before exiting.
//    cvWaitKey(0);
    
}

vector<Point2f> ScanImagePointer(Mat& I, int step)
{
    
    vector<Point2f> points;
    
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    int channels = I.channels();
    
    int nRows = I.rows;
    
    //CG - Multiply each column by the colour channels (i.e. 3 for BGR, and 1 for grayscale)
    //CG - NOTE: As we should only be dealing with GRAYSCALE images, we should only have ONE CHANNEL, and therefore we should only loop through each of the columns once.
    int nCols = I.cols * channels;
    
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; i+=step)
    {
        p = I.ptr<uchar>(i);
        
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for ( j = 0; j < nCols; j+=step)
        {
            
            points.push_back(Point2f(j, i));
            
        }
        
    }
    
    // CG - We divide by three so we are only returning the total count pixels (and not each of the CHANNELS within EACH PIXEL as well).
    return points;
}