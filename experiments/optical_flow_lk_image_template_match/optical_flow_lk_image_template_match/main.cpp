#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

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
    int y = 500;
    int windowsize = 100;
    
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
    
    // Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    
    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
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
    
}