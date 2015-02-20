#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

// CG - Arrowed Line drawing method. (PhilLab - http://stackoverflow.com/questions/10161351) (Built into OpenCV v3.0.0)
static void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1)
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

static void printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);
    
    for (int i = 0; i < (int) parameters.size(); i++) {
        std::string param = parameters[i];
        int type = algorithm->paramType(param);
        std::string helpText = algorithm->paramHelp(param);
        std::string typeText;
        
        switch (type) {
            case cv::Param::BOOLEAN:
                typeText = "bool";
                break;
            case cv::Param::INT:
                typeText = "int";
                break;
            case cv::Param::REAL:
                typeText = "real (double)";
                break;
            case cv::Param::STRING:
                typeText = "string";
                break;
            case cv::Param::MAT:
                typeText = "Mat";
                break;
            case cv::Param::ALGORITHM:
                typeText = "Algorithm";
                break;
            case cv::Param::MAT_VECTOR:
                typeText = "Mat vector";
                break;
        }
        std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    }
}

int main(int argc, char** argv) {
    
    if(argc != 3)
    {
        help();
        return -1;
    }
    
    const bool useRoi = false;
    
    static const bool useText = false;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    Mat frame, grayFrames, rgbFrames, prevGrayFrame, resultFrame, descriptors;
    
    Mat opticalFlow = Mat(img1.rows, img1.cols, CV_32FC3);
    
    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<Point2f> displacement_points;
    vector<KeyPoint> keypoints1, keypoints2;
    
    Point2f diff;
    
    vector<uchar> status;
    vector<float> err;
    
    int i, k;
    
    TermCriteria termcrit(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03);
    
    //CG - Size of the search window when conducting Lucas-Kanade optical flow analysis.
    Size winSize(200, 200);

    //CG - Calculate a central column through the two images that has a width of 10% of the original images.
    double centre_point = img1.cols / 2;
    double width_ten_percent = img1.cols * PERCENTAGE_WIDTH;
    double half_width_ten_percent = width_ten_percent / 2;
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat roi = img1( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img1.rows) );
    Mat roi2 = img2( Rect(centre_point - half_width_ten_percent,0,width_ten_percent,img1.rows) );
    
    
    //CG - TEMP CODE TO SWITCH OFF THE COLUMN THING.
    if (!useRoi) {
        roi = img1;
        roi2 = img2;
    }
    
    //CG - Convert the first ROI to gray scale so we can perform Shi-Tomasi feature detection.
    //cvtColor(img1, prevGrayFrame, cv::COLOR_BGR2GRAY);
    cvtColor(roi, prevGrayFrame, cv::COLOR_BGR2GRAY);

    //SIFT sift(2000,3,0.004);
    //sift(prevGrayFrame, prevGrayFrame, keypoints1, descriptors, false);
    
    //CG - This is the same as the code above, but we are just going about it in a different way.
    //Ptr<FeatureDetector> detector = FeatureDetector::create("FAST");
    
    //CG - We are setting the parameters manually using the 'ALGORITHM' method 'set' (this is the same as in the constructor above).
//    detector->set("contrastThreshold", 0.004);
//    detector->set("nFeatures", 2000);
//    detector->set("nOctaveLayers", 3);
    
    // CG - Non-max supression OFF = > no of features (6000+)
    //detector->set("nonmaxSuppression", false);
    //detector->set("threshold", 15);
    
    //SURF surf(50);
    //surf(prevGrayFrame, prevGrayFrame, keypoints1);
    
    //FastFeatureDetector detector(15);
    //detector.detect(prevGrayFrame, keypoints1);
    
   // printParams(detector);

   // detector->detect(prevGrayFrame, keypoints1);
    
   // KeyPoint::convert(keypoints1, points1);
    
    //CG - Perform Shi-Tomasi feature detection.
    //goodFeaturesToTrack(prevGrayFrame, points1, MAX_COUNT, 0.01, 5, Mat(), 3, 0, 0.04);
    
    img2.copyTo(resultFrame);
    
    cvtColor(roi2, grayFrames, cv::COLOR_BGR2GRAY);
    
    
    //Point2f connorpoint = roi.at<Point2f>(Point(500, 500));

    
    //points1.push_back(connorpoint);
    
    points1 = ScanImagePointer(prevGrayFrame, 50);
    
    //CG - Perform the actual sparse optical flow within the ROI extracted from the two images.
    calcOpticalFlowPyrLK(roi, roi2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    
    cout << "Optical Flow Difference:\n\n";
    
    for (i = k = 0; i < points2.size(); i++) {
        
        //CG - Get a string ready to display the vector number in the image.
        char str[4];
        sprintf(str,"%d",i);
        
        if (useRoi) {
            //CG - We need to move the X position of both the start and end points for each vector over so that it is displayed within the bounds of thr ROI extracted from the main large image.
            points1[i].x += (centre_point - half_width_ten_percent);
            points2[i].x += (centre_point - half_width_ten_percent);
        }

        
        //CG - Draw the vector number above the vector arrow on the image.
        
        if (useText) {
           
            putText(resultFrame, str, points1[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255));
            putText(opticalFlow, str, points1[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255));
            
        }
        
        
        //CG - Push the 'X' coord from the starting position, and the 'Y' coord from the second position, so we can draw a stright line vector showing the displacement (BLUE LINE)
        displacement_points.push_back(Point2f(points1[i].x, points2[i].y));
        
        
        //CG - If the motion vector is going in the UP direction, draw red arrow.
        if ((points1[i].y - points2[i].y) > 0) {
            
            // CG - Note: Scalar COLOUR values use the format (B,G,R)
            arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,0,255));
            
            // CG - Draw blue arrow to indicate displacement.
            arrowedLine(resultFrame, points1[i], displacement_points[i], Scalar(255,0,0));
            
            circle(resultFrame, points1[i], 2, Scalar(255, 0, 0), 1, 1, 0);
            
            arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,0,255));
            
            arrowedLine(opticalFlow, points1[i], displacement_points[i], Scalar(255,0,0));
            
            circle(resultFrame, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
         
        //CG - Otherwise the motion must be going DOWN, so draw a green arrow.
        } else {
            
            arrowedLine(resultFrame, points1[i], points2[i], Scalar(0,255,0));
            
            arrowedLine(resultFrame, points1[i], displacement_points[i], Scalar(255,0,0));
            
            circle(resultFrame, points1[i], 2, Scalar(0, 0, 0), 1, 1, 0);
            
            arrowedLine(opticalFlow, points1[i], points2[i], Scalar(0,255,0));
            
            arrowedLine(opticalFlow, points1[i], displacement_points[i], Scalar(255,0,0));
            
            circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1, 0);
        }
        
        cout << "Vector: " << i << " - X: " << int(points1[i].x - points2[i].x) << ", Y: " << int(points1[i].y - points2[i].y) << ", Norm: " << norm(points1[i]-points2[i]) << ", Displacement: " << norm(points1[i]-displacement_points[i]) << "\n";
        
    }
    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    //CG - 'resultFrame' is already set to the second image ('img2') anyway.
    addWeighted(img1, alpha, resultFrame, 1.0 - alpha , 0.0, resultFrame);
    
    //CG - Resize the images so we can see them on the screen.
   
    resize(img1, img1, Size(img1.cols/2, img1.rows/2));
    resize(img2, img2, Size(img2.cols/2, img2.rows/2));
    resize(resultFrame, resultFrame, Size(resultFrame.cols/2, resultFrame.rows/2));
    resize(opticalFlow, opticalFlow, Size(opticalFlow.cols/2, opticalFlow.rows/2));
    resize(roi, roi, Size(roi.cols/2, roi.rows/2));
    resize(roi2, roi2, Size(roi2.cols/2, roi2.rows/2));
    
    //CG - Display the two raw imnput images side by side for GUI comparison.
    Mat inputDisplay(img1.rows, img1.cols+img2.cols, CV_8UC3);
    Mat left(inputDisplay, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(left);
    Mat right(inputDisplay, Rect(img1.cols, 0, img2.cols, img2.rows));
    img2.copyTo(right);
    
    //CG - Create windows for display.
    namedWindow( "Input Images", WINDOW_NORMAL );
//    namedWindow( "Image 2", WINDOW_NORMAL );
//    namedWindow( "Result", WINDOW_NORMAL );
//    namedWindow( "OF", WINDOW_NORMAL );

    //CG - Show the images on screen.
    imshow("Input Images", inputDisplay);
    
    //imshow("Image 2", roi2);
    
    imshow("Optical Flow Output (Blended Images)", resultFrame);
    
    imshow("Optical Flow Output (Raw)", opticalFlow);
    
    //CG - Wait for the user to press a key before exiting.
    cvWaitKey(0);
    
}

vector<Point2f> ScanImagePointer(Mat& I, int step)
{
    
    int count = 0;
    
    vector<Point2f> points;
    
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    int channels = I.channels();
    
    int nRows = I.rows;
    
    // CG - Multiply each row by the colour channels (i.e. 3 for BGR, and 1 for grayscale)
    int nCols = I.cols * channels;
    
//    if (I.isContinuous())
//    {
//        nCols *= nRows;
//        nRows = 1;
//    }
    
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; i+=step)
    {
        p = I.ptr<uchar>(i);
        
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for ( j = 0; j < nCols; j+=step)
        {
            //cout << int(p[j]) << endl;
            
            count++;
            
            //points.push_back(I.at<Point2f>(i, j));
            
            points.push_back(Point2f(j, i));
            
            //cout << I.at<Point2f>(i, j).x << endl;
            
        }
    }
    
//    // CG - Scan through the rows first (Y)
//    for(int j = 0;j < img.rows;j++){
//        
//        // CG - Then scan through the columns (X)
//        for(int i = 0;i < img.cols;i++){
//            
//            // CG - Need to use the Mat 'step' value to move across pixels.
////            unsigned char b = input[img.step * j + i];
////            unsigned char g = input[img.step * j + i + 1];
////            unsigned char r = input[img.step * j + i + 2];
//            
//            //cout << "B: " << int(b) << ", G: " << int(g) << ", R: " << int(r) << endl;
//            
//            
//            //points.push_back(img.at<Point2f>(i, j));
//            
//            count++;
//        }
//    }
    
    // CG - We divide by three so we are only returning the total count pixels (and not each of the CHANNELS within EACH PIXEL as well).
    return points;
}