#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> ScanImagePointer(Mat inputMat, vector<Point2f> &points, int patchSize = 10);
void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, int& highScoreIndexY);

ofstream myfile;

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
    
    int negativeScalingFactor = 4;
    int patchSize = 100/negativeScalingFactor;
    int histogramCompMethod = CV_COMP_CORREL;
    bool useRGB = false;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    resize(img1, img1, Size(img1.cols/negativeScalingFactor, img1.rows/negativeScalingFactor));
    resize(img2, img2, Size(img2.cols/negativeScalingFactor, img2.rows/negativeScalingFactor));
    
    Mat result, result2, img1ColourTransform, img2ColourTransform, localisedSearchWindow, templatePatch;
    
    // cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2GRAY);
    // cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2GRAY);
    
    
    if (useRGB) {
        
        //cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2RGB);
        //cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2RGB);
        
        cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2GRAY);
        cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2GRAY);
        
        //img1ColourTransform = img1.clone();
        //img2ColourTransform = img2.clone();
        
    } else {
        
        //BGR2HSV = Hue Range: 0-180
        //BGR2HSV_FULL = Hue Range: 0-360
        cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2HSV);
        cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2HSV);
        
    }
    
    //CG - Calculate a central column through the two images that has a percentage width of the original images.
    double imageCentreX = img1ColourTransform.cols / 2;
    double imageROIWidth = img1ColourTransform.cols * 0.3;
    double imageROIHalfWidth = imageROIWidth / 2;
    double imgROIStartX = imageCentreX - imageROIHalfWidth;
    double imgROIEndX = imageCentreX + imageROIHalfWidth;
    
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat image1ROI = img1ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
    Mat image2ROI = img2ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img2ColourTransform.rows) );
    
    Mat opticalFlow = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    
    vector<Point2f> points;
    
    vector<Mat> test = ScanImagePointer(image1ROI, points, patchSize);
    
    std::vector<Mat>::iterator i1;
    std::vector<Point2f>::iterator i2;
    
    
    //myfile.open ("example.dat", ios::out | ios::trunc);
    
    myfile.open ("example2.dat", ios::out | ios::trunc);
    
    
    int txtcount = 0;
    
    for( i1 = test.begin(), i2 = points.begin(); i1 < test.end() && i2 < points.end(); ++i1, ++i2 )
    {
        
//        if (txtcount >=1) {
//            break;
//        }
        
        int val;
        double bestVal;
        
        Point2f originPixelCoords = (*i2);
        
        templatePatch = *i1;
        
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - originPixelCoords.y;
        
        localisedSearchWindow = image2ROI(Rect(originPixelCoords.x, originPixelCoords.y, localisedWindowWidth, localisedWindowHeight) );
        
        myfile << "descriptor x_" << txtcount << " y_" << txtcount <<"\n";
        
        //CG - Calculate the match score between each patch, and return the row index of the TOP-LEFT corner for the patch that returned the highest match score.
        calcPatchMatchScore(localisedSearchWindow, templatePatch, CV_TM_SQDIFF_NORMED, bestVal, val);
        
        myfile << "\n";
        
        //Print out the "best score" for the matching function, and the row index from wihtin the localised search window.
        cout << "MAX POSITION F0R " << txtcount << ": " << val << " - " << bestVal << endl;
        
        cout << "DISPLACEMENT FOR: " << txtcount << ": " << val << "\n" << endl;
        
//        circle(img1, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y),2, Scalar(255,255,255), CV_FILLED, 8,0);
//        
//        circle(img2, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y + val),2, Scalar(255,255,255), CV_FILLED, 8,0);
//        
//        rectangle( img1, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y), Point( (imgROIStartX + originPixelCoords.x + templatePatch.cols), originPixelCoords.y + templatePatch.rows ), Scalar(255, 0, 0), 2, 8, 0 );
//        
//        rectangle( img2, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y + val), Point(imgROIStartX + originPixelCoords.x + templatePatch.cols ,originPixelCoords.y + templatePatch.rows + val), Scalar(255, 0, 255), 2, 8, 0 );
        
        txtcount++;
        
    }
    
    // CG - Draw main ROI windows.
    rectangle( img1, Point(imgROIStartX, 0), Point(imgROIEndX , img1.rows), Scalar(0, 0, 255), 2, 8, 0 );
    rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    imshow("Image 1", img1);
    
    imshow("Image 2", img2);
    
    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
    
    imshow("Merged Result", img2);
    
    myfile.close();
    
    imshow("Search Window", localisedSearchWindow);
    
    imshow("Template", templatePatch);
    
    waitKey(0);
    
    return 0;
    
}

// CG - Here, we are passing 'inputMat' and 'points' by REFERENCE, NOT BY VALUE.
vector<Mat> ScanImagePointer(Mat inputMat, vector<Point2f> &points, int patchSize)
{
    
    CV_Assert(inputMat.depth() != sizeof(uchar));
    
    vector<Mat> mats;
    int nRows = inputMat.rows;
    int nCols = inputMat.cols;
    
    int i,j;
    uchar* p;
    
    //CG - Stop extracting patches when we get 2*patchsize rows away from the bottom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = 0; i < nRows - patchSize * 2; i++)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for (j = 0; j < nCols - patchSize; j++)
        {
            mats.push_back(inputMat(Rect(j,i,patchSize,patchSize)));
            
            //CG - Same as Point2f (typename alias)
            points.push_back(Point_<float>(j, i));
            
        }
        
    }
    
    return mats;
}


void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, int& highScoreIndexY) {
    
    Mat result, currentPatch;
    
    //Set "initialiser" value for best match score.
    int bestScoreYIndex = -1;
    double bestScore = -1;
    
    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better.
    //double bestScore = match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED ? 100 : 0;
    
    for(int i = 0; i < localisedSearchWindow.rows - templatePatch.rows; i++)
    {
        currentPatch = localisedSearchWindow.clone();
        currentPatch = currentPatch(Rect(0, i, templatePatch.cols, templatePatch.rows));
        
        // Create the result matrix
        int result_cols =  currentPatch.cols - templatePatch.cols + 1;
        int result_rows = currentPatch.rows - templatePatch.rows + 1;
        
        result.create(result_cols, result_rows, CV_32FC1);
        
        // Do the Matching and Normalize
        matchTemplate( currentPatch, templatePatch, result, match_method );
        
        // Localizing the best match with minMaxLoc
        double minVal, maxVal;
        
        // We do not need to pass in any 'Point' objects, as we are not interested in getting the "best match point" location back (as the 'result' matrix is only 1px x 1px in size).
        minMaxLoc( result, &minVal, &maxVal, NULL, NULL, Mat() );
        
        // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better.
        if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
        {
            
            if (bestScore == -1 || minVal < bestScore) {
                
                bestScore = minVal;
                bestScoreYIndex = i;
                
            }
            
            //cout << "I: " << i << " - Min Result: " << minVal << endl;
            myfile << i << " " <<  minVal << "\n";
        }
        else
        {
            
            if (bestScore == -1 || maxVal > bestScore) {
                
                bestScore = maxVal;
                bestScoreYIndex = i;
                
            }
            
            //cout << "I: " << i << " - Max Result: " << maxVal << endl;
            myfile << i << " " <<  maxVal << "\n";
        }
        
        
        //CG - Allows the output window displaying the current patch to be updated automatically.
        //cv::waitKey(10);
        //resize(result, result, Size(result.cols*10, result.rows*10));
        //imshow("Normalised RESULT", result);
        
    }
    
    highScore = bestScore;
    highScoreIndexY = bestScoreYIndex;
    
}