#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <thread>
#include <mutex>          // std::mutex

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

std::mutex mtx;           // mutex for critical section

vector<Mat> ScanImagePointer(Mat inputMat, vector<Point2f> &points, int patchSize = 10);
void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, int& highScoreIndexY);
void calcHistMatchScore(Mat localisedSearchWindow, Mat templatePatch, int hist_match_method, double& highScore, int& highScoreIndexY);

ofstream myfile, myfilehist;

static void help()
{
    printf("Usage: ./optical_flow_lk_image_template_match <image1> <image2>\n");
}

static void addToVec() {
    
    mtx.lock();
    
    mtx.unlock();
}

int main(int argc, char** argv) {
    
    if(argc != 3)
    {
        help();
        return -1;
    }
    
    double negativeScalingFactor = 0.40;
    
    double t = (double)getTickCount();
    
    int histogramCompMethod = CV_COMP_CORREL;
    bool useRGB = false;
    int match_method = CV_TM_SQDIFF_NORMED;
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    resize(img1, img1, Size(floor(img1.cols * negativeScalingFactor), floor(img1.rows * negativeScalingFactor)));
    resize(img2, img2, Size(floor(img2.cols * negativeScalingFactor), floor(img2.rows * negativeScalingFactor)));
    
    Mat result, result2, img1ColourTransform, img2ColourTransform, localisedSearchWindow, templatePatch;
    
    // cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2GRAY);
    // cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2GRAY);
    
    
    if (useRGB) {
        
        
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
    double imageROIWidth = img1ColourTransform.cols * 0.4;
    double imageROIHalfWidth = imageROIWidth / 2;
    double imgROIStartX = imageCentreX - imageROIHalfWidth;
    double imgROIEndX = imageCentreX + imageROIHalfWidth;
    
    int patchSize = imageROIWidth * 0.2;
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat image1ROI = img1ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
    Mat image2ROI = img2ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img2ColourTransform.rows) );
    
    Mat opticalFlow = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    
    vector<Point2f> points;
    
    vector<Mat> test = ScanImagePointer(image1ROI, points, patchSize);
    
    std::vector<Mat>::iterator i1;
    std::vector<Point2f>::iterator i2;
    
    
    //myfile.open ("example.dat", ios::out | ios::trunc);
    
    myfile.open ("displacement_patch_20_20.dat", ios::out | ios::trunc);
    myfilehist.open ("displacement_hist_20_20.dat", ios::out | ios::trunc);
    
    int patchCount = 0;
    int rowNumber = 0;
    
    vector<double> v;
    vector<double> v_hist;
    
    myfile << "descriptor y displacement\n";
    myfilehist << "descriptor y displacement\n";
    
    for( i1 = test.begin(), i2 = points.begin(); i1 < test.end() && i2 < points.end(); ++i1, ++i2 )
    {
        
        int val, valHist;
        double bestVal, bestValHist;
        
        Point2f originPixelCoords = (*i2);
        
        if (rowNumber != originPixelCoords.y) {
            
            double sum = accumulate(v.begin(), v.end(), 0.0);
            double mean = sum / v.size();
            
            double sum_hist = accumulate(v_hist.begin(), v_hist.end(), 0.0);
            double mean_hist = sum_hist / v_hist.size();
            
            myfile << rowNumber << " " << mean << "\n";
            myfilehist << rowNumber << " " << mean_hist << "\n";
            
            rowNumber = originPixelCoords.y;
            
            v.clear();
            v_hist.clear();
            
        }
        
//        if (rowNumber >=1) {
//            
//            break;
//        }
        
        templatePatch = (*i1);
        
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - originPixelCoords.y;
        
        localisedWindowHeight = localisedWindowHeight <= (patchSize * 2) ? localisedWindowHeight : (patchSize * 2);
        
        localisedSearchWindow = image2ROI(Rect(originPixelCoords.x, originPixelCoords.y, localisedWindowWidth, localisedWindowHeight) );
        
        //CG - Calculate the match score between each patch, and return the row index of the TOP-LEFT corner for the patch that returned the highest match score.
        
        // Constructs the new thread and runs it. Does not block execution.
        thread t1(calcPatchMatchScore, localisedSearchWindow, templatePatch, match_method, ref(bestVal), ref(val));
        
        // Constructs the new thread and runs it. Does not block execution.
        thread t2(calcHistMatchScore, localisedSearchWindow, templatePatch, histogramCompMethod, ref(bestValHist), ref(valHist));
        
        //calcPatchMatchScore(localisedSearchWindow, templatePatch, match_method, bestVal, val);
        
       // calcHistMatchScore(localisedSearchWindow, templatePatch, histogramCompMethod, bestValHist, valHist);
        
        t1.join();
        t2.join();
        
        v.push_back(val);
        v_hist.push_back(valHist);
        
//        circle(img1, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y),2, Scalar(255,255,255), CV_FILLED, 8,0);
//        circle(img2, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y + val),2, Scalar(255,255,255), CV_FILLED, 8,0);
//////        
//       rectangle( img1, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y), Point( (imgROIStartX + originPixelCoords.x + templatePatch.cols), originPixelCoords.y + templatePatch.rows ), Scalar(255, 0, 0), 2, 8, 0 );
//////        
//       rectangle( img2, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y + val), Point(imgROIStartX + originPixelCoords.x + templatePatch.cols ,originPixelCoords.y + templatePatch.rows + val), Scalar(255, 0, 255), 2, 8, 0 );
////        
        
//       cv::waitKey(1);
//        imshow("tempy", templatePatch);
        
        // Increment the patch count, as we are now moving across.
        patchCount++;
        
    }
    
    myfile.close();
    myfilehist.close();
    
    
////    // CG - Draw main ROI windows.
//    rectangle( img1, Point(imgROIStartX, 0), Point(imgROIEndX , img1.rows), Scalar(0, 0, 255), 2, 8, 0 );
//    rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
////    
////    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
//  double alpha = 0.4;
////    
//    imshow("Image 1", img1);
////    
//    imshow("Image 2", img2);
////    
//    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
////    
//    imshow("Merged Result", img2);
////    
//    imshow("Search Window", localisedSearchWindow);
////    
////    //CG - Should be 160 instead of 161 for row 1, as 160 is the last patch in row 1, before being incremented to row 2.
////    //imshow("blah", test.at(160));
////    
//    imshow("template", templatePatch);
////
//    waitKey(0);
    
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;
    
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
    
    //CG - Stop extracting patches when we get to the bottom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = 0; i < nRows - patchSize * 2; i+=2)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        for (j = 0; j < nCols - patchSize; j+=2)
        {
            mats.push_back(inputMat(Rect(j,i,patchSize,patchSize)));
            
            //CG - Same as Point2f (typename alias)
            points.push_back(Point_<float>(j, i));
            
        }
        
    }
    
    return mats;
}

void calcHistMatchScore(Mat localisedSearchWindow, Mat templatePatch, int hist_match_method, double& highScore, int& highScoreIndexY) {
    
    Mat hist_template, hist_current, currentPatch;
    
    //Set "initialiser" value for best match score.
    int bestScoreYIndex = -1;
    double bestScore = -1;
    
    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    
    const float* ranges[] = { h_ranges, s_ranges };
    
    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };
    
    bool stop = false;
    
    for(int i = 0; i < localisedSearchWindow.rows - templatePatch.rows; i++)
    {
        
        currentPatch = localisedSearchWindow.clone();
        currentPatch = currentPatch(Rect(0, i, templatePatch.cols, templatePatch.rows));
        
        /// Calculate the histogram for the template patch image.
        calcHist( &templatePatch, 1, channels, Mat(), hist_template, 2, histSize, ranges, true, false );
        normalize( hist_template, hist_template, 0, 1, NORM_MINMAX, -1, Mat() );
        
        calcHist( &currentPatch, 1, channels, Mat(), hist_current, 2, histSize, ranges, true, false );
        normalize( hist_current, hist_current, 0, 1, NORM_MINMAX, -1, Mat() );
        
        double histresult = compareHist( hist_current, hist_template, hist_match_method );
        
        //cout << "I: " << i << " - Result: " << histresult << endl;
        
        if( hist_match_method  == CV_COMP_BHATTACHARYYA || hist_match_method == CV_COMP_CHISQR )
        {
            
            if (bestScore == -1 || histresult < bestScore) {
                
                bestScore = histresult;
                bestScoreYIndex = i;
                
            } else {
                
                stop = true;
            }
            
        }
        else
        {
            
            if (bestScore == -1 || histresult > bestScore) {
                
                bestScore = histresult;
                bestScoreYIndex = i;
                
            } else {
                
                stop = true;
                
            }
            
        }
        
        if (stop) {
            
            break;
            
        }
        
    }
    
    highScore = bestScore;
    highScoreIndexY = bestScoreYIndex;
    
}


void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, int& highScoreIndexY) {
    
    Mat result, currentPatch;
    
    //Set "initialiser" value for best match score.
    int bestScoreYIndex = -1;
    double bestScore = -1;
    
    bool stop = false;
    
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
                
            } else {
                
                stop = true;
            }
            
            //cout << "I: " << i << " - Min Result: " << minVal << endl;
            //myfile << i << " " <<  minVal << "\n";
        }
        else
        {
            
            if (bestScore == -1 || maxVal > bestScore) {
                
                bestScore = maxVal;
                bestScoreYIndex = i;
                
            } else {
                
                stop = true;
            }
            
            //cout << "I: " << i << " - Max Result: " << maxVal << endl;
            //myfile << i << " " <<  maxVal << "\n";
        }
        
        
        //CG - Allows the output window displaying the current patch to be updated automatically.
        //cv::waitKey(10);
        //resize(result, result, Size(result.cols*10, result.rows*10));
        //imshow("Normalised RESULT", result);
        
        if (stop) {
            
            break;
            
        }
        
    }
    
    highScore = bestScore;
    highScoreIndexY = bestScoreYIndex;
    
}