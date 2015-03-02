#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <thread>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> ScanImagePointer(Mat inputMat, vector<Point2f>& points, vector<int>& rows, int patchSize = 10);
void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY);
void calcHistMatchScore(Mat localisedSearchWindow, Mat templatePatch, int hist_match_method, double& highScore, int& highScoreIndexY);
vector<double> runTestPatch(Mat image2ROI, int patchSize, int match_method, vector<Mat> patch_templates, vector<Point2f> patch_template_coords);
void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<string> methods, int patchSize, int roiSize, string fileNamePrefix = "result_patch_");
void exportTimeResults(vector<double> timeTaken, vector<string> methods, int patchSize, int roiSize, string fileNamePrefix = "time_test_");


int main(int argc, char** argv) {
    
    double negativeScalingFactor = 0.30;
    
    int histogramCompMethod = CV_COMP_CORREL;
    bool useRGB = false;
    int match_method = CV_TM_SQDIFF_NORMED;
    
    int patchPercentage = 40;
    int roiPercentage = 40;
    
    vector<int> result_rows;
    vector<Point2f> template_coords;
    vector<Mat> patch_templates;
    
    Mat img1ColourTransform, img2ColourTransform;
    
    if(argc != 3)
    {
        return -1;
    }
    
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }
    
    resize(img1, img1, Size(floor(img1.cols * negativeScalingFactor), floor(img1.rows * negativeScalingFactor)));
    resize(img2, img2, Size(floor(img2.cols * negativeScalingFactor), floor(img2.rows * negativeScalingFactor)));
    
    //BGR2HSV = Hue Range: 0-180
    //BGR2HSV_FULL = Hue Range: 0-360
    cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2HSV);
    cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2HSV);
    
    //CG - Calculate a central column through the two images that has a percentage width of the original images.
    double imageCentreX = img1ColourTransform.cols / 2;
    double imageROIWidth = img1ColourTransform.cols * ((double) roiPercentage / 100);
    double imageROIHalfWidth = imageROIWidth / 2;
    double imgROIStartX = imageCentreX - imageROIHalfWidth;
    double imgROIEndX = imageCentreX + imageROIHalfWidth;
    
    int patchSize = imageROIWidth * ((double) patchPercentage / 100);
    
    vector <vector<double > > all_results;
    vector <double> timeTaken;
    
    double testElaspedTime = 0;
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat image1ROI = img1ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
    
    Mat image2ROI = img2ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
    
    patch_templates = ScanImagePointer(image1ROI, template_coords, result_rows, patchSize);
    
    vector<string> methods {"sqdiffnormed", "coeffnormed"};
    
    
        //TEST1
    
        testElaspedTime = (double)getTickCount();
    
        all_results.push_back(runTestPatch(image2ROI, patchSize, CV_TM_SQDIFF_NORMED, patch_templates, template_coords));
    
        timeTaken.push_back(((double)getTickCount() - testElaspedTime)/getTickFrequency());
    
    
        //TEST2
    
        testElaspedTime = (double)getTickCount();
    
        all_results.push_back(runTestPatch(image2ROI, patchSize, CV_TM_CCOEFF_NORMED, patch_templates, template_coords));
    
        timeTaken.push_back(((double)getTickCount() - testElaspedTime)/getTickFrequency());
    
    
    //TEST3
    
    //    testElaspedTime = (double)getTickCount();
    //
    //    all_results.push_back(runTestPatch(image2ROI, patchSize, CV_TM_CCORR_NORMED, patch_templates, template_coords));
    //
    //    timeTaken.push_back(((double)getTickCount() - testElaspedTime)/getTickFrequency());
    
    
    exportResults(all_results, result_rows, methods, roiPercentage, patchPercentage);
    
    exportTimeResults(timeTaken, methods, patchPercentage, roiPercentage);
    
    return 0;
    
}

void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<string> methods, int patchSize, int roiSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << roiSize << "_" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    oStream << "descriptor image_row";
    
    for(std::vector<int>::size_type i = 0; i != methods.size(); i++) {
        
        oStream << " " << methods[i] << "_displacement";
    }
    
    oStream << "\n";
    
    cout << oStream.str();
    
    myfile << oStream.str();
    
    oStream.clear();
    oStream.str("");

    for (int i = 0; i < all_results[0].size(); i++) {
        
        oStream << rows[i];
        
        for(std::vector<double>::size_type j = 0; j != all_results.size(); j++) {
            
            
            //cout << all_results[j][i] << endl;
            
            oStream << " " << all_results[j][i];
        }
        
        oStream << "\n";
        
    }
    
    cout << oStream.str();
    
    myfile << oStream.str();
    
    //cout << "\n\nhjdhsj\n\n";
    
    //    for(std::vector<double>::size_type i = 0; i != all_results.size(); i++) {
    //
    //
    //        for (int j = 0; j < all_results[i].size(); i++) {
    //            cout << all_results[i][j] << endl;
    //        }
    //
    //    }
    
    //    std::vector< std::vector<double> >::const_iterator row;
    //    std::vector<double>::const_iterator col;
    //    for (row = all_results.begin(); row != all_results.end(); ++row) {
    //        for (col = row->begin(); col != row->end(); ++col) {
    //            std::cout << *col << endl;
    //        }
    //    }
    
    //cout<< endl;
    
    
    
    //        oStream << "\n";
    //
    //        //cout << oStream.str();
    //
    //        myfile << oStream.str();
    //
    //        oStream.clear();
    //        oStream.str("");
    
    
    myfile.close();
    
}

void exportTimeResults(vector<double> timeTaken, vector<string> methods, int patchSize, int roiSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << roiSize << "_" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    myfile << "descriptor match_method time_taken\n";
    
    for(std::vector<int>::size_type i = 0; i != timeTaken.size(); i++) {
        
        oStream << methods[i] << " " << timeTaken[i] << "\n";
        
        myfile << oStream.str();
        
        oStream.clear();
        oStream.str("");
    }
    
    myfile.close();
    
}

vector<double> runTestPatch(Mat image2ROI, int patchSize, int match_method, vector<Mat> patch_templates, vector<Point2f> patch_template_coords) {
    
    Mat localisedSearchWindow, templatePatch;
    vector<double> avg_result;
    vector<double> raw_result;
    
    vector<Mat>::iterator i1;
    vector<Point2f>::iterator i2;
    
    int patchCount = 0;
    int rowNumber = (patchSize / 2);
    
    
    for( i1 = patch_templates.begin(), i2 = patch_template_coords.begin(); i1 < patch_templates.end() && i2 < patch_template_coords.end(); ++i1, ++i2 )
    {
        
        double displacement, highestScore;
        
        templatePatch = (*i1);
        
        Point2f originPixelCoords = (*i2);
        
        if (rowNumber != originPixelCoords.y) {
            
            if (rowNumber == 55) {
                
                cout << "twat";
            }
            
            double sum = accumulate(raw_result.begin(), raw_result.end(), 0.0);
            double mean = sum / raw_result.size();
            
            avg_result.push_back(mean);
            
            //cout << avg_result[rowNumber];
            
            rowNumber = originPixelCoords.y;
            
            raw_result.clear();
            
        }
        
        int w = image2ROI.cols;
        int h = image2ROI.rows;
        
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - (originPixelCoords.y - (templatePatch.cols / 2));
        
        //localisedWindowHeight = localisedWindowHeight <= (patchSize * 2) ? localisedWindowHeight : (patchSize * 2);
        
        localisedSearchWindow = image2ROI(Rect(originPixelCoords.x - (templatePatch.cols / 2), originPixelCoords.y - (templatePatch.cols / 2), localisedWindowWidth, localisedWindowHeight));
        
//        imshow("shajadsh", templatePatch);
//        imshow("fhjdshf", localisedSearchWindow);
//        waitKey(1);
        
        // Constructs the new thread and runs it. Does not block execution.
        thread t1(calcPatchMatchScore, localisedSearchWindow, templatePatch, match_method, ref(highestScore), ref(displacement));
        
        t1.join();
        
        raw_result.push_back(displacement);
        
        patchCount++;
        
    }
    
    return avg_result;
    
}

// CG - Here, we are passing 'inputMat' and 'points' by REFERENCE, NOT BY VALUE.
vector<Mat> ScanImagePointer(Mat inputMat, vector<Point2f>& points, vector<int>& rows, int patchSize)
{
    
    CV_Assert(inputMat.depth() != sizeof(uchar));
    
    vector<Mat> mats;
    int nRows = inputMat.rows;
    int nCols = inputMat.cols;
    
    int i,j;
    uchar* p;
    
    //CG - Stop extracting patches when we get to the bottom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = (patchSize / 2); i < (nRows - (patchSize / 2)); i+=2)
    {
        
        p = inputMat.ptr<uchar>(i);
        rows.push_back(i);
        
        for (j = (patchSize / 2); j < (nCols - (patchSize / 2)); j+=2)
        {
            
            int x = j - (patchSize / 2);
            
            int y = i - (patchSize / 2);
            
            mats.push_back(inputMat(Rect(x,y,patchSize - 1 ,patchSize - 1)));
            
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


void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY) {
    
    Mat resultMat, currentPatch;
    
    //Set "initialiser" value for best match score.
    double bestScoreYIndex = -1;
    double bestScore = -1;
    
    bool stop = false;
    
    // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better.
    //double bestScore = match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED ? 100 : 0;
    
    for(int i = 0; i < localisedSearchWindow.rows - templatePatch.rows; i++)
    {
        currentPatch = localisedSearchWindow.clone();
        currentPatch = currentPatch(Rect(0, i, templatePatch.cols, templatePatch.rows));
        
        // Create the resultMat matrix
        int resultMat_cols =  currentPatch.cols - templatePatch.cols + 1;
        int resultMat_rows = currentPatch.rows - templatePatch.rows + 1;
        
        resultMat.create(resultMat_cols, resultMat_rows, CV_32FC1);
        
        // Do the Matching and Normalize
        matchTemplate( currentPatch, templatePatch, resultMat, match_method );
        
        // Localizing the best match with minMaxLoc
        double minVal, maxVal;
        
        // We do not need to pass in any 'Point' objects, as we are not interested in getting the "best match point" location back (as the 'result' matrix is only 1px x 1px in size).
        minMaxLoc( resultMat, &minVal, &maxVal, NULL, NULL, Mat() );
        
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