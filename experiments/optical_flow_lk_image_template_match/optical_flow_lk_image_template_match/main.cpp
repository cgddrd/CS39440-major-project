#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <fstream>
#include <numeric>

#include <cmath>

#include "TemplateMatching.h"
#include "Utils.h"

using namespace cv;
using namespace std;

vector<Mat> getROIPatches(Mat inputMat, vector<Point2f>& points, vector<int>& rows, int patchSize = 10);
void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY);
void calcPatchMatchScore2(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY);
void calcHistMatchScore(Mat localisedSearchWindow, Mat templatePatch, int hist_match_method, double& highScore, int& highScoreIndexY);
vector<double> runTestPatch(Mat image2ROI, int patchSize, int match_method, vector<Mat> patch_templates, vector<Point2f> patch_template_coords, string histFileName);
void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<int> methods, int roiSize, int patchSize, string fileNamePrefix = "result_patch_");
void exportTimeResults(vector<double> timeTaken, vector<int> methods, int roiSize, int patchSize, string fileNamePrefix = "time_test_");
string getMatchMethodName(int matchMethod);
void startTests(Mat img1ColourTransform, Mat img2ColourTransform, vector<int> roiDimensions, vector<int> patchDimensions, vector<int> match_type, int pairNo);
vector<int> calcHistogram(double bucketSize, vector<double> values, double maxVal);

Mat img1;
Mat img2;
double imgROIStartX = 0;

bool simplePatches = false;
bool useGUI = false;
bool exhaustiveSearch = false;

enum
{
    CUSTOM_NORM =6,
    CUSTOM_CORR =7
};

int main(int argc, char** argv) {
    
    Mat img1ColourTransform, img2ColourTransform;
    
    if(argc != 3)
    {
        return -1;
    }
    
    vector<int> methods {CUSTOM_NORM};
    
    string fileRootPath = "../../../eval_data/motion_images/";
    
    //vector<vector<string>> files {{"1.JPG", "2.JPG"}, {"3.JPG", "4.JPG"}, {"5.JPG", "6.JPG"}, {"7.JPG", "8.JPG"}, {"9.JPG", "10.JPG"}, {"11.JPG", "12.JPG"}};
    
    vector<vector<string>> files {{"1.JPG", "2.JPG"}};

    vector<int> patchSizes{50};
    
    vector<int> roiSizes {40};
    
    double totalElaspedTime = (double)getTickCount();
    
    int pairNo = 1;
    
    for(vector<vector<string>>::iterator it = files.begin(); it != files.end(); ++it) {
        
        vector<string> currentFilePair = (*it);
        
        img1 = imread(fileRootPath + currentFilePair[0], CV_LOAD_IMAGE_COLOR);
        img2 = imread(fileRootPath + currentFilePair[1], CV_LOAD_IMAGE_COLOR);
        
        if(img1.empty() || img2.empty())
        {
            printf("Can't read one of the images\n");
            return -1;
        }
        
        cout << "Image Size: " << img1.cols << "px x " << img1.rows << "px.\n";
        
        //BGR2HSV = Hue Range: 0-180
        //BGR2HSV_FULL = Hue Range: 0-360
        cvtColor(img1, img1ColourTransform, cv::COLOR_BGR2HSV);
        cvtColor(img2, img2ColourTransform, cv::COLOR_BGR2HSV);
        
        Mat hsvChannelsImg1[3], hsvChannelsImg2[3];
        
        split(img1ColourTransform, hsvChannelsImg1);
        split(img2ColourTransform, hsvChannelsImg2);
        
        //Set VALUE channel to 0
        hsvChannelsImg1[2]=Mat::zeros(img1ColourTransform.rows, img1ColourTransform.cols, CV_8UC1);
        hsvChannelsImg2[2]=Mat::zeros(img2ColourTransform.rows, img2ColourTransform.cols, CV_8UC1);
        
        merge(hsvChannelsImg1,3,img1ColourTransform);
        merge(hsvChannelsImg2,3,img2ColourTransform);
        
//        startTests(img1ColourTransform, img2ColourTransform, roiSizes, patchSizes, methods, pairNo);
        
        double test1 = (TemplateMatching::calcSSD(img1ColourTransform, img2ColourTransform) / TemplateMatching::calcNormalisationFactorLoop(img1ColourTransform, img2ColourTransform));
        
        double test2 = (TemplateMatching::calcSSD(img1ColourTransform, img2ColourTransform) / TemplateMatching::calcNormalisationFactorLoop2(img1ColourTransform, img2ColourTransform));
        
        cout << "1: " << test1 << "\n2: " << test2 << endl;
        
        pairNo++;
        
        
    }
    
    cout << "\n\n**********************\nTEST END: Time for entire test (secs): " << (((double)getTickCount() - totalElaspedTime)/getTickFrequency()) << endl;
    
    return 0;
    
}

string getMatchMethodName(int matchMethod) {
    
    switch (matchMethod) {
        case CV_TM_SQDIFF_NORMED:
            return "SQDIFF_NORMED";
        case CV_TM_SQDIFF:
            return "SQDIFF";
        case CV_TM_CCORR_NORMED:
            return "CCORR_NORMED";
        case CV_TM_CCORR:
            return "CCORR";
        case CV_TM_CCOEFF_NORMED:
            return "CCOEFF_NORMED";
        case CV_TM_CCOEFF:
            return "CCOEFF";
        case CUSTOM_NORM:
            return "NORM";
        case CUSTOM_CORR:
            return "CORR";
        default:
            return "UNKNOWN";
    }
}

void startTests(Mat img1ColourTransform, Mat img2ColourTransform, vector<int> roiDimensions, vector<int> patchDimensions, vector<int> match_type, int pairNo) {
    
    int testCount = 1;
    
    for(vector<int>::iterator it1 = roiDimensions.begin(); it1 != roiDimensions.end(); ++it1) {
        
        //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
        double imageCentreX = img1ColourTransform.cols / 2;
        double imageROIWidth = img1ColourTransform.cols * ((double) *it1 / 100);
        double imageROIHalfWidth = imageROIWidth / 2;
        imgROIStartX = imageCentreX - imageROIHalfWidth;
        double imgROIEndX = imageCentreX + imageROIHalfWidth;
        
        Mat image1ROI = img1ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
        Mat image2ROI = img2ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img2ColourTransform.rows) );
        
        if (useGUI) {
            rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
        }
        
        for(vector<int>::iterator it2 = patchDimensions.begin(); it2 != patchDimensions.end(); ++it2) {
            
            vector<int> result_rows;
            vector<Point2f> template_coords;
            vector<Mat> patch_templates = getROIPatches(image1ROI, template_coords, result_rows, *it2);
            
            vector <vector<double> > allResults;
            vector <double> timeDurations;
            
            
            for(vector<int>::iterator it3 = match_type.begin(); it3 != match_type.end(); ++it3) {
                
                cout << "BEGIN: Test #" << testCount << ": ROI Size: " << *it1 << ", Patch Size: " << *it2 << ", Match Method: " << getMatchMethodName(*it3) << ", Pair No: " << pairNo << endl;
                
                cout << "ROI Size: " << image2ROI.cols << "px x " << image2ROI.rows << "px" << endl;
                
                double testElaspedTime = (double)getTickCount();
                
                ostringstream oStream;
                
                oStream << "hist_" << *it1 << "_" << *it2 << "_" << getMatchMethodName(*it3) << ".dat";
                
                allResults.push_back(runTestPatch(image2ROI, *it2, *it3, patch_templates, template_coords, oStream.str()));
                
                timeDurations.push_back(((double)getTickCount() - testElaspedTime)/getTickFrequency());
                
                cout << "END: Test #" << testCount << "\n\n";
                
                testCount++;
                
                
            }
            
            std::stringstream sstm;
            sstm << "result_pair" << pairNo;
            
            exportResults(allResults, result_rows, match_type, *it1, *it2, sstm.str());
            
            sstm.str("");
            sstm.clear();
            
            sstm << "tresult_pair" << pairNo;
            
            exportTimeResults(timeDurations, match_type, *it1, *it2, sstm.str());
            
            sstm.str("");
            sstm.clear();
            
        }
        
        
    }
    
}

void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<int> methods, int roiSize, int patchSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << "_roi" << roiSize << "_patch" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    oStream << "descriptor row";
    
    for(std::vector<int>::size_type i = 0; i != methods.size(); i++) {
        
        oStream << " " << getMatchMethodName(methods[i]);
    }
    
    oStream << "\n";
    
    myfile << oStream.str();
    
    oStream.clear();
    oStream.str("");
    
    for (int i = 0; i < all_results[0].size(); i++) {
        
        oStream << rows[i];
        
        for(std::vector<double>::size_type j = 0; j != all_results.size(); j++) {
            
            oStream << " " << all_results[j][i];
        }
        
        oStream << "\n";
        
    }
    
    myfile << oStream.str();
    
    myfile.close();
    
}

void exportTimeResults(vector<double> timeTaken, vector<int> methods, int roiSize, int patchSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << "_roi" << roiSize << "_patch" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    myfile << "descriptor method time\n";
    
    for(std::vector<int>::size_type i = 0; i != timeTaken.size(); i++) {
        
        oStream << getMatchMethodName(methods[i]) << " " << timeTaken[i] << "\n";
        
        myfile << oStream.str();
        
        oStream.clear();
        oStream.str("");
    }
    
    myfile.close();
    
}

vector<int> calcHistogram(double bucketSize, vector<double> values, double maxVal) {
    
    int number_of_buckets = (int)ceil(maxVal / bucketSize);
    
    vector<int> histogram(number_of_buckets);
    
    for(vector<double>::const_iterator it = values.begin(); it != values.end(); ++it) {
        
        int bucket = (int)floor(*it / bucketSize);
        histogram[bucket] += 1;
        
    }
    
    return histogram;
    
}

vector<double> runTestPatch(Mat image2ROI, int patchSize, int match_method, vector<Mat> patch_templates, vector<Point2f> patch_template_coords, string histFileName) {
    
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
        
        if (useGUI) {
            rectangle( img2, Point(imgROIStartX + originPixelCoords.x - (patchSize / 2), originPixelCoords.y - (patchSize / 2)), Point(imgROIStartX + originPixelCoords.x + (patchSize / 2) , originPixelCoords.y + (patchSize / 2)), Scalar(0, 0, 255), 2, 8, 0 );
        }
        
        //Once we reach the end of the row, we need to calculate the average displacement across the entire row.
        if (rowNumber != originPixelCoords.y) {
            
            if (!raw_result.empty()) {
                
                //vector<double> filtered = Utils::filterOutliers(raw_result);
                
                double mean = Utils::calcMean(raw_result);
                // double sdDev = Utils::calcStandardDeviation(filtered);
                
                avg_result.push_back(mean);
                
            } else {
                
                cout << "ERROR: No displacement values obtained for Row #: " << rowNumber;
                
            }
            
            rowNumber = originPixelCoords.y;
            
            raw_result.clear();
            
        }
        
        /* if(rowNumber >=(patchSize / 2) + 1) {
         break;
         } */
        
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - (originPixelCoords.y - (templatePatch.cols / 2));
        
        localisedSearchWindow = image2ROI(Rect(originPixelCoords.x - (templatePatch.cols / 2), originPixelCoords.y - (templatePatch.cols / 2), localisedWindowWidth, localisedWindowHeight));
        
        calcPatchMatchScore(localisedSearchWindow, templatePatch, match_method, highestScore, displacement);
        
        if (useGUI) {
            
            rectangle( img2, Point(imgROIStartX + originPixelCoords.x - (patchSize / 2), originPixelCoords.y - (patchSize / 2) + displacement), Point(imgROIStartX + originPixelCoords.x + (patchSize / 2) , originPixelCoords.y + (patchSize / 2) + displacement), Scalar(0, 255, 0), 2, 8, 0 );
            
            //CG - Allows the output window displaying the current patch to be updated automatically.
            cv::waitKey(10);
            imshow("Search Window", localisedSearchWindow);
            imshow("Template Patch", templatePatch);
            
            
        }
        
        if (displacement <= (templatePatch.rows*2)) {
            // break;
            raw_result.push_back(displacement);
        }
        
        patchCount++;
        
    }
    
    return avg_result;
    
}

vector<Mat> getROIPatches(Mat inputMat, vector<Point2f>& points, vector<int>& rows, int patchSize)
{
    
    CV_Assert(inputMat.depth() != sizeof(uchar));
    
    vector<Mat> mats;
    int nRows = inputMat.rows;
    int nCols = inputMat.cols;
    int halfPatchSize = (patchSize / 2);
    
    int increment = simplePatches ? patchSize : 1;
    
    int i,j;
    uchar* p;
    
    //CG - Stop extracting patches when we get to the bottom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = halfPatchSize; i < (nRows - halfPatchSize); i+=increment)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        //CG - Push back the current row number (used for printing results later on).
        rows.push_back(i);
        
        for (j = halfPatchSize; j < (nCols - halfPatchSize); j+=increment)
        {
            
            int x = j - halfPatchSize;
            
            int y = i - halfPatchSize;
            
            mats.push_back(inputMat(Rect(x,y,patchSize,patchSize)));
            
            //CG - Same as Point2f (typename alias)
            points.push_back(Point_<float>(j, i));
            
        }
        
    }
    
    return mats;
}

void calcPatchMatchScore2(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY) {
    
    Mat resultMat, currentPatch;
    
    //Set "initialiser" value for best match score.
    double bestScoreYIndex = -1;
    double bestScore = -1;
    
    // Create the resultMat matrix
    int resultMat_cols =  localisedSearchWindow.cols - templatePatch.cols + 1;
    int resultMat_rows = localisedSearchWindow.rows - templatePatch.rows + 1;
    
    resultMat.create(resultMat_cols, resultMat_rows, CV_32FC1);
    
    //Do the Matching and Normalize
    matchTemplate( localisedSearchWindow, templatePatch, resultMat, match_method );
    normalize( resultMat, resultMat, 0, 1, NORM_MINMAX, -1, Mat() );
    
    //Localizing the best match with minMaxLoc
    double minVal, maxVal;
    Point minLoc; Point maxLoc;
    Point matchLoc;
    
    //We do not need to pass in any 'Point' objects, as we are not interested in getting the "best match point" location back (as the 'result' matrix is only 1px x 1px in size).
    minMaxLoc( resultMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    
    if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    {
        
        matchLoc = minLoc;
        
        bestScore = minVal;
        
        bestScoreYIndex = matchLoc.y;
        
    }
    else
    {
        matchLoc = maxLoc;
        
        bestScore = maxVal;
        
        bestScoreYIndex = matchLoc.y;
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
    
    if (!exhaustiveSearch) {
        
        for(int i = 0; i < localisedSearchWindow.rows - templatePatch.rows; i++)
        {
            currentPatch = localisedSearchWindow.clone();
            currentPatch = currentPatch(Rect(0, i, templatePatch.cols, templatePatch.rows));
            
            // Create the resultMat matrix
            int resultMat_cols =  currentPatch.cols - templatePatch.cols + 1;
            int resultMat_rows = currentPatch.rows - templatePatch.rows + 1;
            
            resultMat.create(resultMat_cols, resultMat_rows, CV_32FC1);
            
            if (match_method == CUSTOM_NORM) {
                
                //Calculate Euclidean Distance.
                double result = TemplateMatching::calcEuclideanDistanceNorm(templatePatch, currentPatch);
                
                if (bestScore == -1 || result < bestScore) {
                    
                    bestScore = result;
                    bestScoreYIndex = i;
                    
                } else {
                    
                    stop = true;
                    
                }
                
            } else if (match_method == CUSTOM_CORR) {
                
                //Calculate Euclidean Distance.
                double result = TemplateMatching::calcCorrelaton(templatePatch, currentPatch);
                
                if (bestScore == -1 || result > bestScore) {
                    
                    bestScore = result;
                    bestScoreYIndex = i;
                    
                } else {
                    
                    stop = true;
                    
                }
                
            } else {
                
                //Do the Matching and Normalize
                matchTemplate( currentPatch, templatePatch, resultMat, match_method );
                
                //Localizing the best match with minMaxLoc
                double minVal, maxVal;
                
                //We do not need to pass in any 'Point' objects, as we are not interested in getting the "best match point" location back (as the 'result' matrix is only 1px x 1px in size).
                minMaxLoc( resultMat, &minVal, &maxVal, NULL, NULL, Mat() );
                
                //For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better.
                if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
                {
                    
                    if (bestScore == -1 || minVal < bestScore) {
                        
                        bestScore = minVal;
                        bestScoreYIndex = i;
                        
                    } else {
                        
                        stop = true;
                    }
                }
                else
                {
                    
                    if (bestScore == -1 || maxVal > bestScore) {
                        
                        bestScore = maxVal;
                        bestScoreYIndex = i;
                        
                    } else {
                        
                        stop = true;
                    }
                    
                }
                
            }
            
            if (stop) {
                
                break;
                
            }
            
        }
        
        highScore = bestScore;
        highScoreIndexY = bestScoreYIndex;
        
    } else {
        
        calcPatchMatchScore2(localisedSearchWindow, templatePatch, match_method, highScore, highScoreIndexY);
        
    }
    
}