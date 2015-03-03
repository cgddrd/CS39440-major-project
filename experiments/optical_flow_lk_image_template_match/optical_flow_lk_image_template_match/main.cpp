#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <thread>

#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<Mat> getROIPatches(Mat inputMat, vector<Point2f>& points, vector<int>& rows, int patchSize = 10);
void calcPatchMatchScore(Mat localisedSearchWindow, Mat templatePatch, int match_method, double& highScore, double& highScoreIndexY);
void calcHistMatchScore(Mat localisedSearchWindow, Mat templatePatch, int hist_match_method, double& highScore, int& highScoreIndexY);
vector<double> runTestPatch(Mat image2ROI, int patchSize, int match_method, vector<Mat> patch_templates, vector<Point2f> patch_template_coords);
void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<int> methods, int patchSize, int roiSize, string fileNamePrefix = "result_patch_");
void exportTimeResults(vector<double> timeTaken, vector<int> methods, int patchSize, int roiSize, string fileNamePrefix = "time_test_");
string getMatchMethodName(int matchMethod);
void startTests(Mat img1ColourTransform, Mat img2ColourTransform, vector<int> roiDimensions, vector<int> patchDimensions, vector<int> match_type);
vector<int> calcHistogram(double bucketSize, vector<double> values, double maxVal);

Mat img1;
Mat img2;
double imgROIStartX = 0;


int main(int argc, char** argv) {
    
    double negativeScalingFactor = 0.30;
    
    Mat img1ColourTransform, img2ColourTransform;
    
    if(argc != 3)
    {
        return -1;
    }

    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    
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
    
    vector<int> methods {CV_TM_SQDIFF_NORMED};
    
    vector<int> patchSizes{40};
    
    vector<int> roiSizes {40};
    
    double totalElaspedTime = (double)getTickCount();
    
    startTests(img1ColourTransform, img2ColourTransform, roiSizes, patchSizes, methods);
    
    cout << "\n\n**********************\nTEST END: Time for entire test (secs): " << (((double)getTickCount() - totalElaspedTime)/getTickFrequency()) << endl;
    
    imshow("hjhd", img2);
    waitKey();
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
        default:
            return "UNKNOWN";
    }
}

void startTests(Mat img1ColourTransform, Mat img2ColourTransform, vector<int> roiDimensions, vector<int> patchDimensions, vector<int> match_type) {
    
    int testCount = 0;
    
    for(vector<int>::iterator it1 = roiDimensions.begin(); it1 != roiDimensions.end(); ++it1) {
        
        //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
        double imageCentreX = img1ColourTransform.cols / 2;
        double imageROIWidth = img1ColourTransform.cols * ((double) *it1 / 100);
        double imageROIHalfWidth = imageROIWidth / 2;
        imgROIStartX = imageCentreX - imageROIHalfWidth;
        double imgROIEndX = imageCentreX + imageROIHalfWidth;
        
        Mat image1ROI = img1ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
        Mat image2ROI = img2ColourTransform( Rect(imgROIStartX,0,imageROIWidth,img1ColourTransform.rows) );
        
        rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
        
        for(vector<int>::iterator it2 = patchDimensions.begin(); it2 != patchDimensions.end(); ++it2) {
            
            vector<int> result_rows;
            vector<Point2f> template_coords;
            vector<Mat> patch_templates = getROIPatches(image1ROI, template_coords, result_rows, *it2);
            
            vector <vector<double> > allResults;
            vector <double> timeDurations;
            
            
            for(vector<int>::iterator it3 = match_type.begin(); it3 != match_type.end(); ++it3) {
                
                cout << "BEGIN: Test #" << testCount << ": ROI Size = " << *it1 << ", Patch Size = " << *it2 << ", Match Method = " << getMatchMethodName(*it3) << endl;
                
                double testElaspedTime = (double)getTickCount();
                
                allResults.push_back(runTestPatch(image2ROI, *it2, *it3, patch_templates, template_coords));
                
                timeDurations.push_back(((double)getTickCount() - testElaspedTime)/getTickFrequency());
                
                cout << "END: Test #" << testCount << "\n\n";
                
                testCount++;
            }
            
            
            exportResults(allResults, result_rows, match_type, *it1, *it2);
            
            exportTimeResults(timeDurations, match_type, *it1, *it2);
            
        }
        
        
    }
    
}

void exportResults(vector<vector<double > > all_results, vector<int> rows, vector<int> methods, int patchSize, int roiSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << roiSize << "_" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    oStream << "descriptor image_row";
    
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

void exportTimeResults(vector<double> timeTaken, vector<int> methods, int patchSize, int roiSize, string fileNamePrefix) {
    
    ostringstream oStream;
    ofstream myfile;
    
    oStream << fileNamePrefix << roiSize << "_" << patchSize << ".dat";
    
    myfile.open (oStream.str(), ios::out | ios::trunc);
    
    oStream.clear();
    oStream.str("");
    
    myfile << "descriptor match_method time_taken\n";
    
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
        
        rectangle( img2, Point(imgROIStartX + originPixelCoords.x - (patchSize / 2), originPixelCoords.y - (patchSize / 2)), Point(imgROIStartX + originPixelCoords.x + (patchSize / 2) , originPixelCoords.y + (patchSize / 2)), Scalar(0, 0, 255), 2, 8, 0 );
        
        
        //Once we reach the end of the row, we need to calculate the average displacement across the entire row.
        if (rowNumber != originPixelCoords.y) {
            
            if (!raw_result.empty()) {
                
                double sum = accumulate(raw_result.begin(), raw_result.end(), 0.0);
                double mean = sum / raw_result.size();
                
                avg_result.push_back(mean);
                
                if (rowNumber == (patchSize / 2) || rowNumber == 300) {
                    
                    double minVal = *min_element(raw_result.begin(), raw_result.end());
                    double maxVal = *max_element(raw_result.begin(), raw_result.end());
                    
                    double bucketSize = 1;
                    
                   // double minusValue = (bucketSize / (bucketSize * 100));
                    
                    vector<int> hist_result = calcHistogram(bucketSize, raw_result, maxVal + 1);
                    
                    cout<< "RAW VALUES\n";
                    
                    for (std::vector<int>::size_type i = 0; i < raw_result.size(); ++i) {
                        
                        cout << raw_result[i] << "\n";
                        
                    }
                    
                    cout << "\n\nHISTOGRAM\n";
                    
                    for (std::vector<int>::size_type i = 0; i < hist_result.size(); ++i) {
                        
                        //cout << (i * bucketSize) << "-" << ((i + 1) * bucketSize - 1) << " -> " << hist_result[i] << "\n";
                        
                        cout << (i * bucketSize) << " -> " << hist_result[i] << "\n";
                        
                    }

                }
                
            } else {
                
                cout << "ERROR: No displacement values obtained for Row #: " << rowNumber;
                
            }

            rowNumber = originPixelCoords.y;
            
            raw_result.clear();
            
        }
        
        if(rowNumber >=(patchSize / 2) + 1) {
            break;
        }
    
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - (originPixelCoords.y - (templatePatch.cols / 2));
        
        localisedSearchWindow = image2ROI(Rect(originPixelCoords.x - (templatePatch.cols / 2), originPixelCoords.y - (templatePatch.cols / 2), localisedWindowWidth, localisedWindowHeight));
        
        calcPatchMatchScore(localisedSearchWindow, templatePatch, match_method, highestScore, displacement);
        
        raw_result.push_back(displacement);
        
        //CG - Allows the output window displaying the current patch to be updated automatically.
        cv::waitKey(1);
        imshow("Search Window", localisedSearchWindow);
        imshow("Template Patch", templatePatch);
        
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
    
    int i,j;
    uchar* p;
    
    //CG - Stop extracting patches when we get to the bottom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = halfPatchSize; i < (nRows - halfPatchSize); i++)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        //CG - Push back the current row number (used for printing results later on).
        rows.push_back(i);
        
        for (j = halfPatchSize; j < (nCols - halfPatchSize); j++)
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
//        cv::waitKey(1);
//        resize(resultMat, resultMat, Size(resultMat.cols*100, resultMat.rows*100));
//        imshow("Normalised RESULT", resultMat);
        
        if (stop) {
            
            break;
            
        }
        
    }
    
    highScore = bestScore;
    highScoreIndexY = bestScoreYIndex;
    
}