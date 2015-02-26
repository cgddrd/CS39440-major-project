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
    
    Mat result, result2, img1_gray, img2_gray, localisedSearchWindow, templatePatch;
    
    // cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    // cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    
    
    if (useRGB) {
        
        //cvtColor(img1, img1_gray, cv::COLOR_BGR2RGB);
        //cvtColor(img2, img2_gray, cv::COLOR_BGR2RGB);
        
        cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
        cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
        
        //img1_gray = img1.clone();
        //img2_gray = img2.clone();
        
    } else {
        
        //BGR2HSV = Hue Range: 0-180
        //BGR2HSV_FULL = Hue Range: 0-360
        cvtColor(img1, img1_gray, cv::COLOR_BGR2HSV);
        cvtColor(img2, img2_gray, cv::COLOR_BGR2HSV);
        
    }
   
    //CG - Calculate a central column through the two images that has a percentage width of the original images.
    double imageCentreX = img1_gray.cols / 2;
    double imageROIWidth = img1_gray.cols * 0.3;
    double imageROIHalfWidth = imageROIWidth / 2;
    double imgROIStartX = imageCentreX - imageROIHalfWidth;
    double imgROIEndX = imageCentreX + imageROIHalfWidth;
    
    
    //CG - Extract the central column ROI from the two images ready to perform feature detection and optical flow analysis on them.
    Mat image1ROI = img1_gray( Rect(imgROIStartX,0,imageROIWidth,img1_gray.rows) );
    Mat image2ROI = img2_gray( Rect(imgROIStartX,0,imageROIWidth,img2_gray.rows) );
    
    Mat opticalFlow = Mat::zeros(img2.rows, img2.cols, CV_8UC3);
    
    vector<Point2f> points;
    
    vector<Mat> test = ScanImagePointer(image1ROI, points, patchSize);
    
    std::vector<Mat>::iterator i1;
    std::vector<Point2f>::iterator i2;
    
    int txtcount = 0;
    
    for( i1 = test.begin(), i2 = points.begin(); i1 < test.end() && i2 < points.end(); ++i1, ++i2 )
    {
        
//        if (txtcount >=5) {
//          break;
//        }
        
        double currentMaxResult = 0;
        int val = 0;
        
        Point2f originPixelCoords = (*i2);
        
        templatePatch = *i1;
        
        int localisedWindowX = originPixelCoords.x;
        int localisedWindowY = originPixelCoords.y;
        int localisedWindowWidth = templatePatch.cols;
        int localisedWindowHeight = image2ROI.rows - localisedWindowY;
        
        localisedSearchWindow = image2ROI(Rect(localisedWindowX, localisedWindowY, localisedWindowWidth, localisedWindowHeight) );
        
        //Could look at applying a gaussian blur? - Does this help at all?
        //GaussianBlur( localisedSearchWindow, localisedSearchWindow, Size( 123,123 ), 0, 0 );
        //GaussianBlur( templatePatch, templatePatch, Size( 123,123 ), 0, 0 );
    
        MatND hist_template;
        MatND hist_current;
        
        if (useRGB) {
            
//            int imgCount = 1;
//            int dims = 3;
//            const int sizes[] = {256,256,256};
//            const int channels[] = {0,1,2};
//            float rRange[] = {0,256};
//            float gRange[] = {0,256};
//            float bRange[] = {0,256};
//            const float *ranges[] = {rRange,gRange,bRange};
//            
//            calcHist(&templatePatch, imgCount, channels, Mat(), hist_template, dims, sizes, ranges);
//            //normalize( hist_template, hist_template, 0, 1, NORM_MINMAX, -1, Mat() );
            
            // Initialize parameters
            int histSize = 256;    // bin size
            float range[] = { 0, 255 };
            const float *ranges[] = { range };
            
            // Calculate histogram
            calcHist( &templatePatch, 1, 0, Mat(), hist_template, 1, &histSize, ranges, true, false );
            normalize( hist_template, hist_template, 0, 1, NORM_MINMAX, -1, Mat() );
            
        } else {
            
            /// Using 50 bins for hue and 60 for saturation
            int h_bins = 50; int s_bins = 60;
            int histSize[] = { h_bins, s_bins };
            
            // hue varies from 0 to 179, saturation from 0 to 255
            float h_ranges[] = { 0, 180 };
            float s_ranges[] = { 0, 256 };
            
            const float* ranges[] = { h_ranges, s_ranges };
            
            // Use the 0-th and 1-st channels
            int channels[] = { 0, 1 };
            
            /// Calculate the histogram for the template patch image.
            calcHist( &templatePatch, 1, channels, Mat(), hist_template, 2, histSize, ranges, true, false );
            normalize( hist_template, hist_template, 0, 1, NORM_MINMAX, -1, Mat() );
            
        }
        
        for(int i = 0; i < localisedSearchWindow.rows - templatePatch.rows; i++)
        {

            Mat current = localisedSearchWindow.clone();
            
            current = current(Rect(0, i, templatePatch.cols, templatePatch.rows));
            
            if (useRGB) {
                
//                int imgCount = 1;
//                int dims = 3;
//                const int sizes[] = {256,256,256};
//                const int channels[] = {0,1,2};
//                float rRange[] = {0,256};
//                float gRange[] = {0,256};
//                float bRange[] = {0,256};
//                const float *ranges[] = {rRange,gRange,bRange};
//                
//                calcHist(&current, imgCount, channels, Mat(), hist_current, dims, sizes, ranges);
//                //normalize( hist_current, hist_current, 0, 1, NORM_MINMAX, -1, Mat() );
                
                // Initialize parameters
                int histSize = 256;    // bin size
                float range[] = { 0, 255 };
                const float *ranges[] = { range };
                
                // Calculate histogram
                calcHist( &current, 1, 0, Mat(), hist_current, 1, &histSize, ranges, true, false );
                normalize( hist_current, hist_current, 0, 1, NORM_MINMAX, -1, Mat() );
                
            } else {
                
                /// Using 50 bins for hue and 60 for saturation
                int h_bins = 50; int s_bins = 60;
                int histSize[] = { h_bins, s_bins };
                
                // hue varies from 0 to 179, saturation from 0 to 255
                float h_ranges[] = { 0, 180 };
                float s_ranges[] = { 0, 256 };
                
                const float* ranges[] = { h_ranges, s_ranges };
                
                // Use the o-th and 1-st channels
                int channels[] = { 0, 1 };
                
                /// Calculate the histogram for the template patch image.
                calcHist( &current, 1, channels, Mat(), hist_current, 2, histSize, ranges, true, false );
                normalize( hist_current, hist_current, 0, 1, NORM_MINMAX, -1, Mat() );
                
            }
            
            double histresult = compareHist( hist_current, hist_template, histogramCompMethod );
            
            if (histresult > currentMaxResult) {
                
                currentMaxResult = histresult;
                val = i;
                
            }
            
            //CG - Allows the output window displaying the current patch to be updated automatically.
            //cv::waitKey(10);
            //imshow("Current", current);
            
            cout << "I: " << i << " - Result: " << histresult << endl;
            
        }
        
        cout << "MAX POSITION: " << val << " - " << currentMaxResult << endl;
        
        rectangle( img1, Point(imgROIStartX + originPixelCoords.x, originPixelCoords.y), Point( (imgROIStartX + originPixelCoords.x + templatePatch.cols), originPixelCoords.y + templatePatch.rows ), Scalar(255, 0, 0), 2, 8, 0 );
        
        //CG - Histogram top value position rectangle
        rectangle( img2, Point(imgROIStartX + localisedWindowX, localisedWindowY + val), Point(imgROIStartX + localisedWindowX + templatePatch.cols ,localisedWindowY + val + templatePatch.rows), Scalar(255, 0, 255), 2, 8, 0 );
        
        circle(img2, Point(imgROIStartX + localisedWindowX, localisedWindowY + val),2, Scalar(255,255,255), CV_FILLED, 8,0);
        
        txtcount++;
        
    }
    
    // CG - Draw main ROI windows.
    rectangle( img1, Point(imgROIStartX, 0), Point(imgROIEndX , img1.rows), Scalar(0, 0, 255), 2, 8, 0 );
    rectangle( img2, Point(imgROIStartX, 0), Point(imgROIEndX , img2.rows), Scalar(0, 0, 255), 2, 8, 0 );
    
    //CG - <0.5 = more balance to 'resultFrame', >0.5 = more balance to 'img1'.
    double alpha = 0.4;
    
    imshow("Result", img2);
    
    addWeighted(img1, alpha, img2, 1.0 - alpha , 0.0, img2);
    
    imshow("Merged Result", img2);

    imshow("Search Window", localisedSearchWindow);
    
    imshow("Template", templatePatch);
    
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

    CV_Assert(inputMat.depth() != sizeof(uchar));
    
    vector<Mat> mats;
    int nRows = inputMat.rows;
    int nCols = inputMat.cols;
    
    int i,j;
    uchar* p;
    
    //CG - Stop extracting patches when we get 2*patchsize rows away from the botom of the image (no point doing it on the bottom-bottom patches as they won't move anywhere).
    for(i = 0; i < nRows - patchSize * 2; i+=patchSize)
    {
        
        p = inputMat.ptr<uchar>(i);
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for (j = 0; j < nCols - patchSize; j+=patchSize)
        {
            mats.push_back(inputMat(Rect(j,i,patchSize,patchSize)));
            
            //CG - Same as Point2f (typename alias)
            points.push_back(Point_<float>(j, i));
            
        }
        
    }
    
    return mats;
}