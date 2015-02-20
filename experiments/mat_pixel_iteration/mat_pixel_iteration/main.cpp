#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

//Mat& ScanImageIterator(Mat& I);

int ScanImageIterator(Mat& I);
int ScanImageForLoop(Mat& I);
int ScanImagePointer(Mat& I);

int main()
{
    
    Mat img = imread("1.JPG",CV_LOAD_IMAGE_COLOR);
    
    // CG - Used to "time" each iteration approach.
    double t = (double) getTickCount();
    
    int count = ScanImageForLoop(img);
    
    cout << count << endl;
    
    // CG - Get the elasped time in seconds.
    t = ((double)getTickCount() - t)/getTickFrequency();
    
    cout << "Time for NORMAL 'FOR LOOP' (secs): " << t << endl;
    
    
    t = (double) getTickCount();
    
    count = ScanImagePointer(img);
    
    cout << count << endl;
    
    t = ((double)getTickCount() - t)/getTickFrequency();
    
    cout << "Time for EFFICIENT POINTER (secs): " << t << endl;
    
    
    t = (double) getTickCount();
    
    count = ScanImageIterator(img);
    
    cout << count << endl;
    
    t = ((double)getTickCount() - t)/getTickFrequency();
    
    cout << "Time for ITERATOR (secs): " << t << endl;
    
    
    return 0;
    
}

// CG - Loop through the Mat structure using an iterator. Safer than the other approaches, but can be slower.
int ScanImageIterator(Mat& I)
{
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    int count = 0;
    
    // CG - Get the no. of channels, and decide whether we are dealing with a colour or greyscale image.
    const int channels = I.channels();
    
    switch(channels)
    {
        // CG - Grayscale image.
        case 1:
        {
            MatIterator_<uchar> it, end;
            
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
            {
                //*it = table[*it];
                
                cout << int(*it);
            }
            
            
            break;
        }
           
        // CG - Colour image (format BGR (not RGB))
        case 3:
        {
            MatIterator_<Vec3b> it, end;
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
                //(*it)[0] = table[(*it)[0]];
                //(*it)[1] = table[(*it)[1]];
                //(*it)[2] = table[(*it)[2]];
                
                count++;
                
                // CG - We need to CAST the uchar (unsigned char (0-255 only)) to an INT in order to get the TRUE colour value out for each channel.
                //cout << "B: " << int((*it)[0]) << ", G: " << int((*it)[1]) << ", R: " << int((*it)[2]) << endl;
                
            }
        }
    }
    
    return count;
}


// CG - This uses a traditional 'for' loop to go through the Mat data structure. Not the fastest.
int ScanImageForLoop(Mat& img)
{
    int count = 0;
    
    // CG - We need to get a pointer to the actual image data.
    unsigned char *input = (unsigned char*)(img.data);
    
    // CG - Scan through the rows first (Y)
    for(int j = 0;j < img.rows;j++){
        
        // CG - Then scan through the columns (X)
        for(int i = 0;i < img.cols;i++){
            
            // CG - Need to use the Mat 'step' value to move across pixels.
            unsigned char b = input[img.step * j + i];
            unsigned char g = input[img.step * j + i + 1];
            unsigned char r = input[img.step * j + i + 2];
            
            //cout << "B: " << int(b) << ", G: " << int(g) << ", R: " << int(r) << endl;
            
            count++;
        }
    }
    
  return count;
}

// CG - Loop through Mat structure using a 'C' pointer system. This is BLEEDINGLY FAST (Cannot go any faster), but can be more dangerous.
int ScanImagePointer(Mat& I)
{
    
    int count = 0;
    
    // accept only char type matrices
    CV_Assert(I.depth() != sizeof(uchar));
    
    int channels = I.channels();
    
    int nRows = I.rows;
    
    // CG - Multiply each row by the colour channels (i.e. 3 for BGR, and 1 for grayscale)
    int nCols = I.cols * channels;
    
    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    
    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        
        
        // CG - Here we loop through EACH ROW, and EACH COLOUR CHANNEL WITHIN EACH OF THESE ROWS AS WELL!
        for ( j = 0; j < nCols; ++j)
        {
            //cout << int(p[j]) << endl;
            
            count++;
            
        }
    }
    
    // CG - We divide by three so we are only returning the total count pixels (and not each of the CHANNELS within EACH PIXEL as well).
    return count / 3;
}
