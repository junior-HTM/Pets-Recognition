//the image we want to test
const char *myTestImage = "C:/Users/PC/Documents/work/OpenCV-EXP/Object Recognition/bin/Release/chachi.jpg";
//video file to sample while the collecting mode
const char *myVideoFile = "C:/Users/PC/Documents/work/OpenCV-EXP/Object Recognition/bin/Release/IMG_0129.MOV";
//this is where we'll save the data to
const char *trainedImagesDir = "C:/Users/PC/Documents/work/OpenCV-EXP/Object Recognition/bin/Release/data/";
//the log file name
const char *LOG_FILE_NAME = "C:/Users/PC/Documents/work/OpenCV-EXP/Object Recognition/bin/Release/data/log.txt";
//the name of the app
const char *APP_NAME = "Object Recognition";
//the name of the app window
const char *WINDOW_NAME = "Object Recognition";
//call to actions
const char *CAPTURE_CTA = "CHOOSE A NUMBER";
const char *REC_CTA = "CLICK TO RECOGNIZE";

//application modes
const char *TRAIN_MODE= "TRAIN";
const char *COLLECT_MODE= "COLLECT";
const char *RECOGNIZE_MODE= "RECOGNIZE";
const char *MODE= "";

//we'll use this to size down the images we process
const int WINDOW_WIDTH = 640;
//the dimention of the image we'll save for training
const int SAVED_OBJ_WIDTH = 300;

//the training algorithems, using either the Fisherfaces (Principal Component Analysis - PCA)
//or the Eigenfaces (LInear Discriminant Analysis - LDA)
const char *recAlgorithmFisherfaces = "FaceRecognizer.Fisherfaces";
const char *recAlgorithmEigenfaces = "FaceRecognizer.Eigenfaces";
const char *recAlgorithm = recAlgorithmEigenfaces;

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


//a rectangle to masure the mouse movement
Rect mouseBox;
bool drawing_box = false;
bool train_box_on = false;
//Mat of the current frame - image
Mat currentFrame;
//Mat of the source frame - image, we'll use that to reset the mat while drawing
Mat sourceMat;
//check if the video is playing or if we need to pause it
bool isPlaying = false;

//array to sort the Matrixes we collected for training
vector<Mat> objectsArray;
//indexes of the objects we recognized.
//while in the collecting mode, when a user draw
//a rect on the image, they can click a key to save it, we'll use
//the key they selected as our selected index
vector<int> labelsArray;
int selectedIndex = 1;
//unique index for saving the files by, the format we'll use
//will be similar to 1_1.jpg representing /selectedIndex_savedIndex.jpg
int savedIndex = 1;
//reference for the last object selected
Mat lastObject;
//Mat of the recognized area to use while in recognize mode,
//this will be a cropped verison of the currentImage(mouseBox);
Mat recognizeMat;

//define the model for training
Ptr<FaceRecognizer> model;

//=======================================
// foward decleration - convinent way to
// have a rundom order of methods in the same file
// with the ability to refer to them from one another
// I guess this could be replaced with an header interface
//=======================================
void recognizeObject( Mat obj );
Mat resizeImage(Mat src, int width);
Mat getGrayscale(Mat src);
double getSimilarity(const Mat A, const Mat B);
void equalizeObjectHist(Mat &object, bool sliceInHalf);
Mat cleanObject(Mat object);
int getIndexFromFileName(string filename);
int getLastSavedLogIndex();
void saveToLog(string objectName);
void loadFromLog();
void saveCurrentImage();
void highlight( Mat img, Rect mouseRect);
void onMouse(int event, int x, int y, int, void*);
void collectData();
void trainData();
void recognizeObject( Mat obj );
void initRecognition();
//=======================================
// std string - int conversion methods
//=======================================
template <typename T> string toString(T t){
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t){
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

//========================================================
//          Mat minipulation methods
//========================================================

//resize an image to a specific width
Mat resizeImage(Mat src, int width) {
    Mat sized;
    float scale = src.cols / (float) width;
    if (src.cols > width) {
//        resize the image while keeping the proportions
        int h = cvRound( src.rows / scale );
        resize(src, sized, Size(width, h));
    } else {
        sized = src;
    }
    return sized;
}

//convert the image to a grayscale image for faster processing
Mat getGrayscale(Mat src) {
    Mat gray;
    if ( src.channels() == 3 ) {
//        3 channels to grayscale conversion
        cvtColor(src,gray, CV_BGR2GRAY);
    } else if ( src.channels() == 4 ) {
//        4 channels to grayscalce conversion
        cvtColor(src,gray,CV_BGRA2GRAY);
    } else {
//        the image is prob already a grayscale image
        gray = src;
    }
    return gray;
}

//compear two images by calculating the L2 error (square root of sum of squared error)
double getSimilarity(const Mat A, const Mat B) {

    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        double errorL2 = norm(A,B, CV_L2);
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    } else {
        return 100000000.0;  // Return a bad value
    }

}

//adjust the brightness of the found object as a whole or two half
//this is usefull for pictures which were shot where the lights come from the side
void equalizeObjectHist(Mat &object, bool sliceInHalf) {

    if (!sliceInHalf) {
        equalizeHist(object, object);
    } else {
        //adjust the color on two seperate sides;
        int w = object.cols;
        int h = object.rows;

//        equalize the whole object
        Mat whole;
        equalizeHist(object, whole);

//        equalize both sides seperatly
        int middleX = w/2;
        Mat leftSide = object(Rect(0,0,middleX, h));
        Mat rightSide = object(Rect(middleX, 0, w - middleX, h));
        equalizeHist(leftSide, leftSide);
        equalizeHist(rightSide, rightSide);

//        imshow("left", leftSide);
//        imshow("right", rightSide);
//        imshow("whole", whole);

//        combine the two sides and blend the border edges
        for(int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                int v;
//                the first 25% - use the left side
                if (x < w/4) {
                    v = leftSide.at<uchar>(y,x);
                }
//                center left 25% - blend
                else if (x < w*2/4) {
                    int lv = leftSide.at<uchar>(y,x);
                    int wv = whole.at<uchar>(y,x);
                    float f = (x - w*1/4) / (float)(w*0.25f);
                    v = cvRound((1.0f - f) * lv + (f) * wv);
                }
//                center right 25% - blend
                else if ( x < w*3/4) {
                    int rv = rightSide.at<uchar>(y,x-middleX);
                    int wv = whole.at<uchar>(y,x);
                    float f = (x - w*2/4) / (float)(w*0.25f);
                    v = cvRound((1.0f-f) * wv +(f) * rv);
                }
//                right 25% - use the right side
                else {
                    v = rightSide.at<uchar>(y,x-middleX);
                }
                object.at<uchar>(y,x) = v;
            }
        }

    }

}

//Clean an image before saving it using the following steps
//1. convert to grayscale
//2. equalize the histogram of the image, using the two sides of it
//3. reduce the noise with bilateralFilter
Mat cleanObject(Mat object) {

//    imshow("object", object);
    //convert to grayscale
    Mat gray = getGrayscale(object);
//    imshow("gray", gray);

    //adjust the object histogram with equalized hist filter
    equalizeObjectHist(gray, true);
//    imshow("equalized", gray);

    //reduce noize
    Mat filtered = Mat(gray.size(), CV_8U);
    bilateralFilter(gray, filtered, 0, 20.0, 2.0);
//    imshow("filtered", filtered);

    return filtered;

}


//========================================================
//                    Save/log/load
//========================================================

//split a filename (filepath) based on delimiter - / and _ and return the saved index
int getIndexFromFileName(string filename) {
    //something/1_1.jpg

    //clean the filepath first
    string delimiter = "/";
    size_t pos = 0;
//    std::string token;
    while ((pos = filename.find(delimiter)) != std::string::npos) {
//        token = filename.substr(0, pos);
//        std::cout << token << std::endl;
        filename.erase(0, pos + delimiter.length());
    }
//    std::cout << filename << std::endl;

    pos = filename.find("_");
    filename.erase(pos, filename.length());
    //convert to int and return the index
    return atoi(filename.c_str());

}

//return the last saved index based on the number of entries in the log file
int getLastSavedLogIndex() {

    ifstream file;
    string filename = LOG_FILE_NAME;
    file.open(filename.c_str());

    int index = 0;
    string line;
    while(getline(file,line)){
        if ( line !=  "" && line != "\n" ) {
            index++;
        }
    }
    return index;

}


//save new entry to the log file
void saveToLog(string objectName) {
    //save the image name in the log file
    ofstream myfile;
    myfile.open (LOG_FILE_NAME, fstream::in | fstream::out | fstream::app);
    myfile << objectName + "\n";
    myfile.close();
}


//load the data from the log file and push it to the objects and labels arrays;
void loadFromLog() {

    ifstream file;
    string filename = LOG_FILE_NAME;
    file.open(filename.c_str());

    Mat object;
    Mat flipped;
    string line;
    while(getline(file,line)){
        if ( line !=  "" && line != "\n" ) {
            selectedIndex = getIndexFromFileName( line );
            object = imread(line, 1);
            flip(object, flipped, 1);
            objectsArray.push_back(object);
            objectsArray.push_back(flipped);
            labelsArray.push_back(selectedIndex);
            labelsArray.push_back(selectedIndex);
        }
    }

}


//save the current image, if it different enough from the previous one
void saveCurrentImage() {

    //    create a mat out of the mouseBox rect
    Mat cropped = currentFrame(mouseBox);

    //clean the object
    Mat object = cleanObject(cropped);
    double difference = 10000000000.0;
    if (lastObject.data)
        difference = getSimilarity(object, lastObject);

    //only save a new object if its different enough from the last object
    if ( difference > 0.3 ) {
        //flip the object for varity
        Mat flipped;
        flip(object, flipped, 1);
        //sort the data in our objects and labels array
        objectsArray.push_back(object);
        objectsArray.push_back(flipped);
        labelsArray.push_back(selectedIndex);
        labelsArray.push_back(selectedIndex);

        //save the image to the disk for future detection
        string imageFile =  toString(trainedImagesDir) + toString(selectedIndex) + "_" + toString(savedIndex) +".jpg";

        //lets resize it so all the save images are the same
        Mat sized = resizeImage(cropped, SAVED_OBJ_WIDTH);
        imwrite( imageFile , sized );

        //save to the log file
        saveToLog(imageFile);

        //up the index
        savedIndex++;

    } else {
        cout << "object is too similar to the previous object, try again" << endl;
    }

}

//========================================================
//                    Mouse events
//========================================================

//draw the rect on the current image and place the CTA on top of it
void highlight( Mat img, Rect mouseRect){
    rectangle(img, mouseRect, CV_RGB(255,255,0), 2, CV_AA);
    if (MODE == TRAIN_MODE) {
        putText(img, CAPTURE_CTA, Point( mouseRect.x, mouseRect.y - 10 ), CV_FONT_HERSHEY_TRIPLEX, 0.35, cvScalar(255,255,255), 0, CV_AA);
    } else if ( MODE == RECOGNIZE_MODE ) {
        putText(img, REC_CTA, Point( mouseRect.x, mouseRect.y - 10 ), CV_FONT_HERSHEY_TRIPLEX, 0.35, cvScalar(255,255,255), 0, CV_AA);
    }
}

//handle the mouse events
//we'll switch between 3 events
//1. CV_EVENT_MOUSEMOVE - while the mouse move if the drawing box is on, we'll add the mouse position to the mouse rect box
//2. CV_EVENT_LBUTTONDOWN - check for similar objects if we are in the recognize mode.
//                        - reset the current frame image so if we have a rectangle already drawn it will be ereased
//3. CV_EVENT_LBUTTONUP - see if we have a valid rect area and draw it on the image
void onMouse(int event, int x, int y, int, void*) {

    switch( event ){
        //draw the rectangle if the mouse was clicked befor the move
		case CV_EVENT_MOUSEMOVE:
			if( drawing_box ){
				mouseBox.width = x-mouseBox.x;
				//draw a proportional square
				mouseBox.height = mouseBox.width;//y-mouseBox.y;
				isPlaying = false;
			}
			break;

        //on mouse down:
        //either define a new rectangle
        //or save the current one
		case CV_EVENT_LBUTTONDOWN:
		    if (train_box_on) {
                //check if the click was inside the box
                if ( x >= mouseBox.x && x <= mouseBox.x + mouseBox.width ) {
                    if ( y >= mouseBox.y && y <= mouseBox.y + mouseBox.height  ) {
                            if ( MODE == RECOGNIZE_MODE) {
                                recognizeMat = currentFrame(mouseBox);
                                recognizeObject(recognizeMat);
                            }
                    }
                }
            }

            //if clicked out side the current rectangle,
            //start drawing a new one instead
            drawing_box = true;
			mouseBox = Rect( x, y, 0, 0 );
//			reset the current image to the original
            sourceMat.copyTo(currentFrame);
			break;

        //on mouse up
        //finish the rect and draw it on the image
		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( mouseBox.width < 0 ){
				mouseBox.x += mouseBox.width;
				mouseBox.width *= -1;
			}
			if( mouseBox.height < 0 ){
				mouseBox.y += mouseBox.height;
				mouseBox.height *= -1;
			}

//            if we have a valid rect lets draw it on the image
            if (mouseBox.width > 0 && mouseBox.height > 0 ) {
                train_box_on = true;
                highlight( currentFrame, mouseBox );
                isPlaying = false;
            } else {
                //or set the mouse rect to invald
                mouseBox  = Rect(-1,-1,-1,-1);
                train_box_on = false;
                isPlaying = true;
            }
			imshow(WINDOW_NAME, currentFrame);
			break;
	}

}

//========================================================
//                    Collect data
//========================================================

//function to collect data from a video file
//we'll play the video in a slower rate while allowing the user
//to sample areas of it and define them as objects
void collectData() {

    //load the video file
     cv::VideoCapture video(myVideoFile);
    //check for errors
    if ( !video.isOpened() ) {
        cerr << "ERROR: Could not access the video!" << endl;
        exit(1);
    }
    //get the last saved index from the log file
    savedIndex  = getLastSavedLogIndex();
    //and set the objects - labels arrays with data from the log file
    loadFromLog();
    //the name of our window, I like to define this outside the loop so
    //the mouse callback funciton is only needed to be called once
    namedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );
    //add the mouse callbacks
    setMouseCallback(WINDOW_NAME, onMouse, 0);

    //start playing the video
    isPlaying = true;
    //loop till the end of the file
    while ( true ) {
        //check if we need to pause
        if ( isPlaying ) {
            //get the current frame from the video
            video >> sourceMat;
            //check for error - or end of clip
            if ( sourceMat.empty() ) {
                isPlaying = false;
                break;
            } else {
                //copy and resize the source
                currentFrame = resizeImage(sourceMat, WINDOW_WIDTH);
                currentFrame.copyTo(sourceMat);
                //show it in the window
                imshow(WINDOW_NAME, currentFrame);
            }
        }
        int key = waitKey(100);
        if (key == 27) break;

        //save the image if a number was pressed while teh train box is on
        if (train_box_on) {
            //check if the key is numeric
            if ( key >= 49 && key <= 57) {
                //snap the pictuer and load the next image
                selectedIndex = key - 48;
                saveCurrentImage();
                isPlaying = true;
            }
        }


    }


}

//========================================================
//                    Train
//========================================================

//function to train the mode.
//1. load the data from the log file and push it into the objects and labels array with coresponding index values
//2. check if the contrib module is available for training
//3. load the model
//4. train it with the data from the objects and labels arrays
void trainData() {

    loadFromLog();

    //chekc if the contrib module is available first
    bool contribModelStatus = initModule_contrib();
    if ( !contribModelStatus )
            exit(1);

    //define the model
    model = Algorithm::create<FaceRecognizer>(recAlgorithmFisherfaces);
    //check if the model was loaded properly
    if (model.empty())
            exit(1);

    //train the model
    model->train(objectsArray, labelsArray);

}

//========================================================
//                    recognize
//========================================================

//recognize an object - Mat in our collected data using the model
//1. get the eigenvectors - a vector representing the objects in our database
//                     - for Eigenfaces we'll have a vector for each image we used for training
//                     - for Fisherfaces we'll have a vector for each object we trained
//2. get the average object - this will return an mixed version of the trained objects
//3. define a projection Mat based on the avarage object, the eigenvectors and the object we'll like to test
//4. reconstruct and reshape the Mat
//5. compear the similarity between the reconstructed mat and the one we want to test
void recognizeObject( Mat obj ) {

    try{
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat average = model->get<Mat>("mean");

        int height = obj.rows;

        //define a projection Mat based on the eigenvectors, avarage object and the object we'll like to test
        Mat projection = subspaceProject(eigenvectors, average, obj.reshape(1,1));
        //reconstruct the projection and reshape it.
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, average, projection);
        Mat reconstructionMat = reconstructionRow.reshape(1, height);
        Mat reconsructedObj = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconsructedObj, CV_8U, 1, 0 );
//        imshow("Reconstructed", reconsructedObj);

//        compear the recognizeMat to the reconsructedObj
        double similarity = getSimilarity(recognizeMat, reconsructedObj);

        //how accurate we want to be in our definition, the lower the value the more accurate
        //we get, recommanded values for Eigenfaces and Fisherfaces are around the 0.5-0.7.
        //this could be set higher to get higher range of objects
        float obj_treshold = 0.7f;
        //check if the similarity is larger then the treshold
        if (similarity > obj_treshold) {
            identity = model->predict(recognizeMat);
            cout << "Object identity: " << toString(identity) << endl;
        } else {
            cout << "unidentified object" << endl;
        }


    } catch (cv::Exception e) {
        //nothing was found?
    }

}

//initilize the app in recogniton mode
//1. load the data and train the model
//2. read an image file
//3. set a user defined area and use it for testing
void initRecognition() {
    //train the data first
    trainData();
    //load the test image
    sourceMat = imread(myTestImage, 1);
    sourceMat.copyTo(currentFrame);

    namedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );
    setMouseCallback(WINDOW_NAME, onMouse, 0);

    while (true) {
        imshow(WINDOW_NAME, currentFrame);
        int key = waitKey(20);
        if (key == 27) break;
    }

}

//========================================================
//                       Main
//========================================================

//the app will work in 3 modes:
//1. COLLECT_MODE - collecting object to recognize from a video file,
//                  save the trained object into a directory and a log file
//2. TRAIN_MODE - train the model, there is no real use for this mode aside
//                testing for errors in the training process
//3. RECOGNIZE_MODE - load an image, train the model based on the data collected in the log file
//                    and see if it finds any similar objects
int main(int argc, char *argv[]) {

    MODE = RECOGNIZE_MODE;

    cout <<  APP_NAME << " :: " << MODE << endl;

    if (MODE == COLLECT_MODE) {
        collectData();
    } else if (MODE == TRAIN_MODE) {
        trainData();
    } else if (MODE == RECOGNIZE_MODE) {
        initRecognition();
    }

    return 0;

}
