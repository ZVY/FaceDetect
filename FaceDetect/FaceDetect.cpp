// FaceDetect.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

#include <cv.h>
#include <highgui.h>
#include "opencv2/opencv.hpp"

#include <boost/filesystem.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#include "json.hpp"

using namespace cv;

struct ResultDetect
{
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
	CascadeClassifier mouth_cascade;

	std::vector<Rect> faces;
	std::vector<Rect> eyes;
	std::vector<Rect> mouth;
};

std::ofstream file_id("result.json");
std::mutex mut;

namespace bamthread
{
	typedef std::unique_ptr<boost::asio::io_service::work> asio_worker;

	struct ThreadPool {
		ThreadPool(size_t threads) :service(), worker(new asio_worker::element_type(service)) {
			while (threads--)
			{
				auto worker = boost::bind(&boost::asio::io_service::run, &(this->service));
				groupThread.add_thread(new boost::thread(worker));
			}
		}

		template<class F>
		void enqueue(F f) {
			service.post(f);
		}


		~ThreadPool() {
			worker.reset(); 
			groupThread.join_all();
			service.stop();
			file_id.close();			
		}

		boost::asio::io_service service; 
		asio_worker worker;
		boost::thread_group groupThread;
	};
}

void startProc(string ful_path_dir, bamthread::ThreadPool &tp, ResultDetect &detect);
void detectFace(string path, string name_file, ResultDetect &detectInfo);
string getNameFace(int id, String name_file)
{
	static int i = 0;
	string padd = ((id + 1) < 10) ? "0" : "";
	return  ("face" + padd + std::to_string(id + 1) + "_" + name_file);
}

int main(int argc, char** argv)
{		
	string full_puth_dir;
	int numb_worker;

	if (argc > 1)
	{					
		numb_worker = std::stoi(argv[1]);
		if ((numb_worker <= 0) || (numb_worker >= 10)) numb_worker = 3;
		full_puth_dir = argv[2];
		std::cout << "Will be set the following parameters: numder workers = " << numb_worker 
			      << ", full path to directory = " << full_puth_dir << "\n\n";
	}
	else
	{
		numb_worker = 3;
		full_puth_dir = "D:/foto/";
		std::cout << "Will apply the default settings: numder workers = " << numb_worker
			      << ", full path to directory = " << full_puth_dir << "\n\n";
	}

	std::cout << "Start app..\n\n";
	
	ResultDetect detectInfo;
	bamthread::ThreadPool tp(numb_worker);

	startProc(full_puth_dir, tp, detectInfo);
		
	std::cin.get();

	return 0;
}

void startProc(string ful_path_dir, bamthread::ThreadPool &tp, ResultDetect &detect)
{
	String face_cascade_name  = "haarcascade_frontalface_alt_tree.xml";
	String eyes_cascade_name  = "haarcascade_eye.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";

	if (!detect.face_cascade.load(face_cascade_name))   { printf("--- Error loading face_cascade\n");  return; };
	if (!detect.eyes_cascade.load(eyes_cascade_name))   { printf("--- Error loading eyes_cascade\n");  return; };
	if (!detect.mouth_cascade.load(mouth_cascade_name)) { printf("--- Error loading mouth_cascade\n"); return; };
	
	std::cout << "+ Cascades of recognition successfully loaded\n\n";

	namespace fs = boost::filesystem;
	try
	{		
		for (fs::recursive_directory_iterator dir_end, dir(ful_path_dir); dir != dir_end; ++dir)
		{
			fs::path _path(*dir);
			if (!fs::is_directory(_path))
			{	
				string name_file = _path.filename().string();
				if (!name_file.empty())
				{
					string full_path = _path.parent_path().string() + "/" + name_file;
					tp.enqueue(boost::bind(detectFace, full_path, name_file, std::ref(detect)));
				}
			}
		}
	}
	catch (fs::filesystem_error e)
	{
		std::cout << "--- Error processing directory: " << ful_path_dir << "\n";
	}	
}

void detectFace(string full_path, string name_file, ResultDetect &detectInfo)
{	
	std::lock_guard<std::mutex> lk(mut);

	Mat img = imread(full_path);
	if (img.empty())
		return;
	
	Mat frame_gray;
	cvtColor(img, frame_gray, CV_BGR2GRAY);
	// increase the contrast of the image
	equalizeHist(frame_gray, frame_gray);

	// detect faces
	detectInfo.face_cascade.detectMultiScale(frame_gray, detectInfo.faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < detectInfo.faces.size(); i++)
	{
		Rect rect(detectInfo.faces[i]);
		rectangle(img, rect, Scalar(255, 0, 255), 2, 8, 0);

		Mat faceROI = frame_gray(detectInfo.faces[i]);

		//-- In each face, detect eyes		
		detectInfo.eyes_cascade.detectMultiScale(faceROI, detectInfo.eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < detectInfo.eyes.size(); j++)
		{
			Rect rect(detectInfo.faces[i].x + detectInfo.eyes[j].x,
				detectInfo.faces[i].y + detectInfo.eyes[j].y,
				detectInfo.eyes[j].width,
				detectInfo.eyes[j].height);

			rectangle(img, rect, Scalar(255, 0, 0), 2, 8, 0);
		}

		//-- In each face, detect mouth		
		detectInfo.mouth_cascade.detectMultiScale(faceROI, detectInfo.mouth, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30));
		for (size_t j = 0; j < detectInfo.mouth.size(); j++)
		{
			Rect rect(detectInfo.faces[i].x + detectInfo.mouth[j].x,
				detectInfo.faces[i].y + detectInfo.mouth[j].y,
				detectInfo.mouth[j].width,
				detectInfo.mouth[j].height);

			rectangle(img, rect, Scalar(0, 0, 255), 2, 8, 0);
		}
	}

	// display notification
	std::cout << "Photo successfully processed = " << full_path << std::endl 
		      << "All found faces = " << detectInfo.faces.size() << "\n\n";

	// transformation and save the image
	for (int i = 0; i < detectInfo.faces.size(); i++)
	{		
		Rect rect = detectInfo.faces[i];
		IplImage src_img = img;
		
		cvSetImageROI(&src_img, cvRect(rect.x, rect.y, rect.width, rect.height));
		
		IplImage *sub_img = cvCreateImage(cvGetSize(&src_img),
			src_img.depth,
			src_img.nChannels);
	
		cvCopy(&src_img, sub_img, NULL);
		
		cvResetImageROI(&src_img);

		cvFlip(sub_img, NULL, 1);
		cvSaveImage(getNameFace(i,name_file).c_str(), sub_img);
		cvReleaseImage(&sub_img);
	}

	// save json file
	{
		using json = nlohmann::json;
		json j;

		for (int i = 0; i < detectInfo.faces.size(); i++)
		{
			j[getNameFace(i, name_file)] = { { "x:", detectInfo.faces[i].x },{ "y:", detectInfo.faces[i].y } };
		}

		file_id << j;
	}
}