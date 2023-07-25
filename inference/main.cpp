#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

#include "ncnn/net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

#include "opencv2/dnn/dnn.hpp"

#include "hourglass.h"


void PreProcess(cv::Mat& img, int imgSize = 256)
{
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	cv::resize(img, img, cv::Size(imgSize, imgSize));

	std::vector<float> mean_value{ 123.675, 116.28, 103.53 };                  // { 0.485, 0.456, 0.406 };
	std::vector<float> std_value{ 1.0 / 58.395, 1.0 / 57.120, 1.0 / 57.375 };  // { 0.229, 0.224, 0.225 };

	std::vector<cv::Mat> rgbChannels(3);
	cv::split(img, rgbChannels);
	for (auto i = 0; i < rgbChannels.size(); i++)
	{ 
		rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, std_value[i], - mean_value[i] * std_value[i]);
	}

	cv::merge(rgbChannels, img);
 
}

int main(int argc, char* argv[])
{
	// Opt
	int inpSize = 256;
	int nKps = 3;
	const std::string testImgPath = "E:/data/SplitTooth/AddFDIClassAndKeyPoint/keyPoint/black/1 (1)_top_black_flip.png";

	// OpenCV Inference
	const std::string onnxPath = "E:/code/Server223/hourglass-tooth/exp/tooth/model_convert/checkpoint_sim.onnx";
	cv::Mat cvImg = cv::imread(testImgPath);
	cv::dnn::Net cvNet = cv::dnn::readNet(onnxPath);
	int imgW = cvImg.cols, imgH = cvImg.rows;

	PreProcess(cvImg, inpSize);
	cv::Mat inpBlob = cv::dnn::blobFromImage(cvImg);
	cvNet.setInput(inpBlob);
	cv::Mat res = cvNet.forward();
	cv::MatSize resSize = res.size;   // [ B, N_Stack, nKps, outSize, outSize ]
	int nStack = res.size[1];
	int outSize = res.size[3];
	float wRatio = (imgW * 1.0) / (inpSize * 1.0);
	float hRatio = (imgH * 1.0) / (inpSize * 1.0);
	float ratio = (inpSize * 1.0) / outSize;  // 缩放比例
	 
	std::vector<std::pair<float, float>> resKeypoints;
	int lastLayerIdx = (res.size[1] - 1) * res.size[2] * res.size[3] * res.size[4];
	for (int nkp = 0; nkp < nKps; ++nkp) {
		int startIdx = lastLayerIdx + nkp * res.size[3] * res.size[4];
		float maxV = std::numeric_limits<float>::lowest();
		int locX = 0;
		int locY = 0;
		for (int i = 0; i < res.size[3]; ++i) {
			for (int j = 0; j < res.size[4]; ++j) {
				float value = *((float*)res.data + startIdx + i * res.size[3] + j);
				if (value > maxV) {
					locX = j;
					locY = i;
					maxV = value;
				}
			}
		}
		resKeypoints.push_back(std::make_pair(locX * ratio * wRatio, locY * ratio * hRatio));
		std::cout << "locX: " << locX << " locY: " << locY << " value: " << maxV << std::endl;
	}

//
//    // NCNN Inference
//	const std::string paramPath = "E:/code/Server223/hourglass-tooth/exp/tooth/model_convert/checkpoint_sim.param";
//	const std::string binPath = "E:/code/Server223/hourglass-tooth/exp/tooth/model_convert/checkpoint_sim.bin";
//	const std::string paramBinPath = "E:/code/TestC++/ncnn/test_sim.param.bin";
//	const std::string allBin = "E:/code/TestC++/ncnn/test_sim_all.bin";  // cat test_sim.param.bin test_sim.bin > test_sim_all.bin 
//
//	// Load Model
//	ncnn::Net net;
//
//	// net.opt.use_vulkan_compute = true;
//	//// 1.Load model
//	// --- method 1 ---
//	net.load_param(paramPath.c_str());
//	net.load_model(binPath.c_str());
//
//	//// --- method 2 ---
//	//net.load_param_bin(paramBinPath.c_str());
//	//net.load_model(binPath.c_str());
//
//	//// --- method 3 ---
//	// // need #inlucde "test_sim.mem.h"
//	//net.load_param(test_sim_param_bin);
//	//net.load_model(test_sim_bin);
//
//	//// --- method 4 ---
//	//FILE* fp = fopen(allBin.c_str(), "rb");
//	//int a = net.load_param_bin(fp);
//	//int b = net.load_model(fp);
//	//fclose(fp);
//
//	const std::vector<ncnn::Blob>& netBlobs = net.blobs();
//	const std::vector<ncnn::Layer*>& netLayers = net.layers();
//	std::cout << " blobs: " << netBlobs.size() << " layers: " << netLayers.size() << std::endl;
//
//
//	// Load img
//	cv::Mat img = cv::imread(testImgPath);
//	int imgW = img.cols;
//	int imgH = img.rows;
//#ifdef _DEBUG
//	if (false) {
//		cv::imshow("img", img);
//		cv::waitKey(0);
//		cv::destroyAllWindows();
//	}
//#endif
//
//	
//	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, inpSize, inpSize); 
//	const float MEANS[3] = { 123.675, 116.28, 103.53 };
//	const float STD[3] = { 1.0 / 58.395, 1.0 / 57.120, 1.0 / 57.375 };
//	input.substract_mean_normalize(MEANS, STD); 
//
//	ncnn::Extractor ex = net.create_extractor();
//	ex.set_num_threads(4);
//
//	ex.input("img", input);    
//	// get output
//	ncnn::Mat res;  
//
//	ex.extract("outs", res);
//
//	int nChannel = res.c;
//	float wRatio = (imgW * 1.0) / (inpSize * 1.0);
//	float hRatio = (imgH * 1.0) / (inpSize * 1.0);
//	float ratio = (inpSize * 1.0) / res.w;  // 缩放比例
//	int startChannel = nChannel - nKps;   // Just deal last layer
//	std::vector<std::pair<float, float>> resKeypoints;
//	for (int idx = startChannel; idx < nChannel; ++idx) {
//		float* cData = res.channel(idx);
//		float maxV = std::numeric_limits<float>::lowest();
//		int maxX = 0; 
//		int maxY = 0;
//		for (int i = 0; i < res.w; ++i) {
//			for (int j = 0; j < res.h; ++j) {
//				int idx = i * res.w + j;
//				if (cData[idx] > maxV) {
//					maxX = j;
//					maxY = i;
//					maxV = cData[idx];
//				}
//			}
//		}
//
//		resKeypoints.push_back(std::make_pair(maxX * ratio * wRatio, maxY * ratio * hRatio));
//		//resKeypoints.push_back(std::make_pair(maxX, maxY));
//		std::cout << "X: " << maxX << " Y: " << maxY << " value: " << maxV << std::endl;
//	}
//
//	ex.clear();
//	net.clear();
	return 0;
}
