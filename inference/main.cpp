#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

#include "hourglass.h"


int main(int argc, char* argv[])
{
	int testTimes = 100;
	const std::string testImgPath = "E:/data/SplitTooth/AddFDIClassAndKeyPoint/keyPoint/black/1 (1)_top_black_flip.png";
	 
	const std::string onnxPath = "E:/code/Server223/hourglass-tooth/exp/tooth/model_convert/checkpoint_sim.onnx";
	KeyPointCV kpCV;
	KPOption opt;
	auto startCVLoad = std::chrono::system_clock::now();
	kpCV.Build(onnxPath, opt);
	auto endCVLoad = std::chrono::system_clock::now();
	auto durationCVLoad = std::chrono::duration_cast<std::chrono::microseconds>(endCVLoad - startCVLoad);
	std::cout << "cv load model time: " << double(durationCVLoad.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;
	std::vector<Point> keypointsCV;
	auto startCV = std::chrono::system_clock::now();
	if (kpCV.IsValid()) {
		kpCV.Predict(testImgPath, keypointsCV, opt);
		//kpCV.Show(testImgPath, keypoints, opt);
	}
	auto endCV = std::chrono::system_clock::now();
	auto durationCV = std::chrono::duration_cast<std::chrono::microseconds>(endCV - startCV);
	std::cout << "cv predict 100 times spend: " << double(durationCV.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;
	kpCV.Clear();
	 
	const std::string allBin = "E:/code/Server223/hourglass-tooth/exp/tooth/model_convert/checkpoint_sim_all.bin";
	KeyPoint KP;
	KPOption optNN;
	auto startNNLoad = std::chrono::system_clock::now();
	KP.Build(allBin, optNN);
	auto endNNLoad = std::chrono::system_clock::now();
	auto durationNNLoad = std::chrono::duration_cast<std::chrono::microseconds>(endNNLoad - startNNLoad);
	std::cout << "ncnn load model time: " << double(durationNNLoad.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;
	std::vector<Point> keypointsNN;
	auto startNN = std::chrono::system_clock::now();
	if (KP.IsValid()) {
		for (int i = 1; i < testTimes; ++i) {
			KP.Predict(testImgPath, keypointsNN, optNN);
			//KP.Show(testImgPath, keypoints, opt);
		} 
	}
	auto endNN = std::chrono::system_clock::now();
	auto durationNN = std::chrono::duration_cast<std::chrono::microseconds>(endNN - startNN);
	std::cout << "ncnn predict 100 times spend: " << double(durationNN.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " s!" << std::endl;
	
	KP.Clear();

	return 0;
}
