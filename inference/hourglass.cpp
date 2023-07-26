#include "ncnn/net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include "opencv2/opencv.hpp"
#endif
#include "opencv2/dnn/dnn.hpp"
#include "hourglass.h"


KeyPointCV::KeyPointCV()
{
	net_ = nullptr;
}

KeyPointCV::~KeyPointCV()
{
	Clear();
}

bool KeyPointCV::Build(std::string modelPath, const KPOption& opt)
{
	// Load Model
	cv::dnn::Net* net = new cv::dnn::Net();
	*net = cv::dnn::readNetFromONNX(modelPath);
	if (net->empty()) {
		// load model failed
		return false;
	}
	net_ = net;

	return true;
}

bool KeyPointCV::IsValid() const
{
	if (!net_) {
		return false;
	}
	return true;
}

void KeyPointCV::Clear()
{
	if (net_) {
		delete static_cast<cv::dnn::Net*>(net_);
		net_ = nullptr;
	}
}

namespace {
	void PreProcess(cv::Mat& img, int imgSize = 256)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::resize(img, img, cv::Size(imgSize, imgSize));

		std::vector<float> meanValue{ 123.675, 116.28, 103.53 };                  // { 0.485, 0.456, 0.406 };
		std::vector<float> stdValue{ 1.0 / 58.395, 1.0 / 57.120, 1.0 / 57.375 };  // { 0.229, 0.224, 0.225 };

		std::vector<cv::Mat> rgbChannels(3);
		cv::split(img, rgbChannels);
		for (auto i = 0; i < rgbChannels.size(); i++)
		{
			rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, stdValue[i], -meanValue[i] * stdValue[i]);
		}

		cv::merge(rgbChannels, img);

	}
}

bool KeyPointCV::Predict(std::string imgPath, std::vector<Point>& keypoints, const KPOption& opt)
{
	keypoints.clear();
	cv::Mat srcImg = cv::imread(imgPath);
	if (srcImg.empty()) {
		// read img failed
		return false;
	}
	int imgW = srcImg.cols, imgH = srcImg.rows;

	PreProcess(srcImg, opt.targetSize);
	cv::dnn::Net* net = static_cast<cv::dnn::Net*>(net_); 
	cv::Mat inpBlob = cv::dnn::blobFromImage(srcImg);
	net->setInput(inpBlob);
	cv::Mat res = net->forward(); 

	// Parse result
	// cv::MatSize resSize = res.size;   // [ B, N_Stack, nKps, outSize_H, outSize_W ]
	int nStack = res.size[1];
	int nKps = res.size[2];
	int outSize = res.size[3];
	float wRatio = (imgW * 1.0) / (opt.targetSize * 1.0);
	float hRatio = (imgH * 1.0) / (opt.targetSize * 1.0);
	float ratio = (opt.targetSize * 1.0) / outSize;  // 缩放比例
	 
	int lastLayerIdx = (nStack - 1) * nKps * outSize * outSize;
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
		float px = locX * ratio * wRatio;
		float py = locY * ratio * hRatio;
		keypoints.push_back(Point(px, py)); 
	}

	return true;

}

void KeyPointCV::Show(std::string imgPath, const std::vector<Point>& keypoints, const KPOption& opt)
{
	cv::Mat srcImg = cv::imread(imgPath);
	int imgW = srcImg.cols;
	int imgH = srcImg.rows;

	int n = keypoints.size();

	for (int idx = 0; idx < n; ++idx) {
		float x = keypoints[idx].x;
		float y = keypoints[idx].y;
		cv::circle(srcImg, cv::Point2i(x, y), 5, (0, 0, 255), -1);
	}

	cv::imshow("img", srcImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	//cv::imwrite("D:/srcImg.png", srcImg);
}


struct KeyPoint::CPrivate
{
	ncnn::Net net_;
	ncnn::Extractor ex_ = net_.create_extractor();
};

KeyPoint::KeyPoint() : pD_(new CPrivate)
{
	 
}

KeyPoint::~KeyPoint()
{
	Clear();  
}

bool KeyPoint::Build(std::string modelPath, const KPOption& opt)
{
	// net.opt.use_vulkan_compute = true;
	//// 1.Load model
	//// --- method 1 ---
	//net.load_param(paramPath.c_str());
	//net.load_model(binPath.c_str());

	//// --- method 2 ---
	//net.load_param_bin(paramBinPath.c_str());
	//net.load_model(binPath.c_str());

	//// --- method 3 ---
	// // need #inlucde "test_sim.mem.h"
	//net.load_param(test_sim_param_bin);
	//net.load_model(test_sim_bin);

	//// --- method 4 --- 
	// Load Model
	// mpD->net_.opt.use_vulkan_compute = true;

	FILE* fp = fopen(modelPath.c_str(), "rb");
	pD_->net_.load_param_bin(fp);
	pD_->net_.load_model(fp);
	fclose(fp);

	/*const std::vector<ncnn::Blob>& netBlobs = pD_->net_.blobs();
	const std::vector<ncnn::Layer*>& netLayers = pD_->net_.layers();
	std::cout << " blobs: " << netBlobs.size() << " layers: " << netLayers.size() << std::endl;*/

	pD_->ex_.clear();
	pD_->ex_ = pD_->net_.create_extractor();
	pD_->ex_.set_num_threads(4);

	return true;
}

bool KeyPoint::IsValid() const
{ 
	if (!pD_) {
		return false;
	}
	return true;
}

void KeyPoint::Clear()
{
	if (pD_) {
		pD_->ex_.clear();
		pD_->net_.clear();
		delete pD_; 
		pD_ = nullptr;
	}
}

namespace {
	bool Predict_(ncnn::Extractor& ex, cv::Mat& srcImg, std::vector<Point>& keypoints, const KPOption& opt)
	{
		keypoints.clear();

		// 1. Preprocess img
#ifdef _DEBUG
		if (false) {
			cv::imshow("img", srcImg);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
#endif
		ncnn::Mat input = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR2RGB, srcImg.cols, srcImg.rows, opt.targetSize, opt.targetSize);
		input.substract_mean_normalize(opt.MEANS, opt.STD);

		// 2. Inference
		// ex.input("img", input);
		ex.input(0, input);
		// get output
		ncnn::Mat res;

		// ex.extract("outs", res);
		ex.extract(164, res);

		// 3. Parse result
		int imgW = srcImg.cols;
		int imgH = srcImg.rows;
		int nChannel = res.c;
		float wRatio = (imgW * 1.0) / (opt.targetSize * 1.0);
		float hRatio = (imgH * 1.0) / (opt.targetSize * 1.0);
		float ratio = (opt.targetSize * 1.0) / res.w;  // 缩放比例
		int startChannel = nChannel - res.c;   // Just deal last layer 

		for (int idx = startChannel; idx < nChannel; ++idx) {
			float* cData = res.channel(idx);
			float maxV = std::numeric_limits<float>::lowest();
			int maxX = 0;
			int maxY = 0;
			for (int i = 0; i < res.w; ++i) {
				for (int j = 0; j < res.h; ++j) {
					int idx = i * res.w + j;
					if (cData[idx] > maxV) {
						maxX = j;
						maxY = i;
						maxV = cData[idx];
					}
				}
			}

			float px = maxX * ratio * wRatio;
			float py = maxY * ratio * hRatio;
			keypoints.push_back(Point(px, py));
		}

		return true;
	}
}

bool KeyPoint::Predict(std::string imgPath, std::vector<Point>& keypoints, const KPOption& opt)
{
	cv::Mat srcImg = cv::imread(imgPath);
	if (srcImg.empty()) {
		// read img failed
		return false;
	}

	return Predict_(pD_->ex_, srcImg, keypoints, opt);

}

void KeyPoint::Show(std::string imgPath, const std::vector<Point>& keypoints, const KPOption& opt)
{
	cv::Mat srcImg = cv::imread(imgPath);
	int imgW = srcImg.cols;
	int imgH = srcImg.rows;

	int n = keypoints.size();

	for (int idx = 0; idx < n; ++idx) { 
		float x = keypoints[idx].x;
		float y = keypoints[idx].y;
		cv::circle(srcImg, cv::Point2i(x, y), 5, (0, 0, 255), -1);
	}

	cv::imshow("img", srcImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	//cv::imwrite("D:/srcImg.png", srcImg);
}



