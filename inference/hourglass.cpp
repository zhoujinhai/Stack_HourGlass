#include "ncnn/net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "ncnn/simpleocv.h"
#else
#include "opencv2/opencv.hpp"
#endif

#include "hourglass.h"


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
	// Load Model
	// mpD->net_.opt.use_vulkan_compute = true;

	FILE* fp = fopen(modelPath.c_str(), "rb");
	pD_->net_.load_param_bin(fp);
	pD_->net_.load_model(fp);
	fclose(fp);

	const std::vector<ncnn::Blob>& netBlobs = pD_->net_.blobs();
	const std::vector<ncnn::Layer*>& netLayers = pD_->net_.layers();
	std::cout << " blobs: " << netBlobs.size() << " layers: " << netLayers.size() << std::endl;

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
	}
}


bool Predict_(ncnn::Extractor& ex, cv::Mat& srcImg, std::vector<Point>& keypoints, const KPOption& opt)
{

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
	ex.input("img", input);
	// get output
	ncnn::Mat res;

	ex.extract("outs", res);

	// 3. Parse result
	int imgW = srcImg.cols;
	int imgH = srcImg.rows;
	int nChannel = res.c;
	float wRatio = (imgW * 1.0) / (opt.targetSize * 1.0);
	float hRatio = (imgH * 1.0) / (opt.targetSize * 1.0);
	float ratio = (opt.targetSize * 1.0) / res.w;  // 缩放比例
	int startChannel = nChannel - opt.numKp;   // Just deal last layer 

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



