#include <string>
#include <vector> 

class KPOption
{
public:
	const std::vector<std::string> classNames = { "L", "M", "R" };
	const int targetSize = 256; 
	const float MEANS[3] = { 123.675, 116.28, 103.53 };
	const float STD[3] = { 1.0 / 58.395, 1.0 / 57.120, 1.0 / 57.375 }; 
};

class Point
{
public:
	float x, y;
	Point(float px = 0.0, float py = 0.0) : x(px), y(py) {}
};

class KeyPointCV
{
public:

	KeyPointCV();
	~KeyPointCV();

	/*
	* @brief: load the keypoint model that has been conveted to onnx format
	*
	* @param[in] modelPath: the file path of model
	* @param[in] opt: some params for yolact net
	*
	*/
	bool Build(std::string modelPath, const KPOption& opt);

	/*
	* @brief: judge the model's whether load Ok
	*/
	bool IsValid() const;

	void Clear();

	/*
	* @brief: predict the img
	*
	* @param[in] imgPath: the file path of image
	* @param[out] keypoints: the predict result
	* @param[in] opt: some params for predict and network
	*
	*/
	bool Predict(std::string imgPath, std::vector<Point>& keypoints, const KPOption& opt);

	/*
	* @brief: show the predict res
	*/
	void Show(std::string imgPath, const std::vector<Point>& keypoints, const KPOption& opt);

private:
	void* net_;
};
 
class KeyPoint
{
public:

	KeyPoint();
	~KeyPoint();

	/*
	* @brief: load the keypoint model that has been conveted to ncnn format
	*
	* @param[in] modelPath: the file path of model
	* @param[in] opt: some params for yolact net
	*
	*/
	bool Build(std::string modelPath, const KPOption& opt);

	/*
	* @brief: judge the model's whether load Ok
	*/
	bool IsValid() const;

	void Clear();

	/*
	* @brief: predict the img
	*
	* @param[in] imgPath: the file path of image
	* @param[out] keypoints: the predict result
	* @param[in] opt: some params for predict and network
	*
	*/
	 bool Predict(std::string imgPath, std::vector<Point>& keypoints, const KPOption& opt);

	/*
	* @brief: show the predict res
	*/
	void Show(std::string imgPath, const std::vector<Point>& keypoints, const KPOption& opt);

private:
	struct CPrivate;
	CPrivate* pD_;  
};
