#include <cstdlib>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/asio.hpp>

#include "caffe/proto/ipc.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using boost::asio::ip::tcp;

//boost::shared_ptr<tcp::socket> ServerInit(const char * port_str) {
tcp::socket* ServerInit(const char * port_str) {
	boost::asio::io_service io_service;

	int port_num = atoi(port_str);
	tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port_num));

	//boost::shared_ptr<tcp::socket> socket(new tcp::socket(io_service));
	tcp::socket* socket = new tcp::socket(io_service);
	acceptor.accept(*socket);
	
	std::cerr << "Server connect request received\n";

	return socket;
}

//boost::shared_ptr<tcp::socket> ClientInit(const char * host, int port_num) {
tcp::socket* ClientInit(const char * host, int port_num) {
	boost::asio::io_service io_service;

	tcp::resolver resolver(io_service);
	//string port_str = to_string(1543);
	stringstream ssport;
	ssport << port_num;
	tcp::resolver::query query(host, ssport.str());
	tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
	tcp::resolver::iterator end;

	//tcp::socket socket(io_service);
	//boost::shared_ptr<tcp::socket> socket(new tcp::socket(io_service));
	tcp::socket* socket = new tcp::socket(io_service);
	boost::system::error_code error = boost::asio::error::host_not_found;
	while (error && endpoint_iterator != end) {
		socket->close();
		socket->connect(*endpoint_iterator++, error);
	}
	if (error)
		throw boost::system::system_error(error);

	
	std::cerr << "Client connection established\n";

	return socket;
}

int CaffeSendMsg(tcp::socket& socket, CaffeIpc& Msg) {
	int SerializedSize = Msg.ByteSize();
	unsigned char * data_buf = new unsigned char[SerializedSize];
	Msg.SerializeToArray(data_buf, SerializedSize);
	boost::array<int, 1> size_buf;
	size_buf[0] = SerializedSize;
	boost::system::error_code error;
	int len = write(	socket, boost::asio::buffer(size_buf),
						boost::asio::transfer_exactly(sizeof(int)), error);
	len = write(socket, boost::asio::buffer(data_buf, SerializedSize), 
				boost::asio::transfer_exactly(size_buf[0]), error);
	delete[] data_buf;
	CHECK(error == 0) << "Error sending IPC message.\n";
//	if (error == boost::asio::error::eof)
//		return -1; // Connection closed cleanly by peer.
//	else if (error)
//		throw boost::system::system_error(error); // Some other error.
	return len;
}

int CaffeRcvMsg(tcp::socket& socket, CaffeIpc& Msg) {
	boost::system::error_code error;
	boost::array<int, 1> size_buf;
	int len = read(socket, boost::asio::buffer(size_buf), boost::asio::transfer_exactly(sizeof(int)), error);
	if (error == 0) {
		int alloc_size = size_buf[0];
		//std::cerr << "Receiving a msg of size " << alloc_size << std::endl;
		char * data_buf = new char[alloc_size];
		len = read(	socket, boost::asio::buffer(data_buf, alloc_size), 
					boost::asio::transfer_exactly(alloc_size ), error);
		//std::cerr << "Read " << len << " bytes\n";
		if (error == 0) {
			Msg.ParseFromArray(data_buf, alloc_size);
		}
		delete[] data_buf;
			
	}

	CHECK(error == 0) << "Error receiving IPC message.\n";
//	if (error == boost::asio::error::eof)
//		return -1; // Connection closed cleanly by peer.
//	else if (error)
//		throw boost::system::system_error(error); // Some other error.
	
	return len;
}


class OneRun {
public:
	OneRun() {bInit_ = false; }
		  
	void Init(	const string& model_file,
				const string& trained_file,
				string input_layer_name,
				int input_layer_bottom_idx,
				string output_layer_name,
				int output_layer_top_idx,
				int input_num_channels_idx,
				int input_height_idx,
				int input_width_idx);

	bool  Classify(const CaffeIpc_DataParam& ReqData, CaffeIpc_DataParam& RetData);

private:
//	void SetMean(const string& mean_file);

	std::vector<float> Predict(const vector<float>& data_input);

//	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_in_batch_;
	int num_channels_;
	int input_layer_idx_;
	int output_layer_idx_;
	int input_layer_bottom_idx_; // currently the index of the array of bottom blobs of the input layer
	int output_layer_top_idx_; // currently the index of the array of top blobs of the output layer
	int input_num_channels_;
	int input_width_;
	int input_height_;
	cv::Mat mean_;
	std::vector<string> labels_;
	std::vector<float> input_data_;
};

void OneRun::Init(	const string& model_file,
					const string& trained_file,
					string input_layer_name,
					int input_layer_bottom_idx,
					string output_layer_name,
					int output_layer_top_idx,
					int input_num_channels_idx,
					int input_height_idx,
					int input_width_idx
				) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

//	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

//	string input_layer_name = "conv1";
//	int input_layer_bottom_idx = 0;
//	string output_layer_name = "ip2";
//	int output_layer_top_idx = 0;
//	int input_num_channels_idx = 1;
//	int input_height_idx = 2;
//	int input_width_idx = 3;
	int input_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == input_layer_name) {
			input_layer_idx = layer_id;
			break;
		}
	}
	if (input_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << input_layer_name;			
	}
	input_layer_idx_ = input_layer_idx;
	input_layer_bottom_idx_ = input_layer_bottom_idx;
	input_num_channels_ = input_num_channels_idx;
	input_height_ = input_height_idx;
	input_width_ = input_width_idx;
	vector<Blob<float>*>  input_bottom_vec = net_->bottom_vecs()[input_layer_idx];
	num_channels_ = input_bottom_vec[input_layer_bottom_idx_]->shape(input_num_channels_);
	input_geometry_ = cv::Size(	input_bottom_vec[input_layer_bottom_idx_]->shape(input_height_), 
								input_bottom_vec[input_layer_bottom_idx_]->shape(input_width_));
	num_in_batch_ = input_bottom_vec[input_layer_bottom_idx_]->shape(0);
	
	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name;			
	}
	output_layer_idx_ = output_layer_idx;
	output_layer_top_idx_ = output_layer_top_idx;

	bInit_ = true;

}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the values in the output layer */
bool OneRun::Classify(const CaffeIpc_DataParam& ReqData, CaffeIpc_DataParam& RetData) {
	CHECK(bInit_) << "OneRun: Init must be called first\n";
	CHECK_EQ(ReqData.num_params(), ReqData.data_val_size()) << "OneRun data corrupted";
	//std::cerr << ReqData.num_params() << " data items reported. Received: " << ReqData.data_val_size() << ".\n";
	input_data_.clear();
	for (int id = 0; id < ReqData.num_params(); id++) {
		input_data_.push_back(ReqData.data_val(id));

		//std::cout << ReqData.data_val(id) << ", ";
	}
			//std::cout << std::endl;
	std::vector<float> output = Predict(input_data_);
//	CaffeIpc Msg2;
//	Msg2.set_type(CaffeIpc_MsgType_PREDICT_RESULT);
//	CaffeIpc_DataParam& RetData = *(Msg2.mutable_data_param());
	for (uint ir = 0; ir < output.size(); ir++) {
		RetData.add_data_val(output[ir]);
	}
	RetData.set_num_params((int)output.size());
	
	return true;
//	CaffeSendMsg(*socket, Msg2);

//	std::ifstream ifs_data(data_file.c_str());
//	CHECK(ifs_data) << "Unable to open labels file " << data_file;
//	string line;
//	while (std::getline(ifs_data, line)) {
//		float datum = strtof(line.c_str(), NULL);
//		input_data_.push_back(datum);
//	}
//
//	std::vector<float> output = Predict(input_data_);
//
//	for (int i = 0; i < output.size(); i++) {
//		std::cout << output[i] << ", ";
//	}
//	std::cout << std::endl;
//	return output;
}

std::vector<float> OneRun::Predict(const vector<float>& data_input) {
	Blob<float>* input_layer = net_->bottom_vecs()[input_layer_idx_][input_layer_bottom_idx_];
	float* p_begin = input_layer->mutable_cpu_data();  // ->cpu_data();

	int id = 0;
	for (int ib = 0; ib < num_in_batch_; ib++) {
		if (id >= data_input.size()) id = 0; // keep the data a multiple of the num in batch
		for (int ic = 0; ic < num_channels_; ic++) {
			for (int ih = 0; ih < input_geometry_.height; ih++) {
				for (int iw = 0; iw < input_geometry_.width; iw++, id++) {
					*p_begin++ = data_input[id];
				}
			}
		}
	}
  
	net_->ForwardFromTo(input_layer_idx_, output_layer_idx_);

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->top_vecs()[output_layer_idx_][output_layer_top_idx_]; // net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
//void OneRun::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//  Blob<float>* input_layer = net_->input_blobs()[0];
//
//  int width = input_layer->width();
//  int height = input_layer->height();
//  float* input_data = input_layer->mutable_cpu_data();
//  for (int i = 0; i < input_layer->channels(); ++i) {
//    cv::Mat channel(height, width, CV_32FC1, input_data);
//    input_channels->push_back(channel);
//    input_data += width * height;
//  }
//}

//void OneRun::Preprocess(const cv::Mat& img,
//                            std::vector<cv::Mat>* input_channels) {
//  /* Convert the input image to the input image format of the network. */
//  cv::Mat sample;
//  if (img.channels() == 3 && num_channels_ == 1)
//    cv::cvtColor(img, sample, CV_BGR2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 1)
//    cv::cvtColor(img, sample, CV_BGRA2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 3)
//    cv::cvtColor(img, sample, CV_BGRA2BGR);
//  else if (img.channels() == 1 && num_channels_ == 3)
//    cv::cvtColor(img, sample, CV_GRAY2BGR);
//  else
//    sample = img;
//
//  cv::Mat sample_resized;
//  if (sample.size() != input_geometry_)
//    cv::resize(sample, sample_resized, input_geometry_);
//  else
//    sample_resized = sample;
//
//  cv::Mat sample_float;
//  if (num_channels_ == 3)
//    sample_resized.convertTo(sample_float, CV_32FC3);
//  else
//    sample_resized.convertTo(sample_float, CV_32FC1);
//
//  cv::Mat sample_normalized;
//  cv::subtract(sample_float, mean_, sample_normalized);
//
//  /* This operation will write the separate BGR planes directly to the
//   * input layer of the network because it is wrapped by the cv::Mat
//   * objects in input_channels. */
//  cv::split(sample_normalized, *input_channels);
//
//  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//        == net_->input_blobs()[0]->cpu_data())
//    << "Input channels are not wrapping the input layer of the network.";
//}

#ifdef CAFFE_ONE_RUN_MAIN
int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0]
				  << " port_num " << std::endl;
		return 1;
	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
	tcp::socket* socket = ServerInit(argv[1]);

	OneRun classifier;
	for (;;) {
		CaffeIpc Msg1;
		CaffeRcvMsg(*socket, Msg1);

		if (Msg1.type() == CaffeIpc_MsgType_END_NET) {
			std::cerr << "Close connection request sent. Bye. \n";
			break;
		}
		else if (Msg1.type() == CaffeIpc_MsgType_INIT_NET) {
			std::cout << "Received name " << Msg1.init_net_params().input_layer_name() << std::endl;
			const CaffeIpc_InitNetParams& InitMsg = Msg1.init_net_params();
			classifier.Init(InitMsg.model_file(), InitMsg.trained_file(),
							InitMsg.input_layer_name(), InitMsg.input_layer_bottom_idx(),
							InitMsg.output_layer_name(), InitMsg.output_layer_top_idx(),
							InitMsg.input_num_channels_idx(), InitMsg.input_height_idx(), 
							InitMsg.input_width_idx() );
			CaffeIpc Msg2;
			Msg2.set_type(CaffeIpc_MsgType_INIT_NET_DONE);
			CaffeSendMsg(*socket, Msg2);
		}
		else if (Msg1.type() == CaffeIpc_MsgType_NET_PREDICT) {
			const CaffeIpc_DataParam& ReqData = Msg1.data_param();
			CaffeIpc Msg2;
			Msg2.set_type(CaffeIpc_MsgType_PREDICT_RESULT);
			CaffeIpc_DataParam& RetData = *(Msg2.mutable_data_param());
			classifier.Classify(ReqData, RetData);
//			std::cout << ReqData.num_params() << " data items reported. Received: " << ReqData.data_val_size() << ".\n";
//			for (int id = 0; id < ReqData.num_params(); id++) {
//				std::cout << ReqData.data_val(id) << ", ";
//			}
//			std::cout << std::endl;
			CaffeSendMsg(*socket, Msg2);
		}

	}

//  string model_file   = "/home/abba/caffe/toys/ValidConv/train.prototxt";
//  string trained_file = "/guten/data/ValidConv.caffemodel";
//  string mean_file    = "dummymean";
//  string label_file   = "dummylabel";
//  string data_file		= "/home/abba/caffe/toys/ValidConv/OneBoard.csv";
//  OneRun classifier(model_file, trained_file);

  //std::vector<float> output = classifier.Classify(data_file);

//  /* Print the top N predictions. */
//  for (size_t i = 0; i < predictions.size(); ++i) {
//    Prediction p = predictions[i];
//    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//              << p.first << "\"" << std::endl;
//  }
}
#endif // CAFFE_ONE_RUN_MAIN