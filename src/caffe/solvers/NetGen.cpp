#include <cstdlib>


#include <boost/random.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/algorithm/string.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <caffe/caffe.hpp>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <opencv2/core/core.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include "H5Cpp.h"


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/sgd_solvers.hpp"



#include "caffe/GenSeed.pb.h"


#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

//#define USE_CPU
	
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

typedef unsigned long long u64;
typedef boost::chrono::duration<double> sec;

struct LiveSnapshot {
	vector<int> num_nodes_in_layer;
	float lr;
	vector<vector<vector<float> > > weights; // layers of blobs of floats. Note unlike 2D array, layers may have different number of blobs
};


SolverAction::Enum MakeSolverStop()  {
  return SolverAction::STOP;
}

SolverAction::Enum SolverKeepGoing()  {
  return SolverAction::NONE;
}


class NetGenSolver : public SGDSolver<float> {
public:
	explicit NetGenSolver(const SolverParameter& param)
		: SGDSolver<float>(param) { Init(); }
	explicit NetGenSolver(const string& param_file)
		: SGDSolver<float>(param_file) { Init(); }
//	virtual ~NetGenSolver() {
//		instance_count_--;
//		std::cerr << "NetGenSolver deleted. Instance count is " << instance_count_ << "\n";
//	}
	
	void set_loss_layer(int loss_layer) { loss_layer_ = loss_layer; }
	void set_run_time(double run_time) {run_time_ = run_time; }
	float get_avg_loss() { return loss_sum_ / loss_sum_count_; }
	void get_net(shared_ptr<Net<float> >& net) {net = this->net_; }
	Net<float> * get_test_net() {return this->test_nets_[0].get(); }
	void reset_loss_sum() {	SetActionFunction(SolverKeepGoing);
							timer_.stop(); timer_.start();
							loss_sum_= 0.0f; loss_sum_count_ = 0.0f; 
							iter_ = 0; }

protected:
//	static int instance_count_;
	int loss_layer_;
	float loss_sum_;
	float loss_sum_count_;
	double run_time_;
	boost::timer::cpu_timer timer_;
	
	
	virtual void ApplyUpdate();
	void Init();
  
  DISABLE_COPY_AND_ASSIGN(NetGenSolver);
};

void NetGenSolver::Init()
{
//	instance_count_++;
	loss_layer_ = -1;
	loss_sum_= 0;
	loss_sum_count_ = 0;
	run_time_ = 0.0;
	SetActionFunction(SolverKeepGoing);
}
 
void NetGenSolver::ApplyUpdate()
{
	SGDSolver::ApplyUpdate();
	CHECK_GT(loss_layer_, 0) << "Error: Loss layer for NetGenSolver must be set before calling Solve\n";
	
	float loss = *(net_->top_vecs()[loss_layer_][0]->cpu_data());
	loss_sum_ += loss;
	loss_sum_count_++;
	sec seconds = boost::chrono::nanoseconds(timer_.elapsed().user + timer_.elapsed().system);
	if (seconds.count() > run_time_) {
		//iter_ = INT_MAX - 100;
		requested_early_exit_ = true;
		SetActionFunction(MakeSolverStop);
	}
}

//int NetGenSolver::instance_count_ = 0;

class NGNet {
public:
	NGNet(	) {
	}
	
//	virtual ~NGNet() {
//		std::cerr << "NGNet destructor called\n";
//	}

	void Gen(vector<int>& num_nodes_in_layer, int mod_idx_idx, float lr, CaffeGenSeed * config);
	float DoRun(bool bIntersection, double growth_factor);
	float TestOnly();
	void CopyTrainWeightsToTestNet();
	void SetWeights(NGNet* p_ng_net);
	void CopyWeightsAndAddDupLayer(NGNet* p_ng_net, int i_dup_layer);
	int get_ipx_layer_idx(int idx_idx) { return ip_layer_idx_arr_[idx_idx]; }
	void ShakeupWeights();
	void MakeLiveSnapshot(LiveSnapshot& net_snap);
	void SetWeightsFromLiveSnapshot(LiveSnapshot& net_snap);
	void get_first_and_last_node_sizes(int& first_layer_size, int& last_layer_size);
	

	
	void Launch();
	void Init();
	
private:
	shared_ptr<NetGenSolver> solver_;
	shared_ptr<Net<float> > net_;
	vector<int> ip_layer_idx_arr_;
	int ip_layer_idx_idx_;
	int num_test_cases_;
	string output_model_filename_;
	string output_prototxt_filename_;
	string train_proto_str_;
	string test_proto_str_;
};

class NetGen {
public:
	NetGen(string config_file_name, int a_max_no_progress) {
		bInit_ = false; 
		config_file_name_ = config_file_name;
		max_no_progress = a_max_no_progress;
	}

	void PreInit();
	void Init();

	bool  Classify();

private:

	std::vector<pair<float, int> > Predict();

	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	vector<shared_ptr<NGNet> > nets_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	vector<string> words_;
	vector<vector<float> > words_vecs_;
	string word_vector_file_name_;
	int words_per_input_;
	string config_file_name_;
	int max_no_progress;
	
	int GetClosestWordIndex(vector<float>& VecOfWord, int num_input_vals, 
							vector<pair<float, int> >& SortedBest,
							int NumBestKept);

};

// returns a rnd num from -0.5f to 0.5f uniform dist, don't know (care) about endpoints
float rn(void)
{
  boost::mt19937 rng(43);
  static boost::uniform_01<boost::mt19937> zeroone(rng);
  return ((float)zeroone() - 0.5f);
}

void NGNet::CopyWeightsAndAddDupLayer(NGNet* p_ng_net, int i_dup_layer) {
	// when we duplicate there are two layers with the same num_output = output nodes
	// the first is left untouched w.r.t. weights and the second starts as an identity matrix
	// with some random perturbation
	Net<float> * src_net = p_ng_net->net_.get();
	
	for (int il = 0; il < ip_layer_idx_arr_.size(); il++) {
		if (il == i_dup_layer) {
			// At first I thought that I would need to add initialize all non-diagonal weights to some
			// small value off zero - including the biases. I was afraid they would never get 
			// the gradients to move off zero. I have determined empirically that there is no need
			int mod_idx = ip_layer_idx_arr_[il];
			Layer<float>* layer = net_->layers()[mod_idx].get();
			for (int ib=0; ib < layer->blobs().size(); ib++) {
				Blob<float>* weights = layer->blobs()[ib].get() ;
				if (weights->count() == 0) {
					continue;
				}
				float * pw = weights->mutable_cpu_data();
				int num_inputs = 1;
				if (weights->num_axes() == 1) { // bias
					for (int iw = 0; iw < weights->count(); iw++) {
						pw[iw] = 0.0f;
					}
				}
				else {
					num_inputs = weights->shape(1);
					int num_outputs = weights->shape(0);
					CHECK_EQ(num_inputs, num_outputs) << "The number of inputs and outputs to a duplicated layer should be equal";
					for (int jw = 0; jw < num_outputs; jw+=2) {
						for (int iw = 0; iw < num_inputs; iw++) {
							pw[(jw * num_inputs) + iw] = ((iw == jw) ? 1.0f : 0.0f);
						}
					}
				}
			}
		}
		else {
			int src_il = il;
			if (il > i_dup_layer) {
				src_il = il - 1;
			}
			int src_mod_idx = p_ng_net->ip_layer_idx_arr_[src_il];
			int mod_idx = ip_layer_idx_arr_[il];
			Layer<float>* layer = net_->layers()[mod_idx].get();
			Layer<float>* src_layer = src_net->layers()[src_mod_idx].get();
			for (int ib=0; ib < layer->blobs().size(); ib++) {
				Blob<float>* weights = layer->blobs()[ib].get() ;
				Blob<float>* src_weights = src_layer->blobs()[ib].get() ;
				if (weights->count() > 0) {
					const float * psw = src_weights->cpu_data();
					float * pw = weights->mutable_cpu_data();
					for (int iw = 0; iw < weights->count(); iw++) {
						pw[iw] = psw[iw];
					}
				}
			}
		}
	}
}

void NGNet::SetWeights(NGNet* p_ng_net)
{
	Net<float> * launch_net = p_ng_net->net_.get();
	
	int mod_idx = ((ip_layer_idx_idx_ >= 0) ? ip_layer_idx_arr_[ip_layer_idx_idx_] : -1);
	int mod_post_idx = ((ip_layer_idx_idx_ >= 0) ? ip_layer_idx_arr_[ip_layer_idx_idx_+1] : -1);
	for (int il = 0; il < launch_net->layers().size(); il++) {
		Layer<float>* layer = launch_net->layers()[il].get();
		for (int ib=0; ib < layer->blobs().size(); ib++) {
			Blob<float>* weights = layer->blobs()[ib].get() ;
			Blob<float>* new_weights = net_->layers()[il]->blobs()[ib].get() ;
			if (weights->count() == 0) {
				continue;
			}
			if (weights->count() == new_weights->count()) {
				const float * pw = weights->cpu_data();
				float * pnw = new_weights->mutable_cpu_data();
				for (int iw = 0; iw < weights->count(); iw++) {
					pnw[iw] = pw[iw];
				}
			}
			else if (new_weights->count() == (weights->count() * 2)) {
				const float * pw = weights->cpu_data();
				float * pnw = new_weights->mutable_cpu_data();
//#pragma message "don't leave cFrac 0"		
				float cFrac = 0.1f; // must be less than .5. Result look good at 0.1
				// the idea here is that we are doubling the number of output nodes from
				// the mod_idx layer so each output node should have roughly half the activation
				// layer. We do this by halving each weight and then adding some random 
				// delta
				// the next layer, the mod_post_idx should have the same weights roughly
				// because the inputs already have half the activation
				float rtwice = 1.0f;
				if (il == mod_idx) {
					rtwice = 0.5f;
				}
				int num_inputs = 1;
				if (weights->num_axes() > 1) { 
					num_inputs = weights->shape(1);
				}
				int num_outputs = weights->shape(0);
				for (int jw = 0; jw < num_outputs; jw++) {
					for (int iw = 0; iw < num_inputs; iw++) {
						//float adj = rn() * rmod;
						//adj = 0.0f;
						int jwo = jw * 2;
						if (il == mod_idx) {
							float val = pw[(jw * num_inputs) + iw];
							float adj = val * cFrac * rn();
							pnw[(jwo * num_inputs) + iw] 
								= (val * rtwice) + adj;
							pnw[((jwo + 1) * num_inputs) + iw] 
								= (val * rtwice) - adj;
						}
						else if (il == mod_post_idx) {
							int ii = (jw * num_inputs) + iw;
							pnw[ii * 2] = pw[ii];
							pnw[(ii * 2) + 1] = pw[ii];
						}
					}
				}
			}
			else if (new_weights->count() == (weights->count() / 2)) {
				const float * pw = weights->cpu_data();
				float * pnw = new_weights->mutable_cpu_data();
				// in this case we are combining nodes for the output of mod_idx
				// so the activation of the combined nodes will be higher 
				// however we have to double the weights on the mod_idx
				// because we don't want the combination of weights but their average
				int num_inputs = 1;
				if (weights->num_axes() > 1) { 
					num_inputs = weights->shape(1);
				}
				int num_outputs = weights->shape(0);
				if (il == mod_idx) {
					for (int jw = 0; jw < num_outputs; jw+=2) {
						for (int iw = 0; iw < num_inputs; iw++) {
							int jwo = jw / 2;
							pnw[(jwo * num_inputs) + iw] 
								=		(pw[(jw * num_inputs) + iw]  
									+	pw[((jw + 1) * num_inputs) + iw]) ;
							
						}
					}
				}
				else if (il == mod_post_idx) {
					int num_new_inputs = num_inputs / 2;
					for (int jw = 0; jw < num_outputs; jw++) {
						for (int iw = 0; iw < num_new_inputs; iw++) {
							int ii = (jw * num_new_inputs) + iw;
							pnw[ii] = pw[ii * 2] + pw[(ii * 2) + 1];
							
						}
					}
				}
			}
//			{
//				// test code
//				if ((il == 4) || (il == 6)) {
//					const float * pw = weights->cpu_data();
//					std::cerr << "old weights: ";
//					for (int iw = 0; iw < weights->count(); iw++) {
//						std::cerr << pw[iw] << ", ";
//					}
//					std::cerr << std::endl;
//					const float * pnw = new_weights->cpu_data();
//					std::cerr << "new weights: ";
//					for (int iw = 0; iw < new_weights->count(); iw++) {
//						std::cerr << pnw[iw] << ", ";
//					}
//					std::cerr << std::endl;
//				}
//			}

		}
	}
}

void NGNet::ShakeupWeights()
{
	float c_max_shake = 0.5f; 
	for (int il = 0; il < net_->layers().size(); il++) {
		Layer<float>* layer = net_->layers()[il].get();
		for (int ib=0; ib < layer->blobs().size(); ib++) {
			Blob<float>* weights = layer->blobs()[ib].get() ;
			if (weights->count() > 0) {
				float * pw = weights->mutable_cpu_data();
				float max_val = -FLT_MAX;
				float min_val = FLT_MAX;
				for (int iw = 0; iw < weights->count(); iw++) {
					float val = pw[iw];
					float adj = val * c_max_shake * rn();
					pw[iw] = val + adj ;
					if (val > max_val) {
						max_val = val;
					}
					if (val < min_val) {
						min_val = val;
					}
				}
				for (int iw = 0; iw < weights->count(); iw++) {
					if ((rand() % 30) == 0) {
						pw[iw] = max_val ;
					}
					else if ((rand() % 30) == 0) {
						pw[iw] = min_val ;
					}
					else if ((rand() % 30) == 0) {
						pw[iw] = 0.0f ;
					}
				}
			}
		}
	}
	
}


void NetGen::PreInit()
{
#ifdef USE_CPU
	Caffe::set_mode(Caffe::CPU); 
	return;
#endif // USE_CPU
	
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU); 
#else
	Caffe::set_mode(Caffe::GPU);
#endif
}
 
const bool cb_ReLU = true;
const bool cb_Sigmoid = true;
const bool cb_drop = true;

const string cHD5Str1 = "name: \"GramPosValid\"\n"
					"layer {\n"
					"	name: \"data\"\n"
					"	type: \"HDF5Data\"\n"
					"	top: \"data\"\n"
					"	top: \"label\"\n"	
					"	include {\n";
const string cHD5Str2 = "	}\n"
					"	hdf5_data_param {\n"
					"		source: ";
const string cHD5StrBatch128 = "\n"
					"		batch_size: 128\n"
					"	}\n"
					"}\n";
const string cHD5StrBatch1 = "\n"
					"		batch_size: 1\n"
					"	}\n"
					"}\n";

string CreateHD5TrainStr(string train_file)
{
	string ret_str = cHD5Str1 
					+ "		phase: TRAIN\n"
					+ cHD5Str2
					+ "\"" + train_file + "\""
					+ cHD5StrBatch128;
	
	return ret_str;
					
}

string CreateHD5TestStr(string test_file)
{
	string ret_str = cHD5Str1 
					+ "		phase: TEST\n"
					+ cHD5Str2
					+ "\"" + test_file + "\""
					+ cHD5StrBatch128; // can be cHD5StrBatch1 but it's nice to see a real result in the log file
					
	return ret_str;
}

string CreateReLUStr()
{
	return string(	"layer {\n"
					"  name: \"squash#id#\"\n"
					"  type: \"ReLU\"\n"
					"  bottom: \"ip#id#\"\n"
					"  top: \"ip#id#s\"\n"
					"}\n");

}

string CreateSigmoidStr()
{
	return string(	"layer {\n"
					"  name: \"squash#id#\"\n"
					"  type: \"Sigmoid\"\n"
					"  bottom: \"ip#id#\"\n"
					"  top: \"ip#id#s\"\n"
					"}\n");

}

string CreateDropStr()
{
	return string(	"layer {\n"
					"  name: \"drop#id#\"\n"
					"  type: \"Dropout\"\n"
					"  bottom: \"ip#id#s\"\n"
					"  top: \"ip#id#s\"\n"
					"  dropout_param {\n"
					"    dropout_ratio: #drop_rate#\n"
					"  }\n"
					"  include {\n"
					"    phase: TRAIN\n"
					"  }\n"
					"}\n");


}

string AddSoftmaxAndAccuracyStr (int prev_id, int& layers_so_far, int num_accuracy_candidates)
{
	string modi = 
		"layer {\n"
		"  name: \"loss\"\n"
		"  type: \"SoftmaxWithLoss\"\n"
		"  bottom: \"ip#0#s\"\n"
		"  bottom: \"label\"\n"
		"  top: \"sm-loss\"\n"
		"}\n"
		"layer {\n"
		"  name: \"accuracy\"\n"
		"  type: \"Accuracy\"\n"
		"  bottom: \"ip#0#s\"\n"
		"  bottom: \"label\"\n"
		"  top: \"accuracy\"\n"
		"  accuracy_param {\n"
		"     top_k: #topk#\n"
		"  }"
		"}\n";

	string sid = boost::lexical_cast<string>(prev_id);
	modi = boost::replace_all_copy(modi, "#0#", sid);
	string stopk = boost::lexical_cast<string>(num_accuracy_candidates);
	modi = boost::replace_all_copy(modi, "#topk#", stopk);
	
	layers_so_far += 2;
	return modi;

}

string AddCrossEntropyAndEuclideanStr (int prev_id, int& layers_so_far)
{
	string modi = 
		"layer {\n"
		"  name: \"loss\"\n"
		" type: \"SigmoidCrossEntropyLoss\"\n"
		"  bottom: \"ip#0#\"\n"
		"  bottom: \"label\"\n"
		"  top: \"cross_entropy_loss\"\n"
		"  loss_weight: 1\n"
		"}\n"
		"layer {\n"
		"  name: \"readable-loss\"\n"
		"  type: \"EuclideanLoss\"\n"
		"  bottom: \"ip#0#s\"\n"
		"  bottom: \"label\"\n"
		"  top: \"el_error\"\n"
		"  loss_weight: 0\n"
		"}\n";
	string sid = boost::lexical_cast<string>(prev_id);
	modi = boost::replace_all_copy(modi, "#0#", sid);
	
	layers_so_far += 2;
	return modi;

}

string AddInnerProductStr(	bool b_ReLU, bool b_Sigmoid, bool b_drop, int id, 
							int num_output, int& layers_so_far, float dropout) 
{
	string frame = 
		"layer {\n"
		"  name: \"ip#id#\"\n"
		"  type: \"InnerProduct\"\n"
		"  bottom: \"#prev_top#\"\n"
		"  top: \"ip#id#\"\n"
		"  param {\n"
		"    lr_mult: 1\n"
		"  }\n"
		"  param {\n"
		"    lr_mult: 2\n"
		"  }\n"
		"  inner_product_param {\n"
		"    num_output: #num_output#\n"
		"    weight_filler {\n"
		"      type: \"xavier\"\n"
		"    }\n"
		"    bias_filler {\n"
		"      type: \"constant\"\n"
		"    }\n"
		"  }\n"
		"}\n";
	string modi =		frame + (b_ReLU ? CreateReLUStr() : "") 
					+	(b_Sigmoid ? CreateSigmoidStr() : "") 
					+	(b_drop ? CreateDropStr() : "");
	
	string sid = boost::lexical_cast<string>(id);
	string s_input = string("ip") + boost::lexical_cast<string>(id-1) + "s";
	if (id == 1) {
		s_input = "data";
	}
	string s_num_output = boost::lexical_cast<string>(num_output);
	string s_dropout = boost::lexical_cast<string>(dropout);
	modi = boost::replace_all_copy(modi, "#id#", sid);
	modi = boost::replace_all_copy(modi, "#num_output#", s_num_output);
	modi = boost::replace_all_copy(modi, "#drop_rate#", s_dropout);
	modi = boost::replace_all_copy(modi, "#prev_top#", s_input);
	
	layers_so_far += 1 + (b_ReLU ? 1 : 0) + (b_Sigmoid ? 1 : 0) + (b_drop ? 1 : 0);
	
	return modi;
	
}

string CreateSolverParamStr( float lr)
{
	string modi =	
					"test_iter: 100\n"
					"test_interval: 10000\n" // pro forma
					"base_lr: #lr#\n"
					"lr_policy: \"step\"\n"
					"gamma: 0.9\n"
					"stepsize: 100000\n"
					"display: 2000\n" // should be ~2000
					"max_iter: 500000\n"
					"iter_size: 1\n"
					"momentum: 0.9\n"
					"weight_decay: 0.0005\n"
					"snapshot: 100000\n"
					"snapshot_prefix: \"/devlink/caffe/data/NetGen/GramPosValid/models/g\"\n"
					"snapshot_after_train: false\n"
#ifdef USE_CPU
					"solver_mode: CPU\n";
#else	
					"solver_mode: GPU\n";
#endif					

	string s_lr = boost::lexical_cast<string>(lr);
	modi = boost::replace_all_copy(modi, "#lr#", s_lr);
	return modi;
}


void NGNet::Gen(vector<int>& num_nodes_in_layer, int mod_idx_idx, float lr, CaffeGenSeed * config) {
	const float c_drop_rate = 0.2f;
	string solver_params_str =	CreateSolverParamStr(lr);
	SolverParameter solver_param;
	google::protobuf::TextFormat::ParseFromString(solver_params_str, &solver_param);
	NetParameter* net_param = solver_param.mutable_train_net_param();
	string net_def = CreateHD5TrainStr(config->train_list_file_name());
	int num_layers_so_far = 1;
	const int c_num_data_split_layers = 1;
	ip_layer_idx_arr_.clear();
	ip_layer_idx_arr_.push_back(num_layers_so_far + c_num_data_split_layers);
	for (int in = 0; in < num_nodes_in_layer.size(); in++) {
//#pragma message "don't leave relu off on both train and test"		
		net_def += AddInnerProductStr(	cb_ReLU, !cb_Sigmoid, cb_drop, in+1, 
										num_nodes_in_layer[in], num_layers_so_far, 
										c_drop_rate);
		ip_layer_idx_arr_.push_back(num_layers_so_far + c_num_data_split_layers);
	}
	net_def += AddInnerProductStr(	!cb_ReLU, cb_Sigmoid, !cb_drop, 
									num_nodes_in_layer.size() + 1, 
									config->num_output_nodes(), 
									num_layers_so_far, 0.0f);
	int loss_layer = num_layers_so_far + 2 - 1; // two layers of split in this config - 1 for zero based index
	int num_accuracy_candidates = 1;
	if (config->has_num_accuracy_candidates()) {
		num_accuracy_candidates = config->num_accuracy_candidates();
	}
	string end_str;
	switch (config->net_end_type()) {
		case CaffeGenSeed::END_VALID:
		case CaffeGenSeed::END_ONE_HOT:
			end_str = AddSoftmaxAndAccuracyStr(	num_nodes_in_layer.size() + 1, 
												num_layers_so_far, 
												num_accuracy_candidates);
			break;
		case CaffeGenSeed::END_MULTI_HOT:
			end_str = AddCrossEntropyAndEuclideanStr(	num_nodes_in_layer.size() + 1, 
														num_layers_so_far);
			break;
			
	}
	net_def += end_str;
	train_proto_str_ = net_def;
	google::protobuf::TextFormat::ParseFromString(net_def, net_param);

	net_param = solver_param.add_test_net_param();
	net_def = CreateHD5TestStr(config->test_list_file_name());
	num_layers_so_far = 1;
	for (int in = 0; in < num_nodes_in_layer.size(); in++) {
		net_def += AddInnerProductStr(	cb_ReLU, !cb_Sigmoid, !cb_drop, in+1, 
										num_nodes_in_layer[in], num_layers_so_far, 
										0.0f);
	}
	net_def += AddInnerProductStr(	!cb_ReLU, cb_Sigmoid, !cb_drop, 
									num_nodes_in_layer.size()+1, 
									config->num_output_nodes(), num_layers_so_far, 
									0.0f);
	net_def += end_str;

	test_proto_str_ = net_def;
	google::protobuf::TextFormat::ParseFromString(net_def, net_param);
	
//	shared_ptr<caffe::Solver<float> >
//		solver(caffe::GetSolver<float>(solver_param));
	solver_.reset(new NetGenSolver(solver_param));
	solver_->get_net(net_);
	solver_->set_loss_layer(loss_layer);
	ip_layer_idx_idx_ = mod_idx_idx;
	num_test_cases_ = config->num_test_cases();
	output_model_filename_ = config->model_file_name();
	output_prototxt_filename_ = config->proto_file_name();
	 
//	Net<float> * test_net = solver_->get_test_net();
//	HDF5DataLayer<float> * data_layer =  dynamic_cast<HDF5DataLayer<float>*>(test_net->layers()[0].get());
//	data_layer->hdf_blobs_[0]->shape(0)
}

void NGNet::get_first_and_last_node_sizes(int& first_layer_size, int& last_layer_size)
{
	first_layer_size = net_->top_vecs()[0][0]->shape(1);
	int last_layer_idx = ip_layer_idx_arr_.back();
	last_layer_size = net_->top_vecs()[last_layer_idx][0]->shape(1);
}

float NGNet::DoRun(bool bIntersection, double growth_factor) {
	// growth factor is the growth of the number of weights from an arbitrary 1000 weights
//#pragma message "return growth factor" 	
	// following lines undoes the whole growth_factor logic
	growth_factor = 1.0;
	solver_->reset_loss_sum();
	const double c_highway_run_time = 40.0; // this is a function of the patience requirement of the user
	const double c_intersection_run_time = 6.0;
	const double c_base_run_time = 2.0;
	double run_time = c_highway_run_time;
	if (bIntersection) {
		run_time = c_intersection_run_time;
	}
	run_time *=  growth_factor;
	run_time += c_base_run_time;
	solver_->set_run_time(run_time);
    solver_->Solve();
	//float loss = solver_->get_avg_loss();

//			{
//				// test code
//				int mod_idx = ip_layer_idx_arr_[1];
//				Layer<float>* layer = net_->layers()[mod_idx].get();
//				for (int ib=0; ib < layer->blobs().size(); ib++) {
//					Blob<float>* weights = layer->blobs()[ib].get() ;
//					const float * pw = weights->cpu_data();
//					std::cerr << "weights: ";
//					for (int iw = 0; iw < weights->count(); iw++) {
//						std::cerr << pw[iw] << ", ";
//					}
//					std::cerr << std::endl;
//				}
//			}
	return TestOnly();
	
}

float  NGNet::TestOnly() {
	float loss = 0.0f;
	Net<float> * test_net = solver_->get_test_net();
	vector<Blob<float>*> bottom_vec;
	int num_tests = num_test_cases_; 
	LOG(INFO) << "Testing " << num_tests << " records. \n";
	for (int it = 0; it < num_tests; it++) {
		float iter_loss;
		test_net->Forward(bottom_vec, &iter_loss);
		loss += iter_loss;
	}
	LOG(INFO) << "Testing done. \n";
	
	return loss / (float)num_tests;
}

void NGNet::CopyTrainWeightsToTestNet()
{
	Net<float> * test_net = solver_->get_test_net();
	test_net->ShareTrainedLayersWith(net_.get());
}


void NGNet::MakeLiveSnapshot(LiveSnapshot& net_snap) {
	
	net_snap.weights.clear();
	for (int il = 0; il < net_->layers().size(); il++) {
		net_snap.weights.push_back(vector<vector<float> >());
		vector<vector<float> >& layer_weights = net_snap.weights.back();
		Layer<float>* layer = net_->layers()[il].get();
		for (int ib=0; ib < layer->blobs().size(); ib++) {
			layer_weights.push_back(vector<float>());
			vector<float>& blob_weights = layer_weights.back();
			Blob<float>* weights = layer->blobs()[ib].get() ;
			if (weights->count() == 0) {
				continue;
			}
			const float * pw = weights->cpu_data();
			for (int iw = 0; iw < weights->count(); iw++) {
				blob_weights.push_back(pw[iw]);
			}
		}
	}
	CopyTrainWeightsToTestNet();
	LOG(INFO) << "Snapshotting to binary proto file " << output_model_filename_;
	// N.B. We are sending the test proto
	Net<float> * test_net = solver_->get_test_net();
	NetParameter net_param;
	test_net->ToProto(&net_param, false);
	WriteProtoToBinaryFile(net_param, output_model_filename_);
	std::ofstream proto_f(output_prototxt_filename_.c_str());
	if (proto_f.is_open()) { 
		proto_f << test_proto_str_; 		
	}

	
}

void NGNet::SetWeightsFromLiveSnapshot(LiveSnapshot& net_snap) {
	for (int il = 0; il < net_->layers().size(); il++) {
		Layer<float>* layer = net_->layers()[il].get();
		for (int ib=0; ib < layer->blobs().size(); ib++) {
			vector<float>& blob_weights =  net_snap.weights[il][ib];
			Blob<float>* weights = layer->blobs()[ib].get() ;
			if (weights->count() == 0) {
				continue;
			}
			float * pw = weights->mutable_cpu_data();
			for (int iw = 0; iw < weights->count(); iw++) {
				pw[iw] = blob_weights[iw];
			}
		}
	}
	
}

enum ModAction {
	ModActionDoubleLayer,
	ModActionHalfLayer,
	ModActionDoubleLR,
	ModActionHalfLR,
	NumModActions 
};
void NetGen::Init() {
	CaffeGenSeed config;
	CHECK(ReadProtoFromTextFile(config_file_name_, &config));
	config.train_list_file_name();

	int num_nodes_in_last_layer = config.num_output_nodes(); 
	vector<int> ip_layer_idx_arr;
	vector<int> num_nodes_in_layer;
	num_nodes_in_layer.push_back(1000); //core start 10 or 5
	num_nodes_in_layer.push_back(300); // core start 3  
	const float c_start_lr = 0.01; // reasonab;e start 0.01
	const float c_lr_mod_factor = 1.2f;
	float lr = c_start_lr;
	LiveSnapshot snap;
	snap.lr = lr;
	snap.num_nodes_in_layer = num_nodes_in_layer;
	shared_ptr<NGNet> ng_net(new NGNet());
	const int c_num_to_test = 3;
	ng_net->Gen(num_nodes_in_layer, -1, lr, &config);
	int first_layer_size, last_layer_size;
	ng_net->get_first_and_last_node_sizes(first_layer_size, last_layer_size);
	double c_base_num_weights = 10000.0;	// keep at 1000+. Actually a patience param but we cnn set that with the base timing
	int num_fails = 0;
	float best_loss = FLT_MAX;
	//boost::timer::cpu_timer progress_timer;
	//const int max_no_progress = 3;
	float pre_loss = ng_net->TestOnly();
	std::cerr << "Loss starting at " << pre_loss << ".\n";
	bool b_test_once = true;

	while(true) {
		int num_weights_total = (first_layer_size * num_nodes_in_layer[0]) ;
		for (int innl = 0; innl < num_nodes_in_layer.size()-1; innl++) {
			num_weights_total += (num_nodes_in_layer[innl] * num_nodes_in_layer[innl+1]);
		}
		num_weights_total += (		num_nodes_in_layer[num_nodes_in_layer.size()-1] 
								*	last_layer_size);
		std::cerr << "total num weights " << num_weights_total << "\n";
		
		double growth_factor = (double)num_weights_total
								/ c_base_num_weights;
		
		float loss_highway = ng_net->DoRun(false, growth_factor);

		if (b_test_once)
		{
			// test code
			//b_test_once = false;
		}
//		{
//			// test code
//			// code doubles a layer and then halves it to check that the result is more or less the same as at start
//			vector<int> num_nodes_in_layer_mod = num_nodes_in_layer;
//			shared_ptr<NGNet> ng_net_2;
//			ng_net_2.reset(new NGNet());
//			num_nodes_in_layer_mod[1] *= 2;
//			ng_net_2->Gen(num_nodes_in_layer_mod, 1, lr, &config);
//			ng_net_2->SetWeights(ng_net.get());
//			ng_net_2->CopyTrainWeightsToTestNet();
//			float loss_1 = ng_net->TestOnly();
//			float loss_2 = ng_net_2->TestOnly();
//			std::cerr << "loss on weights double went from " << loss_1 << " to " << loss_2 << ".\n";
//			shared_ptr<NGNet> ng_net_3;
//			ng_net_3.reset(new NGNet());
//			ng_net_3->Gen(num_nodes_in_layer, 1, lr, &config);
//			ng_net_3->SetWeights(ng_net_2.get());
//			ng_net_3->CopyTrainWeightsToTestNet();
//			float loss_3 = ng_net_2->TestOnly();
//			float loss_4 = ng_net_3->TestOnly();
//			std::cerr << "loss on weights half went from " << loss_3 << " to " << loss_3 << ".\n";
//		}
	
		shared_ptr<NGNet> ng_net_better;
		ModAction better_action = NumModActions;
		int better_action_param = 0;
		bool b_found_one_improvement = false;
		float loss_intersection_change = 0.0f;
		vector<pair<ModAction, int> > ActionOpts;
		for (int inl = 0; inl < num_nodes_in_layer.size(); inl++) {
			ActionOpts.push_back(make_pair(ModActionDoubleLayer, inl));
			ActionOpts.push_back(make_pair(ModActionHalfLayer, inl));
		}
		ActionOpts.push_back(make_pair(ModActionDoubleLR, -1));
		ActionOpts.push_back(make_pair(ModActionHalfLR, -1));
		vector<pair<int, pair<ModAction, int>  > > phases;
		for (int i_phase = 0; i_phase < ActionOpts.size(); i_phase++) {
			phases.push_back(make_pair(rand(), ActionOpts[i_phase]));
		}
		std::sort(phases.begin(), phases.end());
		int DiscourageDoublingFactor = 6; // a number from 1 to 10, rand() must beat it to play the option
		for (int ima = 0; ima < c_num_to_test; ima++) {
			vector<int> num_nodes_in_layer_mod = num_nodes_in_layer;
			shared_ptr<NGNet> ng_net_2;
			ModAction mod_action = phases[ima].second.first;
			int mod_action_param = phases[ima].second.second;
//			{
//				// test code 
//				mod_action = ModActionHalfLayer;
//				mod_action_param = 1;
//				ima = c_num_to_test - 1;
//				
//			}
			switch(mod_action) {
				case ModActionDoubleLayer:
					if ((rand() % 10) <= DiscourageDoublingFactor) continue;
					std::cerr << "trying double layer " << mod_action_param + 1 << "\n";
					ng_net_2.reset(new NGNet());
					num_nodes_in_layer_mod[mod_action_param] *= 2;
					ng_net_2->Gen(num_nodes_in_layer_mod, mod_action_param, lr, &config);
					break;
				case ModActionHalfLayer:
					if ((num_nodes_in_layer_mod[mod_action_param] % 2) == 1) {
						// don't bother if not even
						continue;
					}
					std::cerr << "trying halving layer " << mod_action_param + 1 << "\n";
					ng_net_2.reset(new NGNet());
					num_nodes_in_layer_mod[mod_action_param] /= 2;
					ng_net_2->Gen(num_nodes_in_layer_mod, mod_action_param, lr, &config);
					break;
				case ModActionDoubleLR:
					std::cerr << "trying increase learning rate\n";
					ng_net_2.reset(new NGNet());
					ng_net_2->Gen(num_nodes_in_layer, -1, lr * c_lr_mod_factor, &config);
					break;
				case ModActionHalfLR:
					//if (lr < 0.02f) continue; // experiment limiting lr halving
					std::cerr << "trying decrease learning rate\n";
					ng_net_2.reset(new NGNet());
					ng_net_2->Gen(num_nodes_in_layer, -1, lr / c_lr_mod_factor, &config);
					break;
				default:
					std::cerr << "better_action unexpected option\n";
					break;
			}
			ng_net_2->SetWeights(ng_net.get());
//			{
//				// test code
//				ng_net_2->CopyTrainWeightsToTestNet();
//				float loss_1 = ng_net->TestOnly();
//				float loss_2 = ng_net_2->TestOnly();
//				std::cerr << "loss on weights copy (with random) went from " << loss_1 << " to " << loss_2 << ".\n";
//			}
			loss_intersection_change = ng_net_2->DoRun(true, growth_factor); // there are more nodes here, so growth_factor could change but not good for comparison
			std::cerr << "change loss went from " << loss_highway << " to " << loss_intersection_change << ". \n";
			if (loss_highway > loss_intersection_change) {
				//ng_net.reset(ng_net_2.get());
				b_found_one_improvement = true;
				ng_net_better = ng_net_2;
				better_action = mod_action;	
				better_action_param = mod_action_param;
				break;
			}
		} // end pahse loop
		
		float loss_intersection_continue = ng_net->DoRun(true, growth_factor);
		float this_best_loss;
		std::cerr << "continue loss went from " << loss_highway << " to " << loss_intersection_continue << ". \n";
		if (b_found_one_improvement && (loss_intersection_change < loss_intersection_continue)) {
			this_best_loss = loss_intersection_change;
			//num_fails = 0;
			switch(better_action) {
				case ModActionDoubleLayer:
					num_nodes_in_layer[better_action_param] *= 2;
					std::cerr	<< "upgrading layer " << better_action_param + 1 
								<< " nodes to " << num_nodes_in_layer[better_action_param] << "\n";
					break;
				case ModActionHalfLayer:
					num_nodes_in_layer[better_action_param] /= 2;
					std::cerr	<< "upgrading by half layer " << better_action_param + 1 
								<< " nodes to " << num_nodes_in_layer[better_action_param] << "\n";
					break;
				case ModActionDoubleLR:
					lr *= c_lr_mod_factor;
					std::cerr << "upgrading by increasing learning rate to " << lr << "\n";
					break;
				case ModActionHalfLR:
					lr /= c_lr_mod_factor;
					std::cerr << "upgrading by decreasing learning rate to " << lr << "\n";
					break;
				case NumModActions:
				default:
					std::cerr << "better_action unexpected option\n";
					break;
			}			
			ng_net = ng_net_better;
		}
		else if (!b_found_one_improvement && (loss_intersection_continue > loss_highway) ) {
			this_best_loss = loss_highway;
//			if (num_fails >= c_max_fails) {
//				std::cerr << "Optimization complete. No better option\n";
//				break;
//			}
//			num_fails++;
//			std::cerr << "Failed to improve, num fails is now " << num_fails << std::endl;
		}
		else {
			// third, default option is to keep going with the current net
			this_best_loss = loss_intersection_continue;
		}
		if ((this_best_loss < best_loss) && ((best_loss - this_best_loss) > (best_loss / 1000.0f))) {
			best_loss = this_best_loss;
			//progress_timer.stop(); progress_timer.start();
			std::cerr << "New record for best loss set: " << best_loss << "\n";
			snap.lr = lr; 
			snap.num_nodes_in_layer = num_nodes_in_layer;
			ng_net->MakeLiveSnapshot(snap);
			num_fails = 0;
		}
//		sec seconds = boost::chrono::nanoseconds(progress_timer.elapsed().user + progress_timer.elapsed().system);
//		if (seconds.count() > max_wait_for_progress) {
//			std::cerr	<< "Optimization complete. record for lowest score not set for " 
//						<< seconds.count() << " seconds.\n";
//			break;
//		}
		num_fails++;
		if ((num_fails % 3) == 0) {
			std::cerr << "We have now had " << num_fails << " attempts without improvement\n";
		}
		if ((num_fails % 9) == 0) {
			lr = c_start_lr;
			std::cerr << "Additional step. Resetting learning rate to " << lr << "\n";
		}
		if ((num_fails % 15) == 0) {
			lr = c_start_lr;
			std::cerr << "Additional step to get out of mess. Going back to snapshot of last record low\n";
			lr = snap.lr; 
			num_nodes_in_layer = snap.num_nodes_in_layer;
			ng_net.reset(new NGNet());
			ng_net->Gen(num_nodes_in_layer, -1, lr, &config);
			ng_net->SetWeightsFromLiveSnapshot(snap);
			ng_net->CopyTrainWeightsToTestNet();
		}
		if (num_fails  == 21) {
			float loss_1 = ng_net->TestOnly();
			ng_net->ShakeupWeights();
			float loss_2 = ng_net->TestOnly();
			std::cerr << "Additional step. Random Shakeup. Loss on shakeup went from " << loss_1 << " to " << loss_2 << ".\n";
		}
		if (num_fails  == 30) {
			// carefull with ifs and elses here. You don't want to create an impossibe if
			// Make sure that there is a reset to best live snapshot on a lower num_fails value
			// than the one used to select this option. 
			std::cerr << "We seem to be at a stable minimum, attempting to add an extra layer to the net\n";
			// before adding layer, make the starting point a snapshot of the best low so far
			// floowing code commented out on assumption that new layer is a multiple of return to snapshot
//			lr = snap.lr; 
//			num_nodes_in_layer = snap.num_nodes_in_layer;
//			ng_net.reset(new NGNet());
//			ng_net->Gen(num_nodes_in_layer, -1, lr, &config);
//			ng_net->SetWeightsFromLiveSnapshot(snap);
//			ng_net->CopyTrainWeightsToTestNet();
			vector<int> num_nodes_in_layer_mod = num_nodes_in_layer;
			// you can duplicate the last layer, but num_nodes_in_layer only refers to the middle layers, so +1
			int i_layer_duplicate = rand() % (num_nodes_in_layer_mod.size() + 1); 
			std::cerr << "duplicating layer " << i_layer_duplicate << "\n";
			if (i_layer_duplicate < num_nodes_in_layer_mod.size()) {
				vector<int>::iterator itn = num_nodes_in_layer_mod.begin() + i_layer_duplicate;
				num_nodes_in_layer_mod.insert(itn, num_nodes_in_layer_mod[i_layer_duplicate]);
			}
			else {
				num_nodes_in_layer_mod.push_back(num_nodes_in_last_layer);
			}
			shared_ptr<NGNet> ng_net_2;
			ng_net_2.reset(new NGNet());
			ng_net_2->Gen(num_nodes_in_layer_mod, -1, lr, &config);
			ng_net_2->CopyWeightsAndAddDupLayer(ng_net.get(), i_layer_duplicate+1); // index of *duplicated* layer
			{
				// test code
				ng_net_2->CopyTrainWeightsToTestNet();
				float loss_1 = ng_net->TestOnly();
				float loss_2 = ng_net_2->TestOnly();
				std::cerr << "loss on duplicating layer went from " << loss_1 << " to " << loss_2 << ".\n";
			}
			ng_net = ng_net_2;
			num_nodes_in_layer = num_nodes_in_layer_mod;
		}
		if (num_fails >= max_no_progress) {
			std::cerr << "Optimization complete. No better option\n";
			break;
		}
	}
	
	
	bInit_ = true;

}


/* Return the values in the output layer */
bool NetGen::Classify() {
	CHECK(bInit_) << "NetGen: Init must be called first\n";
	
	

	return true;
}
 

/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

void CallNetGen(const string& proto_fname, int num_tries)
{
	//NetGen generator("/devlink/caffe/data/NetGen/gengen1455787125/data/config.prototxt");
	// second arg, how many tries without progress
	NetGen generator(proto_fname, num_tries);
	vector<shared_ptr<NGNet> > nets;
	generator.PreInit();
	generator.Init();
	
}
 
#ifdef CAFFE_NET_GEN_MAIN
int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0]
				  << " <gengen prototxt filename>" << std::endl;
		return 1;
	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
	
	//NetGen generator("/devlink/caffe/data/NetGen/gengen1455787125/data/config.prototxt");
	// second arg, how many tries without progress
	NetGen generator(argv[1], 3);
	vector<shared_ptr<NGNet> > nets;
	generator.PreInit();
	generator.Init();

	
}
#endif // CAFFE_MULTINET_MAIN
    