#include <cstdlib>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include "H5Cpp.h"

#include "caffe/proto/GenData.pb.h"
#include "caffe/GenData.hpp"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
namespace fs = boost::filesystem;


typedef unsigned long long u64;

struct SModelData {
	CGenDef* NetGenData;
	string model_file_name_;
	string trained_file_name_;
	string input_layer_name_ ;
	string output_layer_name_;
};

class SingleNet {
public:
	SingleNet(	const string& model_file,
				const string& trained_file,
				const string& input_layer_name,
				const string& output_layer_name) {
		model_file_ = model_file;
		trained_file_ = trained_file;
		input_layer_name_ = input_layer_name;
		output_layer_name_ = output_layer_name;
	}
	
	void Init();
	Blob<float>* GetVec(bool b_top, int layer_idx, int branch_idx);
	Blob<float>* GetInputVec() {
		return GetVec(true, input_layer_idx_, input_layer_top_idx_);
	}
	Blob<float>* GetOutputVec() {
		return GetVec(true, output_layer_idx_, output_layer_top_idx_);
	}
	void PrepForInput() {
		net_->ForwardFromTo(0, input_layer_idx_);
	}
	float ComputeOutput() {
		return net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);
	}
	int input_layer_dim() { return input_layer_dim_; }
	int input_layer_idx() { return input_layer_idx_; }
	int output_layer_dim() { return output_layer_dim_; }
	
private:
	shared_ptr<Net<float> > net_;
	int input_layer_idx_;
	int input_layer_top_idx_; // currently the index of the array of top_vectors for this net
	int output_layer_idx_;
	int output_layer_top_idx_; // currently the index of the array of top_vectors for this net
	string model_file_;
	string trained_file_;
	string input_layer_name_ ;
	string output_layer_name_;
	int input_layer_dim_;
	int output_layer_dim_;
};

class MultiNet {
public:
	MultiNet(string a_data_core_dir) {
		data_core_dir_ = a_data_core_dir;
		bInit_ = false; 
	}

	void PreInit();
	void Init(	vector<SingleNet>& nets,
				const string& word_file_name,
				const string& word_vector_file_name);

	bool  Classify();
	bool  PredictTheOne(int inet);

	vector<SModelData> model_data_arr_;

private:

	std::vector<pair<float, int> > Predict();


	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	vector<SingleNet>* p_nets_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	vector<string> words_;
	vector<vector<float> > words_vecs_;
	string word_vector_file_name_;
	int words_per_input_;
	CGenDefTbls * tbls_data_;
	string data_core_dir_;
	
	int GetClosestWordIndex(vector<float>& VecOfWord, int num_input_vals, 
							vector<pair<float, int> >& SortedBest,
							int NumBestKept, vector<vector<float> >& lookup_vecs,
							int RandFactor = 100);

};

class CGenGen {
public:
    CGenGen(vector<SSentenceRec>& aSentenceRec,
			vector<CorefRec>& aCorefList,
			vector<SSentenceRecAvail>& aSentenceAvailList,
			vector<DataAvailType>& aCorefAvail,
			vector<string>& aDepNames) :
		SentenceRec(aSentenceRec), CorefList(aCorefList),
        SentenceAvailList(aSentenceAvailList), CorefAvail(aCorefAvail),
		DepNames(aDepNames) {
    }
    
	void DoGen(string& data_core_dir);
    
private:

    vector<SSentenceRec>& SentenceRec; 
    vector<CorefRec>& CorefList;
    vector<SSentenceRecAvail>& SentenceAvailList; 
    vector<DataAvailType>& CorefAvail;  
	vector<string>& DepNames;
};

template <typename T>
string gen_multinet_to_string ( T Number )
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}


void CGenGen::DoGen(string& data_core_dir) {
	string gen_name = string("gengen") + gen_multinet_to_string(time(NULL));
	CaffeGenData gen_data;
	gen_data.set_name(gen_name);
	gen_data.set_iterate_type(CaffeGenData::ITERATE_DEP);
	gen_data.set_data_src(CaffeGenData::DATA_SRC_BOOKS);
	gen_data.set_files_core_dir(data_core_dir + string("NetGen/") + gen_name + "/");
	gen_data.set_test_list_file_name("data/test");
	gen_data.set_train_list_file_name("data/train");
	gen_data.set_net_end_type(CaffeGenData::END_ONE_HOT);
	gen_data.set_proto_file_name("models/train.prototxt");
	gen_data.set_model_file_name("models/g_best.caffemodel");
	gen_data.set_config_file_name("data/config.prototxt");
	gen_data.set_netgen_output_file_name("models/netgen_output.prototxt");
	gen_data.set_num_accuracy_candidates(1);
//	gen_data.set_vec_tbls_core_path("/devlink/caffe/data/tables/");
//	CaffeGenData::VecTbl * vec_tbl = gen_data.add_vec_tbls();
//	vec_tbl->set_name("DepVecTbl");
//	vec_tbl->set_path("DepOneHot");
//	vec_tbl = gen_data.add_vec_tbls();
//	vec_tbl->set_name("POSVecTbl");
//	vec_tbl->set_path("POSOneHot");
//	vec_tbl = gen_data.add_vec_tbls();
//	vec_tbl->set_name("POSNumTbl");
//	vec_tbl->set_path("POSNum");
//	gen_data.set_dep_name_vec_tbl("DepVecTbl");
	int iVarName = 0;
	vector<pair<string, string> > InputFields;
	for (int irec = 0; irec < SentenceRec.size(); irec++) {
		vector<DepRec>& deps = SentenceRec[irec].Deps;
		for (int iidep = 0; iidep < deps.size(); iidep++) {
			string sVarName = string("DepType") 
					+ gen_multinet_to_string(iVarName++);
			CaffeGenData::DataField * data_field = gen_data.add_data_fields();
			data_field->set_var_name(sVarName);
			data_field->set_field_type(CaffeGenData::FIELD_TYPE_DEP_NAME);
			if (SentenceAvailList[irec].Deps[iidep].iDep == datConst) {
				int iDepType = (int)(signed char)(deps[iidep].iDep);
				InputFields.push_back(make_pair(sVarName, "DepVecTbl")); // because we are iterating dep recs
				if ((iDepType < 0) || (iDepType >= DepNames.size())) {
					continue;
				}
				string& DepTypeName = DepNames[iDepType];
				CaffeGenData::DataFilter * data_filter = gen_data.add_data_filters();
				data_filter->set_var_name(sVarName);
				data_filter->set_match_string(DepTypeName);
			}
			else if (SentenceAvailList[irec].Deps[iidep].iDep == datTheOne) {
				CaffeGenData::FieldTranslate * field_translate 
						= gen_data.add_output_field_translates();
				field_translate->set_var_name(sVarName);
				field_translate->set_table_name("DepNum"); // because the putput is a dep name but its one hot so we use a num tbl
			}
			else {
				continue;
			}
			if (	(deps[iidep].Dep != -1) 
				&&	(SentenceAvailList[irec].Deps[iidep].Dep == datConst)) {
				
				sVarName = string("Dep") 
						+ gen_multinet_to_string(iVarName++);
				data_field = gen_data.add_data_fields();
				data_field->set_var_name(sVarName);
				data_field->set_field_type(CaffeGenData::FIELD_TYPE_DEP_RWID);
				int WID = (int)(signed char)deps[iidep].Dep;
				
				CaffeGenData::DataTranslate * data_translate = gen_data.add_data_translates();
				data_translate->set_translate_type(CaffeGenData::DATA_TRANSLATE_RWID_TO_WORD);
				data_translate->set_match_name(sVarName);
				sVarName = string("POS") 
					+ gen_multinet_to_string(iVarName++);
				data_translate->set_var_name(sVarName);
				data_translate->set_field_type(CaffeGenData::FIELD_TYPE_POS);
				SentenceRec[irec].OneWordRec[WID].POS;
				DataAvailType dat = SentenceAvailList[irec].WordRecs[WID].POS;
				if (dat == datConst) {
					
				}
				else if (dat == datTheOne) {
					CaffeGenData::FieldTranslate * field_translate 
							= gen_data.add_output_field_translates();
					// sVarNum should still hold the var name of the data translate on this
					// branch of the if
					field_translate->set_var_name(sVarName);
					field_translate->set_table_name("POSNumTbl"); // integer output tables for POS, which is what we're looking for
				}
			}

		}
	}
	
	for (int iin = 0; iin < InputFields.size(); iin++) {
		CaffeGenData::FieldTranslate * field_translate 
				= gen_data.add_input_field_translates();
		field_translate->set_var_name(InputFields[iin].first);
		field_translate->set_table_name(InputFields[iin].second); // integer output tables for POS, which is what we're looking for
	}
	const string ProtoCoreDir = data_core_dir + "GenGen/";
	const string fname = ProtoCoreDir + gen_name + ".prototxt";
	ofstream gengen_ofs(fname.c_str());
	google::protobuf::io::OstreamOutputStream* gengen_output 
		= new google::protobuf::io::OstreamOutputStream(&gengen_ofs);
	//ofstream f_config(ConfigFileName); // I think this one is wrong
	if (gengen_ofs.is_open()) {
		google::protobuf::TextFormat::Print(gen_data, gengen_output);
		
	}
	delete gengen_output;

	const string NetGenCoreDir = data_core_dir + "NetGen/";
	const string dname = NetGenCoreDir + gen_name;
	fs::path dir(dname);
	if (!fs::create_directory(fs::path (dname))) {
		cerr << "DoGen Error: Failed to create directory for NetGen! \n";
		return;
	}
	const string dname_models = dname + "/models";
	if (!fs::create_directory(fs::path (dname_models))) {
		cerr << "DoGen Error: Failed to create directory for NetGen! \n";
		return;
	}
	const string dname_data = dname + "/data";
	if (!fs::create_directory(fs::path (dname_data))) {
		cerr << "DoGen Error: Failed to create directory for NetGen! \n";
		return;
	}
	
}


void SingleNet::Init(	) {


	input_layer_top_idx_ = 0;
	output_layer_top_idx_ = 0;
	
	/* Load the network. */
	net_.reset(new Net<float>(model_file_, TEST));
	net_->CopyTrainedLayersFrom(trained_file_);

	
	int input_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == input_layer_name_) {
			input_layer_idx = layer_id;
			break;
		}
	}
	if (input_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << input_layer_name_;			
	}

	input_layer_idx_ = input_layer_idx;
	
	input_layer_top_idx_ = 0;

	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_top_idx_];
	input_layer_dim_ = input_layer->shape(1);

	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name_) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name_;			
	}
	output_layer_idx_ = output_layer_idx;
	Blob<float>* output_layer = net_->top_vecs()[output_layer_idx_][output_layer_top_idx_];
	output_layer_dim_ = output_layer->shape(1);
	
	
}

Blob<float>* SingleNet::GetVec(bool b_top, int layer_idx, int branch_idx)
{
	if (b_top) {
		return net_->top_vecs()[layer_idx][branch_idx];
	}
	else {
		return net_->bottom_vecs()[layer_idx][branch_idx];
	}
}

void MultiNet::PreInit()
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	
	string sModelDir = data_core_dir_ + "GenGen";
	string sTblModelsFile = "/TblDefs/tbls.prototxt";
	const bool cb_model_owns_tbls = true;
	
	string fname = sModelDir + sTblModelsFile;
	tbls_data_ = new CGenDefTbls (fname);
	if (!tbls_data_->bInitDone) {
		cerr << "Failed to initialize tables def file. Terminating.\n";
		return;
	}
	
	fs::directory_iterator end_iter;
	
	if ( fs::exists(sModelDir) && fs::is_directory(sModelDir)) {
		for( fs::directory_iterator dir_iter(sModelDir) ; dir_iter != end_iter ; ++dir_iter)	{
			if (fs::is_regular_file(dir_iter->status()) ) {
				if (dir_iter->path().extension() == ".prototxt") {
					std::cerr << dir_iter->path() << std::endl;
					model_data_arr_.push_back(SModelData());
					SModelData& model_data = model_data_arr_.back();
					
					CGenDef * init_data = new CGenDef(tbls_data_, !cb_model_owns_tbls);

					if (	!init_data->ModelInit(dir_iter->path().c_str()) 
						||	!init_data->ModelPrep()) {
						delete init_data;
						return;
					}
					model_data.NetGenData = init_data;
					// extract these from the actual model!!!
					model_data.input_layer_name_ = "data";
					model_data.output_layer_name_ = "squash3";
					model_data.model_file_name_ = init_data->getGenData()->files_core_dir() + init_data->getGenData()->proto_file_name();
					model_data.trained_file_name_ = init_data->getGenData()->files_core_dir() + init_data->getGenData()->model_file_name();

				}
				
			}
		}
		
//    {
//      result_set.insert(result_set_t::value_type(fs::last_write_time(dir_iter->path()), *dir_iter));
//    }
//  }
	}
}

void MultiNet::Init(	vector<SingleNet>& nets,
						const string& word_file_name,
						const string& word_vector_file_name) {

	word_vector_file_name_ = word_vector_file_name;
	//output_layer_idx_arr_ = vector<int>(5, -1);

	words_per_input_ = 1;
	words_per_input_ = 4;
	
	vector<CorefRec> CorefSoFar;
	vector<SSentenceRec> SentenceList(1);
	vector<DataAvailType> CorefAvail;
	vector<SSentenceRecAvail> SentenceAvailList(1);
	SSentenceRec& SRec = SentenceList[0];
	SRec.OneWordRec.push_back(WordRec());
	SRec.Deps.push_back(DepRec());
	DepRec& InitDep = SRec.Deps.back();
	InitDep.iDep = 0; // root
	InitDep.Gov = (uchar)-1;
	InitDep.Dep = 0; // first word rec
	WordRec& InitWRec = SRec.OneWordRec.back();
	InitWRec.POS = "";

	SSentenceRecAvail& SRecAvail = SentenceAvailList[0];
	SRecAvail.WordRecs.push_back(SWordRecAvail(datNotSet));
	SRecAvail.Deps.push_back(SDepRecAvail());
	SDepRecAvail& DepAvail = SRecAvail.Deps.back();
	DepAvail.iDep = datConst;
	DepAvail.Gov = datNotSet;
	DepAvail.Dep = datConst;
	SWordRecAvail& WRecAvail = SRecAvail.WordRecs.back();
	WRecAvail.POS = datTheOne;
	
	
	p_nets_ = &nets;

	CGenGen GenGen(	SentenceList, CorefSoFar, SentenceAvailList, CorefAvail, 
					tbls_data_->getDepNamesTbl());
	GenGen.DoGen(data_core_dir_);
	
	for (int in = 0; in < nets.size(); in++) {
		SModelData& model_data = model_data_arr_[in];

		
		vector<pair<int, int> >& InputTranslateTbl = model_data.NetGenData->getInputTranslateTbl();
		vector<pair<int, int> > OutputTranslateTbl = model_data.NetGenData->getOutputTranslateTbl();
		//int NumOutputNodesNeeded = -1;

		CGenModelRun GenModelRun(	*model_data.NetGenData, SentenceList, CorefSoFar, 
									SentenceAvailList, CorefAvail);
		
		GenModelRun.setReqTheOneOutput();

		if (!GenModelRun.DoRun()) {
			return;
		}

		vector<SDataForVecs >& DataForVecs = GenModelRun.getDataForVecs();
		
		if (DataForVecs.size() == 0) {
			continue;
		}
		
		if (DataForVecs.size() != 1) {
			cerr << "Not clear how there could be more than one records for the one. Please investigate\n";
			continue;
		}

		SingleNet& net = nets[in];
		net.Init();
		
		vector<pair<string, vector<float> > > VecArr;
		vector<vector<vector<float> >* >& VecTblPtrs = model_data.NetGenData->getVecTblPtrs();
		vector<map<string, int>*>& TranslateTblPtrs = model_data.NetGenData->getTranslateTblPtrs();

		Blob<float>* predict_input_layer =  net.GetInputVec();
		Blob<float>* predict_output_layer = net.GetOutputVec();

		net.PrepForInput(); // I think this loads the hdl5 data which can't be prevented
		float* p_in = predict_input_layer->mutable_cpu_data();
		float * ppd = p_in;
		
		vector<int>& IData = (DataForVecs[0].IData);
				
		for (int ii = 0; ii < InputTranslateTbl.size(); ii++) {
			pair<int, int>& itt = InputTranslateTbl[ii];
			vector<float>& vec = (*VecTblPtrs[itt.second])[IData[ii]];
			for (int iv = 0; iv < vec.size(); iv++) {
				*ppd++ = vec[iv];
			}			
		}
		

		/*float v_loss = */net.ComputeOutput();
		const float* p_v_out = predict_output_layer->cpu_data();  
		vector<float> VecOfOutput;
		int NumValsInOutput = net.output_layer_dim();
		for (int iact = 0; iact < NumValsInOutput; iact++) {
			VecOfOutput.push_back(*p_v_out++);
		}
		vector<pair<float, int> > SortedBest;
		vector<pair<float, int> > SortedBestDummy;
		//vector<int>& OData = (DataForVecs[0].OData);
		pair<int, int>& ott = OutputTranslateTbl[0]; // there may only be one output
		vector<vector<float> > vec_for_one_hot;
		vector<vector<float> >* p_lookup_vec = VecTblPtrs[ott.second];
		if (model_data.NetGenData->getGenData()->net_end_type() == CaffeGenData::END_ONE_HOT) {
			vector<vector<float> >& one_hot_tbl = *(VecTblPtrs[ott.second]);
			for (int ioht = 0; ioht < one_hot_tbl.size(); ioht++) {
				vec_for_one_hot.push_back(vector<float>(NumValsInOutput, 0.0f));
				vector<float>& vec = vec_for_one_hot.back();
				vec[one_hot_tbl[ioht][0]] = 1.0f;
			}
			p_lookup_vec = &vec_for_one_hot;
		}
		srand((int)time(NULL)); for (int ir=0; ir<10; ir++) rand();
		int iMinDiffLbl = GetClosestWordIndex(VecOfOutput, NumValsInOutput,
				SortedBestDummy, 1, *(p_lookup_vec), 100);
		map<string, int>& SymTbl = *(TranslateTblPtrs[ott.second]);
		map<string, int>::iterator itSymTbl = SymTbl.begin();
		for (; itSymTbl != SymTbl.end(); itSymTbl++) {
			if (itSymTbl->second == iMinDiffLbl) {
				cerr << "New value: " << itSymTbl->first;
				break;
			}
		}
		
	}

#if 0	
	
	std::ifstream str_words(word_file_name.c_str(), std::ifstream::in);
	if (str_words.is_open() ) {
		string ln;
		//for (int ic = 0; ic < cVocabLimit; ic++) {
		while (str_words.good()) {
			string w;
			getline(str_words, ln, ' ');
			//VecFile >> w;
			w = ln;
			if (w.size() == 0) {
				break;
			}
			words_.push_back(w);
			words_vecs_.push_back(vector<float>());
			vector<float>& curr_vec = words_vecs_.back();
			int num_input_vals = nets[0].input_layer_dim() / words_per_input_;
			for (int iwv = 0; iwv < num_input_vals; iwv++) {
				if (iwv == num_input_vals - 1) {
					getline(str_words, ln);
				}
				else {
					getline(str_words, ln, ' ');
				}
				float wv;
				//wv = stof(ln);
				wv = (float)atof(ln.c_str());
				curr_vec.push_back(wv);
			}

		}
	}
	//Blob<float>*  input_bottom_vec = net_->top_vecs()[input_layer_idx][input_layer_bottom_idx_];
#endif	

	bInit_ = true;

}

#if 0

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
#endif // 0

int  MultiNet::GetClosestWordIndex(	vector<float>& VecOfWord, int num_input_vals, 
									vector<pair<float, int> >& SortedBest, int NumBestKept, 
									vector<vector<float> >& lookup_vecs, int rand_factor)
{
	float MinDiff = num_input_vals * 2.0f;
	int iMinDiff = -1;
	float ThreshDiff = MinDiff;
	for (int iwv =0; iwv < lookup_vecs.size(); iwv++ ) {
		float SumDiff = 0.0f;
		for (int iv = 0; iv < num_input_vals; iv++) {
			float Diff = VecOfWord[iv] - lookup_vecs[iwv][iv];
			SumDiff += Diff * Diff;
		}
		SumDiff *= (1.0f + ((float)(rand() % rand_factor)/(float)rand_factor));
		if (SumDiff < MinDiff) {
			MinDiff = SumDiff;
			iMinDiff = iwv;
		}
		if (SumDiff < ThreshDiff) {
			SortedBest.push_back(make_pair(SumDiff, iwv));
			std::sort(SortedBest.begin(), SortedBest.end());
			if (SortedBest.size() > NumBestKept) {
				SortedBest.pop_back();
				ThreshDiff = SortedBest.back().first;
			}
		}
	}
	return iMinDiff;
}

bool MultiNet::PredictTheOne(int inet) {
	// inet - which net we will use to predict. Maybe should be in SingleNet
	
	return true;
}

/* Return the values in the output layer */
bool MultiNet::Classify() {
	CHECK(bInit_) << "MultiNet: Init must be called first\n";
	
	vector<pair<string, vector<float> > > VecArr;
	int num_vals_per_word = (*p_nets_)[0].input_layer_dim() / words_per_input_;
	Blob<float>* predict_input_layer = (*p_nets_)[0].GetInputVec();
	Blob<float>* predict_label_layer = (*p_nets_)[0].GetVec(
		true, (*p_nets_)[0].input_layer_idx(), 1);
	Blob<float>* valid_input_layer = (*p_nets_)[1].GetInputVec();
	Blob<float>* predict_output_layer = (*p_nets_)[0].GetOutputVec();
	Blob<float>* valid_output_layer = (*p_nets_)[1].GetOutputVec();
	
	int CountMatch = 0;
	int NumTestRecs = 5000;
	for (int ir = 0; ir < NumTestRecs; ir++) {
		(*p_nets_)[0].PrepForInput();
		const float* p_in = predict_input_layer->cpu_data();  // ->cpu_data();
		const float* p_lbl = predict_label_layer->cpu_data();
		//net_->ForwardFromTo(0, input_layer_idx_);

		//string w = words_[isym];
		//std::cerr << w << ",";
		const int cNumInputWords = 4;
		const int cNumValsPerWord = 100;
		int iMinDiffLbl = -1;
		vector<pair<float, int> > SortedBest;
		vector<pair<float, int> > SortedBestDummy;
		vector<int> ngram_indices(5, -1);
		int cNumBestKept = 10;
		for (int iw = 0; iw < cNumInputWords; iw++) {
			vector<float> VecOfWord;
			vector<float> VecOfLabel;
			for (int iact = 0; iact < cNumValsPerWord; iact++) {
				//*p_in++ = words_vecs_[isym][iact];
				VecOfWord.push_back(*p_in++);
			}
			if (iw == 1) {				
				for (int iact = 0; iact < cNumValsPerWord; iact++) {
					//*p_in++ = words_vecs_[isym][iact];
					VecOfLabel.push_back(*p_lbl++);
				}
				iMinDiffLbl = GetClosestWordIndex(VecOfLabel, num_vals_per_word,
													SortedBestDummy, 1, words_vecs_);
			}
			int iMinDiff = GetClosestWordIndex(VecOfWord, num_vals_per_word, 
												SortedBestDummy, 1, words_vecs_);
			if (iMinDiff != -1) {
				int word_idx = ((iw <= 1) ? iw : iw+1);
				ngram_indices[word_idx] = iMinDiff;
				string w = words_[iMinDiff];
				std::cerr << w << " ";
				if (iw == 1) {
					if (iMinDiffLbl == -1) {
						std::cerr << "XXX ";
					}
					else {
						string l = words_[iMinDiffLbl];
						std::cerr << "(" << l << ") ";
					}
				}
			}
		}
		std::cerr << std::endl;

		/*float loss = */(*p_nets_)[0].ComputeOutput();
		//float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);

		const float* p_out = predict_output_layer->cpu_data();  
		vector<float> output;
		vector<float> VecOfWord;
		for (int io = 0; io < predict_output_layer->shape(1); io++) {
			float data = *p_out++;
			VecOfWord.push_back(data);
		}
		int iMinDiff = GetClosestWordIndex(	VecOfWord, num_vals_per_word,
											SortedBest, cNumBestKept, words_vecs_);
		if (iMinDiff != -1) {
			std::cerr << "--> ";
			vector <pair<float, int> > ReOrdered;
			for (int ib = 0; ib < SortedBest.size(); ib++) {
				ngram_indices[2] = SortedBest[ib].second;
				(*p_nets_)[1].PrepForInput();
				float* p_v_in = valid_input_layer->mutable_cpu_data();  // ->cpu_data();
				const int cNumValidInputWords = 5;
				for (int iw = 0; iw < cNumValidInputWords; iw++) {
					for (int iact = 0; iact < cNumValsPerWord; iact++) {
						*p_v_in++ = words_vecs_[ngram_indices[iw]][iact];
					}
				}
				/*float v_loss = */(*p_nets_)[1].ComputeOutput();
				const float* p_v_out = valid_output_layer->cpu_data();  
				float v_val = p_v_out[1]; // seems p_v_out[0] is 1 - p_v_out[1]
				
				string w = words_[SortedBest[ib].second];
				ReOrdered.push_back(make_pair(SortedBest[ib].first / (v_val * v_val * v_val), SortedBest[ib].second));
				std::cerr << w << " (" << SortedBest[ib].first << " vs. " << v_val << "), ";
			}
			std::cerr << std::endl << "Reordered: ";
			std::sort(ReOrdered.begin(), ReOrdered.end());
			for (int iro = 0; iro < ReOrdered.size(); iro++) {
				std::cerr <<  words_[ReOrdered[iro].second] << " (" << ReOrdered[iro].first << "), ";
			}
			
			if (iMinDiffLbl == ReOrdered.front().second) {
				CountMatch++;
			}
		}
		std::cerr << std::endl;
		//VecArr.push_back(make_pair(w, output));
	}
	
	std::cerr << CountMatch << " records hit exactly out of " << NumTestRecs << "\n";
	
	std::ofstream str_vecs(word_vector_file_name_.c_str());
	if (str_vecs.is_open()) { 
		//str_vecs << VecArr[0].second.size() << " ";
		for (int iv = 0; iv < VecArr.size(); iv++) {
			pair<string, vector<float> >& rec = VecArr[iv];
			str_vecs << rec.first << " ";
			vector<float>& vals = rec.second;
			for (int ir = 0; ir < vals.size(); ir++) {
				str_vecs << vals[ir];
				if (ir == vals.size() - 1) {
					str_vecs << std::endl;
				}
				else {
					str_vecs << " ";
				}
			}
		}
	}



	return true;
}

//std::vector<pair<float, int> > MultiNet::Predict() {
//
//
//
//
//	return output;
//}

/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_MULTINET_MAIN
int main(int argc, char** argv) {
//	if (argc != 3) {
//		std::cerr << "Usage: " << argv[0]
//				  << " deploy.prototxt network.caffemodel" << std::endl;
//		return 1;
//	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
//	string model_file   = "/home/abba/caffe-recurrent/toys/MultiNet/VecPredict/train.prototxt";
//	string trained_file = "/devlink/caffe/data/MultiNet/VecPredict/models/v_iter_500000.caffemodel";
	string word_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordList.txt";
	string word_vector_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordVectors.txt";
//	string model_file   = "/home/abba/caffe-recurrent/toys/LSTMTrain/WordToPos/train.prototxt";
//	string trained_file = "/devlink/caffe/data/LSTMTrain/WordToPos/models/a_iter_1000000.caffemodel";
//	string word_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordList.txt";
//	string word_vector_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordVectors.txt";
	string input_layer_name = "data";
	string output_layer_name = "SquashOutput";
	
	MultiNet classifier("/devlink/caffe/data/");
	classifier.PreInit();
	vector<SingleNet> nets;
	for (int isn=0; isn < classifier.model_data_arr_.size(); isn++) {
		SModelData& model_data = classifier.model_data_arr_[isn];
		nets.push_back(SingleNet(	model_data.model_file_name_, model_data.trained_file_name_, 
									model_data.input_layer_name_, model_data.output_layer_name_));
	}
#if 0
	nets.push_back(SingleNet(
		"/home/abba/caffe-recurrent/toys/WordEmbed/VecPredict/train.prototxt",
		"/devlink/caffe/data/WordEmbed/VecPredict/models/v_iter_83869.caffemodel",
		"data", "SquashOutput"));
	nets.push_back(SingleNet(
		"/devlink/caffe/data/NetGen/GramPosValid/models/train.prototxt",
		"/devlink/caffe/data/NetGen/GramPosValid/models/g_best.caffemodel",
		"data", "squash3"));
#endif // 0	
	classifier.Init(nets,
					word_file_name, 
					word_vector_file_name);
	//classifier.Classify();
	
}
#endif // CAFFE_MULTINET_MAIN
  