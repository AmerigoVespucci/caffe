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
#include <boost/asio.hpp>
#include "H5Cpp.h"

#include "caffe/proto/GenData.pb.h"
#include "caffe/proto/GenDef.pb.h"
#include "caffe/GenData.hpp"

#include "caffe/proto/ipc.pb.h"
#include "caffe/util/ipc.hpp"
#include "caffe/proto/GenDef.pb.h"

void CallNetGen(const string& proto_fname, int num_tries); // in place of using a global header file for the different caffe additions

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
namespace fs = boost::filesystem;
const char WORD_VEC_TBL_NAME[] = "Words6000";				

const int c_prep_gen_port_num = 1544;
const char * c_host_name = "0.0.0.0";



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

	bool PreInit();
	bool ComposeSentence(	vector<SingleNet>& nets,
				const string& word_file_name,
				const string& word_vector_file_name,
				bool& b_fatal);

	bool  Classify();
	bool  PredictTheOne(int inet);

	vector<SModelData> model_data_arr_;

private:

	std::vector<pair<float, int> > Predict();


	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);
	void AddModel(string model_file_name);

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
	tcp::socket* ClientSocket;
	

	
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
			vector<string>& aDepNames,
			tcp::socket* aSocket) :
		SentenceRec(aSentenceRec), CorefList(aCorefList),
        SentenceAvailList(aSentenceAvailList), CorefAvail(aCorefAvail),
		DepNames(aDepNames) {
	ClientSocket = aSocket;
	
    }
    
	bool DoGen(string& data_core_dir, bool& b_fatal);
	string& getNewModelFileName() { return new_model_file_name; }
	bool PrepGen(const string& proto_name, const string& config_name, bool& b_fatal);
    
private:

    vector<SSentenceRec>& SentenceRec; 
    vector<CorefRec>& CorefList;
    vector<SSentenceRecAvail>& SentenceAvailList; 
    vector<DataAvailType>& CorefAvail;  
	vector<string>& DepNames;
	string new_model_file_name;
	tcp::socket* ClientSocket;
};

template <typename T>
string gen_multinet_to_string ( T Number )
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}

void CreateNetValue(CaffeGenDef& gen_data, int& i_var_name, 
					CaffeGenDef::VarExtractType vet, const string& src_var_name, 
					bool b_input, const char * table_name) {
	if (b_input) {
		// adding a real net value input, so clean out dummy
		// (real may actually be THE addition of the dummy but whatever)
		int inv_dummy = -1;
		for (int inv = 0; inv < gen_data.net_values_size(); inv++) {
			if (gen_data.net_values(inv).vet() == CaffeGenDef::vetDummy) {
				inv_dummy = inv;
				break;
			}
		}
		if (inv_dummy != -1) {
			::google::protobuf::RepeatedPtrField< CaffeGenDef::NetValue >* net_values = gen_data.mutable_net_values();
			int i_last = gen_data.net_values_size() - 1;
			if (inv_dummy != i_last) {
				net_values->SwapElements(inv_dummy, i_last);
			}
			net_values->RemoveLast();
		}
	}
	CaffeGenDef::NetValue * net_value = gen_data.add_net_values();
	string s_net_value_var_name = string("NetValue") 
			+ gen_multinet_to_string(i_var_name++);
	net_value->set_var_name(s_net_value_var_name);
	net_value->set_var_name_src(src_var_name);
	net_value->set_vet(vet);
	net_value->set_b_input(b_input);
	net_value->set_vec_table_name(table_name);
	
}

bool CGenGen::PrepGen(const string& proto_name, const string& config_name, bool& b_fatal) {
	CaffeIpc MsgPrep;
	MsgPrep.set_type(CaffeIpc::PREP_GEN);
	CaffeIpc::PrepGenParam* prep_gen_param = MsgPrep.mutable_prep_gen_param();
	prep_gen_param->set_gengen_filename(proto_name);
	b_fatal = false;

	CaffeIPCSendMsg(*ClientSocket, MsgPrep);
	{
		CaffeIpc MsgBack;
		CaffeIPCRcvMsg(*ClientSocket, MsgBack);
		if (MsgBack.type() == CaffeIpc::PREP_GEN_FAILED) {
			LOG(INFO) << "Prep gen failed but not giving up yet!\n";
			return false;			
		}
		else if (MsgBack.type() != CaffeIpc::PREP_GEN_DONE) {
			std::cerr << "Problem with prep server init. Client bids bye!\n";
			b_fatal = true;
			return false;
		}
	}
	
	//IPCClientClose(ClientSocket);
	
	CallNetGen(config_name, 1);
	
	return true;
}

bool SimplifyMatchRequirements(	vector<pair<int, bool> >& AccessFilterOrderTbl, 
								CaffeGenDef& gen_data, 
								map<string, int>& var_tbl_idx_map,
								int& i_var_name)
{
	/*
	make sure that if you delete the last input net_value that you replace it with the dummy
	make sure you don't delete the last node, even on the access removal part
			handle ultimate failure of this function by starting another sentence or taking a step back
	 */
			
	const bool cb_net_val_input = true;
	int num_access_nodes = 0;
	int num_access_with_match = 0;
	int i_first_in_order_tbl_with_match = -1; // first from the end. note. note also not the index of the access (access_order.first)
	int i_last_in_order_tbl_with_match = -1;
	// The following number beyond the first access record with a match, we will remove the 
	// the access record itself rather than remove the match string
	const int remove_num_beyond_match = 4; 
	
	
	for (int i_access = AccessFilterOrderTbl.size() - 1; i_access >= 0; i_access--) {
		pair<int, bool>& access_order = AccessFilterOrderTbl[i_access];
		
		if (!access_order.second) continue;
		
		num_access_nodes++;
		
		const CaffeGenDef::DataAccess& access = gen_data.access_fields(access_order.first);
		if (access.has_dep_type_to_match() || access.has_pos_to_match()) {
			num_access_with_match++;
			if (i_last_in_order_tbl_with_match == -1) {
				i_last_in_order_tbl_with_match = i_access;
			}
			i_first_in_order_tbl_with_match = i_access;
		}
		
	}
	
	if (num_access_nodes <= 1) {
		LOG(INFO) << "SimplifyMatchRequirements fails. Cannot simplify beyond one access node.\n";
		return false;
	}
	
	// the following represents how long the tail of access nodes with a restricting 
	// match requirement is. The tail, in this case is the opposite end of the target
	// output node and is the last to be executed
	int not_matching_gap = AccessFilterOrderTbl.size() - i_last_in_order_tbl_with_match;

	int i_limit_match_remove = ((	not_matching_gap > remove_num_beyond_match) 
								?	i_last_in_order_tbl_with_match
								:	i_first_in_order_tbl_with_match);
	
	for (int i_access = AccessFilterOrderTbl.size() - 1; i_access >= 0; i_access--) {
		pair<int, bool>& access_order = AccessFilterOrderTbl[i_access];
		
		if (!access_order.second) { // we only want to process the access records not the filters, for now
			continue;
		}
		
		if (i_access <= i_limit_match_remove) {
			LOG(INFO) << "SimplifyMatchRequirements: no valid pos_to_match or dep_to_match to clear.\n";
			break;
		}
		
		CaffeGenDef::DataAccess* access = gen_data.mutable_access_fields(access_order.first);
		if (access->has_pos_to_match()) {
			access->clear_pos_to_match();
			const string& src_var_name = access->var_name();
			CreateNetValue(	gen_data, i_var_name, CaffeGenDef::vetPOS, 
							src_var_name, cb_net_val_input, "POSVecTbl");
			return true;
		}
		if (access->has_dep_type_to_match()) {
			access->clear_dep_type_to_match();
			const string& src_var_name = access->var_name();
			CreateNetValue(	gen_data, i_var_name, CaffeGenDef::vetDepName, 
							src_var_name, cb_net_val_input, "DepVecTbl");
			return true;
		}
		
	}
	
	for (int i_access = AccessFilterOrderTbl.size() - 1; i_access >= 0; i_access--) {
		pair<int, bool>& access_order = AccessFilterOrderTbl[i_access];
		if (!access_order.second) continue;
		CaffeGenDef::DataAccess* access = gen_data.mutable_access_fields(access_order.first);
		if (access->has_dep_type_to_match() || access->has_pos_to_match()) {
			LOG(INFO) << "Surprise! Seem to be trying to delete an access node with a match requirement.\n";
			continue;
		}
		const string& access_name = access->var_name();
		bool b_keep_deleting_filters = true;
		while (b_keep_deleting_filters) {
			b_keep_deleting_filters = false;
			for (int idf = 0; idf < gen_data.data_filters_size(); idf++) {
				const CaffeGenDef::DataFilter& filter = gen_data.data_filters(idf);
				const CaffeGenDef::DataFilterOneSide& left_side = filter.left_side();
				const CaffeGenDef::DataFilterOneSide& right_side = filter.right_side();
				if (	(left_side.var_name_src() != access_name) 
					&&	(right_side.var_name_src() != access_name)) {
					continue;
				}
				::google::protobuf::RepeatedPtrField< CaffeGenDef::DataFilter >* data_filters = gen_data.mutable_data_filters();
				int i_last = gen_data.data_filters_size() - 1;
				if (idf != i_last) {
					data_filters->SwapElements(idf, i_last);
				}
				data_filters->RemoveLast();
				b_keep_deleting_filters = true;
				break;
			}
		}
		bool b_keep_deleting_net_values = true;
		while (b_keep_deleting_net_values) {
			b_keep_deleting_net_values = false;
			for (int inv = 0; inv < gen_data.net_values_size(); inv++) {
				const CaffeGenDef::NetValue& nv = gen_data.net_values(inv);
				if (nv.var_name_src() != access_name) continue;
				::google::protobuf::RepeatedPtrField< CaffeGenDef::NetValue >* net_values = gen_data.mutable_net_values();
				int i_last = gen_data.net_values_size() - 1;
				if (inv != i_last) {
					net_values->SwapElements(inv, i_last);
				}
				net_values->RemoveLast();
				b_keep_deleting_net_values = true;
				break;
			}
		}
		int num_nv_inputs_left = 0;
		for (int inv = 0; inv < gen_data.net_values_size(); inv++) {
			const CaffeGenDef::NetValue& nv = gen_data.net_values(inv);
			if (nv.b_input()) {
				num_nv_inputs_left++;
			}
		}
		if (num_nv_inputs_left == 0) {
			CreateNetValue(	gen_data, i_var_name, 
							CaffeGenDef::vetDummy, string(""), 
							cb_net_val_input, "YesNoTbl");
		}
		
		::google::protobuf::RepeatedPtrField< CaffeGenDef::DataAccess >* accesses = gen_data.mutable_access_fields();
		int i_last = gen_data.access_fields_size() - 1;
		if (access_order.first != i_last) {
			accesses->SwapElements(access_order.first, i_last);
		}
		accesses->RemoveLast();
		return true;

	}
	return false;
}

void RemoveOneToMatchStr(map<string, int>& var_names_map, CaffeGenDef& gen_data, string& the_one_var_name, int& i_var_name)
{
	const bool cb_net_val_input = true;
	vector<string> vars_found;
	vector<string> next_next_vars_away;
	vector<string> next_vars_away;
	next_vars_away.push_back(the_one_var_name);
	vars_found.push_back(the_one_var_name);
	
	while (true) {
		for (int i_var = 0; i_var < next_vars_away.size(); i_var++) {
			string focus_var_name = next_vars_away[i_var];

			for (int i_filter = 0; i_filter < gen_data.data_filters_size(); i_filter++) {
				const CaffeGenDef::DataFilter& data_filter = gen_data.data_filters(i_filter);
				string other_var;
				bool b_match = false;
				if (data_filter.left_side().var_name_src() == focus_var_name) {
					other_var = data_filter.right_side().var_name_src();
					b_match = true;
				}
				else if (data_filter.right_side().var_name_src() == focus_var_name) {
					other_var = data_filter.left_side().var_name_src();
					b_match = true;
				}
				if (!b_match) continue;
				map<string, int>::iterator it_names_map = var_names_map.find(other_var);
				if (it_names_map == var_names_map.end()) {
					continue;
				}
				bool b_found = false;
				for (int i_already_found = 0; i_already_found < vars_found.size(); i_already_found++) {
					if (other_var == vars_found[i_already_found]) {
						b_found = true;
						break;
					}
				}
				if (b_found) {
					b_match = false;
				}
				if (!b_match) continue;
				next_next_vars_away.push_back(other_var);
				vars_found.push_back(other_var);
				 
			}
		}
		if (next_next_vars_away.empty() && (gen_data.access_fields_size() > 0)) {
			int i_remove = rand() % next_vars_away.size();
			string focus_var_name = next_vars_away[i_remove];
			map<string, int>::iterator it_names_map = var_names_map.find(focus_var_name);
			if (it_names_map == var_names_map.end()) {
				LOG(INFO) << "Error. Should be impossible for var not to be in table\n";
				continue;
			}
			int i_access = it_names_map->second;

//			::google::protobuf::RepeatedPtrField< ::CaffeGenDef_DataAccess >* access_fields = gen_data.mutable_access_fields();
//			int i_last = gen_data.access_fields_size() - 1;
//			if (i_access != i_last) {
//				access_fields->SwapElements(i_access, i_last);
//			}
//			access_fields->RemoveLast();
//			var_names_map.erase(focus_var_name);
			CaffeGenDef::DataAccess* access = gen_data.mutable_access_fields(i_access);
			if (access->has_pos_to_match()) {
				access->clear_pos_to_match();
				const string& src_var_name = access->var_name();
				CreateNetValue(	gen_data, i_var_name, CaffeGenDef::vetPOS, 
								src_var_name, cb_net_val_input, "POSVecTbl");
			}
			if (access->has_dep_type_to_match()) {
				access->clear_dep_type_to_match();
				const string& src_var_name = access->var_name();
				CreateNetValue(	gen_data, i_var_name, CaffeGenDef::vetDepName, 
								src_var_name, cb_net_val_input, "DepVecTbl");
			}
			var_names_map.erase(focus_var_name);
			break;
		}
		next_vars_away = next_next_vars_away;
		next_next_vars_away.clear();
	}
}

bool CGenGen::DoGen(string& data_core_dir, bool& b_fatal) {
	b_fatal = true; // initially all errors are fatal
	const bool cb_net_val_input = true;
	string gen_name = string("gengen") + gen_multinet_to_string(time(NULL));
	CaffeGenDef gen_data;

	gen_data.set_name(gen_name);
	gen_data.set_files_core_dir(data_core_dir + string("NetGen/") + gen_name + "/");
	gen_data.set_test_list_file_name("data/test");
	gen_data.set_train_list_file_name("data/train");
	gen_data.set_net_end_type(CaffeGenDef::END_ONE_HOT);
	gen_data.set_proto_file_name("models/train.prototxt");
	gen_data.set_model_file_name("models/g_best.caffemodel");
	gen_data.set_config_file_name("data/config.prototxt");
	gen_data.set_netgen_output_file_name("models/netgen_output.prototxt");
	gen_data.set_num_accuracy_candidates(1);

	int iVarName = 0;
	//bool b_dummy_input = true; // true until a real net value has been used for input

	CreateNetValue(	gen_data, iVarName, 
					CaffeGenDef::vetDummy, string(""), 
					cb_net_val_input, "YesNoTbl");

	map<string, int> var_names_map;
	string the_one_var_name;
	
	//vector<pair<string, striniVarNameg> > InputFields;
	for (int irec = 0; irec < SentenceRec.size(); irec++) {
		vector<pair<int, string> > gov_one_siders;
		vector<pair<int, string> > dep_one_siders;
		
		vector<DepRec>& deps = SentenceRec[irec].Deps;
		for (int iidep = 0; iidep < deps.size(); iidep++) {
			string sVarName = string("DepType") 
					+ gen_multinet_to_string(iVarName++);
			CaffeGenDef::DataAccess * access_field = gen_data.add_access_fields();
			access_field->set_var_name(sVarName);
			access_field->set_access_type(CaffeGenDef::ACCESS_TYPE_DEP);
			int gov = (int)(signed char)deps[iidep].Gov;
			if (gov >= 0) {
				gov_one_siders.push_back(make_pair(iidep, sVarName));
			}
			int dep = (int)(signed char)deps[iidep].Dep;
			if (dep >= 0) {
				dep_one_siders.push_back(make_pair(iidep, sVarName));
			}
//			CaffeGenDef::NetValue * net_value = gen_data.add_net_values();
//			string s_net_value_var_name = string("NetValue") 
//					+ gen_multinet_to_string(iVarName++);
//			net_value->set_var_name(s_net_value_var_name);
//			net_value->set_var_name_src(sVarName);
//			net_value->set_vet(CaffeGenDef::vetDepName);
			if (SentenceAvailList[irec].Deps[iidep].iDep == datConst) {
				int iDepType = (int)(signed char)(deps[iidep].iDep);
				//InputFields.push_back(make_pair(sVarName, "DepVecTbl")); // because we are iterating dep recs
				if ((iDepType < 0) || (iDepType >= DepNames.size())) {
					::google::protobuf::RepeatedPtrField< CaffeGenDef::DataAccess >*
							access_fields = gen_data.mutable_access_fields();
					access_fields->RemoveLast();
					continue;
				}
				string& DepTypeName = DepNames[iDepType];
				access_field->set_dep_type_to_match(DepTypeName);
				var_names_map[sVarName] = gen_data.access_fields_size() - 1;
//				net_value->set_b_input(true);
//				net_value->set_vec_table_name("DepVecTbl");
//				CaffeGenData::DataFilter * data_filter = gen_data.add_data_filters();
//				data_filter->set_var_name(sVarName);
//				data_filter->set_match_string(DepTypeName);
			}
			else if (SentenceAvailList[irec].Deps[iidep].iDep == datTheOne) {
				CreateNetValue(	gen_data, iVarName, 
								CaffeGenDef::vetDepName, sVarName, 
								!cb_net_val_input, "DepNumTbl");
				the_one_var_name = sVarName;
			}
			else {
				::google::protobuf::RepeatedPtrField< CaffeGenDef::DataAccess >*
						access_fields = gen_data.mutable_access_fields();
				access_fields->RemoveLast();
				continue;
			}
		}
		
		vector<WordRec>& wrecs = SentenceRec[irec].OneWordRec;
		for (int iwrec = 0; iwrec < wrecs.size(); iwrec++) {
			string sVarName = string("WREC") 
					+ gen_multinet_to_string(iVarName++);
			CaffeGenDef::DataAccess * access_field = gen_data.add_access_fields();
			access_field->set_var_name(sVarName);
			access_field->set_access_type(CaffeGenDef::ACCESS_TYPE_WORD);
//			CaffeGenDef::NetValue * net_value = gen_data.add_net_values();
//			string s_net_value_var_name = string("NetValue") 
//					+ gen_multinet_to_string(iVarName++);
//			net_value->set_var_name(s_net_value_var_name);
//			net_value->set_var_name_src(sVarName);
//			net_value->set_vet(CaffeGenDef::vetPOS);
			for (int iBoth= 0; iBoth < 2; iBoth++) {
				vector<pair<int, string> >& one_siders = (iBoth ? dep_one_siders : gov_one_siders);
				CaffeGenDef::MatchType mt = CaffeGenDef::mtDEP_GOV_RWID;
				if (iBoth == 1) {
					mt = CaffeGenDef::mtDEP_DEP_RWID;
				}
				for (int ios = 0; ios < one_siders.size(); ios++) {
					DepRec& dep = SentenceRec[irec].Deps[one_siders[ios].first];
					int WID = (iBoth ? dep.Dep : dep.Gov);
					if (WID != iwrec) {
						continue;
					}
					CaffeGenDef::DataFilter * data_filter = gen_data.add_data_filters();
					// allocate and pass to structure thereby passing (de)allocation responsibility
					CaffeGenDef::DataFilterOneSide * left_side = new CaffeGenDef::DataFilterOneSide;
					left_side->set_var_name_src(one_siders[ios].second);
					left_side->set_mt(mt);
					data_filter->set_allocated_left_side(left_side);
					//do right side
					CaffeGenDef::DataFilterOneSide * right_side = new CaffeGenDef::DataFilterOneSide;
					right_side->set_var_name_src(sVarName);
					right_side->set_mt(CaffeGenDef::mtWORD_RWID);
					data_filter->set_allocated_right_side(right_side);
				}
			}
			if (SentenceAvailList[irec].WordRecs[iwrec].POS == datConst) {
				string sPOS = wrecs[iwrec].POS;
				access_field->set_pos_to_match(sPOS);
				var_names_map[sVarName] = gen_data.access_fields_size() - 1;
//				net_value->set_b_input(true);
//				net_value->set_vec_table_name("POSVecTbl"); // WORD_VEC_TBL_NAME
				if (SentenceAvailList[irec].WordRecs[iwrec].Word == datConst) {
					CreateNetValue(	gen_data, iVarName, CaffeGenDef::vetWord, sVarName, 
									cb_net_val_input, "WordVecTbl");
				}
				else if (SentenceAvailList[irec].WordRecs[iwrec].Word == datTheOne) {
					// double output for word. The word and the max dep govs
					CreateNetValue(	gen_data, iVarName, CaffeGenDef::vetWord, sVarName, 
									!cb_net_val_input, "WordVecTbl");
					CreateNetValue(	gen_data, iVarName, CaffeGenDef::vetNumDepGovs, sVarName, 
									!cb_net_val_input, "OrdinalVecTbl");
					gen_data.set_net_end_type(CaffeGenDef::END_MULTI_HOT);
					the_one_var_name = sVarName;
				}
				
			}
			else if (SentenceAvailList[irec].WordRecs[iwrec].POS == datTheOne) {
//				net_value->set_b_input(false);
//				net_value->set_vec_table_name("POSNumTbl");
				CreateNetValue(	gen_data, iVarName, CaffeGenDef::vetPOS, sVarName, 
								!cb_net_val_input, "POSNumTbl");
				gen_data.set_net_end_type(CaffeGenDef::END_ONE_HOT);
				the_one_var_name = sVarName;
			}
			else {
				cerr << "Error. Unknown option for availability of word rec\n";
				continue;
			}
		}
	}

	bool b_keep_trying = true;
	while (b_keep_trying) {
		{
			string ProtoCoreDir = data_core_dir + "GenGen/";
			string proto_fname = ProtoCoreDir + gen_name + ".prototxt";
			ofstream gengen_ofs(proto_fname.c_str());
			google::protobuf::io::OstreamOutputStream* gengen_output 
				= new google::protobuf::io::OstreamOutputStream(&gengen_ofs);
			//ofstream f_config(ConfigFileName); // I think this one is wrong
			bool b_all_good = false;
			if (gengen_ofs.is_open()) {
				google::protobuf::TextFormat::Print(gen_data, gengen_output);
				b_all_good = true;

			}
			delete gengen_output;
			if (!b_all_good) {
				cerr << "Error. Failed to open and parse prototext file: " << proto_fname << endl;
				return false;
			}
			new_model_file_name = proto_fname;
		}

		const string NetGenCoreDir = data_core_dir + "NetGen/";
		const string dname = NetGenCoreDir + gen_name;
		fs::path dir(dname);
		if (!fs::create_directory(fs::path (dname))) {
			cerr << "DoGen Error: Failed to create directory for NetGen! \n";
			return false;
		}
		const string dname_models = dname + "/models";
		if (!fs::create_directory(fs::path (dname_models))) {
			cerr << "DoGen Error: Failed to create directory for NetGen! \n";
			return false;
		}
		const string dname_data = dname + "/data";
		if (!fs::create_directory(fs::path (dname_data))) {
			cerr << "DoGen Error: Failed to create directory for NetGen! \n";
			return false;
		}

		string config_fname = data_core_dir + "NetGen/" + gen_name + "/data/config.prototxt";
		// Errors in the following section not all fatal
		b_fatal = false;
		if (!PrepGen(new_model_file_name, config_fname, b_fatal)) {
			LOG(INFO) << "DoGen Error: Failed to prepare data for NetGen! \n";
			fs::remove_all(fs::path(dname));
			fs::remove(fs::path(new_model_file_name));
			if (b_fatal) {
				return false;
			}
			if (var_names_map.empty()) {
				LOG(INFO) << "DoGen Completion: Generalizing for NetGen is not longer possible. Abandoning this extension for the sentence. \n";
				return false;
			}
			vector<pair<int, bool> > AccessFilterOrderTbl;
			//CaffeGenDef * gen_data,
			map<string, int> var_tbl_idx_map;
			if (!CreateAccessOrder(	AccessFilterOrderTbl, &gen_data,
									var_tbl_idx_map) ) {
				return false;
			}

			if (!SimplifyMatchRequirements(	AccessFilterOrderTbl, gen_data,
											var_tbl_idx_map, iVarName)) {
				LOG(INFO) << "DoGen Completion: Generalizing for NetGen is no longer possible. Abandoning this extension for the sentence. \n";
				return false;
			}
			//RemoveOneToMatchStr(var_names_map, gen_data, the_one_var_name, iVarName);
			continue;
		}
		b_keep_trying = false;
	}
	
	return true;
	
#ifdef STILL_TO_PROCESS
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
			for (int i_both = 0; i_both < 2; i_both++) {
				int WID = -1;
				if (i_both == 0) {
					if (	(deps[iidep].Gov != (uchar)-1) 
						&&	(SentenceAvailList[irec].Deps[iidep].Gov == datConst)) {

						sVarName = string("Gov") 
								+ gen_multinet_to_string(iVarName++);
						data_field = gen_data.add_data_fields();
						data_field->set_var_name(sVarName);
						data_field->set_field_type(CaffeGenData::FIELD_TYPE_GOV_RWID);
						WID = (int)(signed char)deps[iidep].Gov;
					}
				}
				else {
					if (	(deps[iidep].Dep != -1) 
						&&	(SentenceAvailList[irec].Deps[iidep].Dep == datConst)) {

						sVarName = string("Dep") 
								+ gen_multinet_to_string(iVarName++);
						data_field = gen_data.add_data_fields();
						data_field->set_var_name(sVarName);
						data_field->set_field_type(CaffeGenData::FIELD_TYPE_DEP_RWID);
						WID = (int)(signed char)deps[iidep].Dep;
					}
				}
				if (WID == -1) {
					continue;
				}
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
					InputFields.push_back(make_pair(sVarName, "POSVecTbl")); // because we are iterating dep recs
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

#endif // #ifdef STILL_TO_PROCESS
	
}


void SingleNet::Init() {


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

void MultiNet::AddModel(string model_file_name)
{
	const bool cb_model_owns_tbls = true;
	model_data_arr_.push_back(SModelData());
	SModelData& model_data = model_data_arr_.back();

	CGenDef * init_data = new CGenDef(tbls_data_, !cb_model_owns_tbls);

	if (	!init_data->ModelInit(model_file_name) 
		||	!init_data->ModelPrep()) {
		delete init_data;
		model_data_arr_.pop_back();
		return;
	}
	model_data.NetGenData = init_data;
	// extract these from the actual model!!!
	model_data.input_layer_name_ = "data";
	model_data.output_layer_name_ = "squash3";
	model_data.model_file_name_ = init_data->getGenDef()->files_core_dir() + init_data->getGenDef()->proto_file_name();
	model_data.trained_file_name_ = init_data->getGenDef()->files_core_dir() + init_data->getGenDef()->model_file_name();
	
}

bool MultiNet::PreInit()
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	ClientSocket = NULL;
	
	int num_tries = 0;
	const int c_max_tries = 5;
	while (ClientSocket == NULL) {
		usleep(1000);
		ClientSocket = IPCClientInit(c_host_name, c_prep_gen_port_num);
		if (num_tries++ > c_max_tries) {
			cerr << "Error. Failed to make contact with data preparation server.\n";
			return false;
		}
		
	}
	
	
	string sModelDir = data_core_dir_ + "GenGen";
	string sTblModelsFile = "/TblDefs/tbls.prototxt";
	
	string fname = sModelDir + sTblModelsFile;
	tbls_data_ = new CGenDefTbls (fname);
	if (!tbls_data_->bInitDone) {
		cerr << "Failed to initialize tables def file. Terminating.\n";
		return false;
	}
	
	fs::directory_iterator end_iter;
	
	if ( fs::exists(sModelDir) && fs::is_directory(sModelDir)) {
		for( fs::directory_iterator dir_iter(sModelDir) ; dir_iter != end_iter ; ++dir_iter)	{
			if (fs::is_regular_file(dir_iter->status()) ) {
				if (dir_iter->path().extension() == ".prototxt") {
					LOG(INFO) << dir_iter->path() << std::endl;
					AddModel(dir_iter->path().c_str());
				}
				
			}
		}
		
//    {
//      result_set.insert(result_set_t::value_type(fs::last_write_time(dir_iter->path()), *dir_iter));
//    }
//  }
	}
	
	return true;
}

enum ETheOneType {
	totPOS,
	totWord,
	totDepName,
};

void PrintSentence(SSentenceRec& SRec, vector<string>& DepNamesTbl, vector<int>& max_dep_gov_list)  {
	std::vector<DepRec>& deps = SRec.Deps;
	vector<WordRec>& wrecs = SRec.OneWordRec;
	
	for (int id = 0; id < deps.size(); id++) {
		string gov_pos, dep_pos, dep_word, gov_word;
		int gov_max_govs = 0, dep_max_govs = 0;
		
		if (deps[id].Gov  < (uchar)255) {
			gov_pos = wrecs[deps[id].Gov].POS;
			gov_word = wrecs[deps[id].Gov].Word;
			gov_max_govs = max_dep_gov_list[deps[id].Gov];
		}
		else {
			gov_pos = "."; gov_word = ".";
		}
		if (deps[id].Dep  < (uchar)255) {
			dep_pos = wrecs[deps[id].Dep].POS;
			dep_word = wrecs[deps[id].Dep].Word;
			dep_max_govs = max_dep_gov_list[deps[id].Dep];
		}
		else {
			dep_pos = "."; dep_word = ".";
		}
		cerr	<< DepNamesTbl[deps[id].iDep] << " (" 
				<< gov_word << "{" << gov_pos << "/" << gov_max_govs << "} [" << (int)deps[id].Gov << "], "
				<< dep_word << "{" << dep_pos << "/" << dep_max_govs << "} [" << (int)deps[id].Dep << "])\n";
		LOG(INFO)	<< DepNamesTbl[deps[id].iDep] << " (" 
				<< gov_word << "{" << gov_pos << "/" << gov_max_govs << "} [" << (int)deps[id].Gov << "], "
				<< dep_word << "{" << dep_pos << "/" << dep_max_govs << "} [" << (int)deps[id].Dep << "])\n";
		google::FlushLogFiles(0);
	}
	
}

bool MultiNet::ComposeSentence(	vector<SingleNet>& nets,
						const string& word_file_name,
						const string& word_vector_file_name,
						bool& b_fatal) {

	const int c_max_word_length = 15;
	const int c_max_max_dep_govs = 5;
	word_vector_file_name_ = word_vector_file_name;
	//output_layer_idx_arr_ = vector<int>(5, -1);

	words_per_input_ = 1;
	
	vector<CorefRec> CorefSoFar;
	vector<SSentenceRec> SentenceList(1);
	vector<vector<int> > max_dep_gov_list(1);
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
	max_dep_gov_list[0].push_back(0);

	SSentenceRecAvail& SRecAvail = SentenceAvailList[0];
	SRecAvail.WordRecs.push_back(SWordRecAvail(datNotSet));
	SRecAvail.Deps.push_back(SDepRecAvail());
	SDepRecAvail& DepAvail = SRecAvail.Deps.back();
	DepAvail.iDep = datConst;
	DepAvail.Gov = datConst;
	DepAvail.Dep = datConst;
	SWordRecAvail& WRecAvail = SRecAvail.WordRecs.back();
	WRecAvail.POS = datTheOne;
	int i_srec_the_one = 0;
	int i_wid_the_one = 0;
	int i_did_the_one = -1;
	ETheOneType tot = totPOS;
	
	
	
	p_nets_ = &nets;

	CGenGen GenGen(	SentenceList, CorefSoFar, SentenceAvailList, CorefAvail, 
					tbls_data_->getDepNamesTbl(), ClientSocket);
	//GenGen.PrepGen(string("blah"));
	
	bool bKeepGoing = true;
	bool b_pick_last_net = false;
	
	vector<SDataForVecs > pre_vec_net_data;
	
	while (bKeepGoing) {
		int i_net_chosen = -1;
		for (int in = 0; in < nets.size(); in++) {
			if (b_pick_last_net && (in < (nets.size() - 1) )) {
				continue;
			}
			b_pick_last_net = false; 
			
			SModelData& model_def = model_data_arr_[in];

			//int NumOutputNodesNeeded = -1;
			vector<SDataForVecs > DataForVecs;
			
			CGenModelRun GenModelRun(	*model_def.NetGenData, SentenceList, CorefSoFar, 
										SentenceAvailList, CorefAvail, DataForVecs);

			GenModelRun.setReqTheOneOutput();

			if (!GenModelRun.DoRun()) {
				b_fatal = true;
				return false;
			}

			if (DataForVecs.size() == 0) {
				continue;
			}

			if (DataForVecs.size() != 1) {
				LOG(INFO) <<  "ComposeSentence: Not clear how there could be " << DataForVecs.size() << " records for the one. Please investigate\n";
				sort(DataForVecs.begin(), DataForVecs.end(), SDataForVecs::SortFn);
			}
			
			i_net_chosen = in;
			pre_vec_net_data = DataForVecs;
			break;
		}
		
		if (i_net_chosen == -1) {
			if (!GenGen.DoGen(data_core_dir_, b_fatal)) {
				// might have made b_fatal = true but exiting anyway
				// if, not go for another sentence
				if (b_fatal) {
					return false;
				}
				break;
			}
			
			// run gotit2
			// run netgen
			// add new net to nets
			// break; // remove
			// go back and try current sentence on last item of list
			string& new_net_file_name = GenGen.getNewModelFileName();
			AddModel(new_net_file_name);
			SModelData& model_data = model_data_arr_.back();
			nets.push_back(SingleNet(	model_data.model_file_name_, model_data.trained_file_name_, 
										model_data.input_layer_name_, model_data.output_layer_name_));
			b_pick_last_net = true;
			continue;
		}

		// if fall through to here, a net was chosen to predict datTheOne
		
		SingleNet& net = nets[i_net_chosen];
		net.Init();

		SModelData& model_data = model_data_arr_[i_net_chosen];


		vector<pair<int, int> >& InputTranslateTbl = model_data.NetGenData->getInputTranslateTbl();
		vector<pair<int, int> > OutputTranslateTbl = model_data.NetGenData->getOutputTranslateTbl();
		
		vector<pair<string, vector<float> > > VecArr;
		vector<vector<vector<float> >* >& VecTblPtrs = model_data.NetGenData->getVecTblPtrs();
		vector<map<string, int>*>& TranslateTblPtrs = model_data.NetGenData->getTranslateTblPtrs();

		Blob<float>* predict_input_layer =  net.GetInputVec();
		Blob<float>* predict_output_layer = net.GetOutputVec();

		net.PrepForInput(); // I think this loads the hdl5 data which can't be prevented
		float* p_in = predict_input_layer->mutable_cpu_data();
		float * ppd = p_in;
		
		vector<int>& IData = (pre_vec_net_data[0].IData);
				
		for (int ii = 0; ii < InputTranslateTbl.size(); ii++) {
			pair<int, int>& itt = InputTranslateTbl[ii];
			vector<float>& vec = (*VecTblPtrs[itt.second])[IData[ii]];
			for (int iv = 0; iv < vec.size(); iv++) {
				*ppd++ = vec[iv];
			}			
		}
		

		/*float v_loss = */net.ComputeOutput();
		const float* p_v_out = predict_output_layer->cpu_data();  
		bool b_the_one_found = false;
		string new_val_for_the_one;
		int index_for_the_one = -1;
		bool b_max_dep_govs_found = false;
		int max_dep_govs = -1;
		vector<int>& output_set_num_nodes = model_data.NetGenData->getOutputSetNumNodes();
		for (int ios = 0; ios < output_set_num_nodes.size(); ios++) {
			vector<float> VecOfOutput;
			//int NumValsInOutput = net.output_layer_dim();
			for (int iact = 0; iact < output_set_num_nodes[ios]; iact++) {
				VecOfOutput.push_back(*p_v_out++);
			}
			vector<pair<float, int> > SortedBest;
			vector<pair<float, int> > SortedBestDummy;
			//vector<int>& OData = (DataForVecs[0].OData);
			
			pair<int, int>& ott = OutputTranslateTbl[ios]; 
			vector<vector<float> > vec_for_one_hot;
			vector<vector<float> >* p_lookup_vec = VecTblPtrs[ott.second];
			if (model_data.NetGenData->getGenDef()->net_end_type() == CaffeGenDef::END_ONE_HOT) {
				vector<vector<float> >& one_hot_tbl = *(VecTblPtrs[ott.second]);
				for (int ioht = 0; ioht < one_hot_tbl.size(); ioht++) {
					vec_for_one_hot.push_back(vector<float>(output_set_num_nodes[ios], 0.0f));
					vector<float>& vec = vec_for_one_hot.back();
					vec[one_hot_tbl[ioht][0]] = 1.0f;
				}
				p_lookup_vec = &vec_for_one_hot;
			}
			srand((int)time(NULL)); for (int ir=0; ir<10; ir++) rand();
			int iMinDiffLbl = GetClosestWordIndex(VecOfOutput, output_set_num_nodes[ios],
					SortedBestDummy, 1, *(p_lookup_vec), 100);
			if (ott.second == model_data.NetGenData->getOrdinalVecTblIdx()) {
				// there should only be one output other than the first in cases where 
				// the first is word. This output is then 
				LOG(INFO) << "New max dep govs value: " << iMinDiffLbl << ".\n";
				b_max_dep_govs_found = true;
				max_dep_govs = max(iMinDiffLbl, c_max_max_dep_govs);
			}
			else { // normal case, output translation is a word, depname, pos etc.
				// for all cases besides word, there is only one output 
				// for word searches the forst ouitput is the word itself and the
				// 
				if (TranslateTblPtrs[ott.second] == NULL) {
					cerr << "Serious error. accessing translate tbl for yesno or ordinals that have no sym table";
					b_fatal = true;
					return false;
				}
				map<string, int>& SymTbl = *(TranslateTblPtrs[ott.second]);
				map<string, int>::iterator itSymTbl = SymTbl.begin();
				for (; itSymTbl != SymTbl.end(); itSymTbl++) {
					if (itSymTbl->second == iMinDiffLbl) {
						string stot = ((tot == totPOS) ? ("POS") : ((tot == totWord) ? ("word") : ("depname")));
						LOG(INFO) << "New " << stot << " value: " << itSymTbl->first << endl;
						new_val_for_the_one = itSymTbl->first;
						index_for_the_one = iMinDiffLbl;
						b_the_one_found = true;
						break;
					}
				}
			}
		}
		
		if (!b_the_one_found ) {
			cerr << "Error: No value found for datTheOne. Cannot continue!\n";
			break;
		}
		
		bool b_added_word = false;
		switch (tot) {
			case totPOS:
				SentenceList[i_srec_the_one].OneWordRec[i_wid_the_one].POS 
						= new_val_for_the_one;
				SentenceAvailList[i_srec_the_one].WordRecs[i_wid_the_one].POS
						= datConst;
				// right away ask for a word
				SentenceList[i_srec_the_one].OneWordRec[i_wid_the_one].Word.clear(); 
				SentenceAvailList[i_srec_the_one].WordRecs[i_wid_the_one].Word
						= datTheOne;
				b_added_word = true;
				break;
			case totWord:
				SentenceList[i_srec_the_one].OneWordRec[i_wid_the_one].Word 
						= new_val_for_the_one;
				SentenceAvailList[i_srec_the_one].WordRecs[i_wid_the_one].Word
						= datConst;
				if (b_max_dep_govs_found) {
					max_dep_gov_list[i_srec_the_one][i_wid_the_one] = max_dep_govs;
//					SentenceAvailList[i_srec_the_one].WordRecs[i_wid_the_one].max_dep_govs 
//							= max_dep_govs;
				}
				break;
			case totDepName:
				SentenceList[i_srec_the_one].Deps[i_did_the_one].iDep 
						= index_for_the_one;
				SentenceAvailList[i_srec_the_one].Deps[i_did_the_one].iDep
						= datConst;
				break;
			default:
				cerr <<"Not coded yet.";
				b_fatal = true;
				return false;
		}
		
		if (b_added_word) {
			// adding already done. Move on.
			tot = totWord;
			continue;
		}
		// first shot at adding: Make sure there are no dangling ends to dep records
		bool b_new_one_set = false;
		for (int irec = 0; irec < SentenceList.size(); irec++) {
			vector<DepRec>& deps = SentenceList[irec].Deps;
			for (int iidep = 0; iidep < deps.size(); iidep++) { 
				SDepRecAvail& depavail = SentenceAvailList[irec].Deps[iidep];
				bool b_gov_found = false;
				if (depavail.Gov == datNotSet) {
					LOG(INFO) << "Surprising result. Expect the gov to be set before the dep\n";
					b_new_one_set = true;
					b_gov_found = true;
				}
				else if (depavail.Dep == datNotSet) {
					b_new_one_set = true;
					b_gov_found = false;
				}
				if (b_new_one_set) {
					i_srec_the_one = irec;
					int WID = SentenceList[irec].OneWordRec.size();
					SentenceList[irec].OneWordRec.push_back(WordRec());
					max_dep_gov_list[irec].push_back(0);
					SentenceAvailList[irec].WordRecs.push_back(SWordRecAvail(datNotSet));
					SentenceAvailList[irec].WordRecs[WID].POS = datTheOne;
					i_wid_the_one = WID;
					tot = totPOS;
					if (b_gov_found) {
						deps[iidep].Gov = WID;
						depavail.Gov = datConst;
					}
					else {
						deps[iidep].Dep = WID;
						depavail.Dep = datConst;
					}
					break;
					
				}
			}
		}
		
		if (b_new_one_set) {
			continue;
		}
		// next shot at adding: Create a dep between two existing records
		// next shot at adding: Create a dangling dep record on one of the words
		// update 13/3/06. Looks like each word gets only one dep but can have multiple govs
		for (int irec = 0; irec < SentenceList.size(); irec++) {
			int num_wrecs = SentenceList[irec].OneWordRec.size();
			if (num_wrecs > c_max_word_length) {
				LOG(INFO) << "Sentence too long. Completing.\n";
				PrintSentence(SentenceList[irec], tbls_data_->getDepNamesTbl(), max_dep_gov_list[irec]);
				return true;
			}
			int max_govs_diff = 0;
			int WID = -1;
			for (int isr = 0; isr < num_wrecs; isr++) {
				int num_govs = 0;
				vector<DepRec>& deps = SentenceList[irec].Deps;
				for (int iidep = 0; iidep < deps.size(); iidep++) {
					int gov = (int)(signed char)deps[iidep].Gov;
					if (gov == isr) {
						num_govs++;
					}
				}
				int govs_diff = max_dep_gov_list[irec][isr] - num_govs;
				if (govs_diff > max_govs_diff) {
					WID = isr;
					max_govs_diff = govs_diff;
				}
			}
			if (WID == -1) {
				cerr << "Sentence complete.\n";
				PrintSentence(SentenceList[irec], tbls_data_->getDepNamesTbl(), max_dep_gov_list[irec]);
				return true;
			}
			//int WID = rand() % num_wrecs;
			{
				string gov_pos = SentenceList[irec].OneWordRec[WID].POS;
				string gov_word = SentenceList[irec].OneWordRec[WID].Word;

				LOG(INFO)	<< "Adding a dep relation for " << WID << ": " << gov_word 
							<< " {" << gov_pos << "}.\n";
				cerr << "Sentence so far:\n";
				PrintSentence(SentenceList[irec], tbls_data_->getDepNamesTbl(), max_dep_gov_list[irec]);
			}
			i_srec_the_one = irec;
			int DID = SentenceList[irec].Deps.size();
			SentenceList[irec].Deps.push_back(DepRec());
			SentenceAvailList[irec].Deps.push_back(SDepRecAvail(datNotSet));
			DepRec& deprec = SentenceList[irec].Deps.back();
			deprec.Dep = deprec.Gov = (uchar)-1;
//			if ((rand() % 2) == 0) {
//				deprec.Dep = WID;
//				SentenceAvailList[irec].Deps[DID].Dep = datConst;
//			}
//			else {
				deprec.Gov = WID;
				SentenceAvailList[irec].Deps[DID].Gov = datConst;
//			}
			SentenceAvailList[irec].Deps[DID].iDep = datTheOne;
			i_did_the_one = DID;
			tot = totDepName;
			// we are only working on 1 sentence for now
			b_new_one_set = true;
			break;
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
	return true;

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
		SumDiff *=	(1.0f +		((float)(rand() % rand_factor)
							/	((float)rand_factor * (float)rand_factor)));
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
				LOG(INFO) << w << " ";
				if (iw == 1) {
					if (iMinDiffLbl == -1) {
						LOG(INFO) << "XXX ";
					}
					else {
						string l = words_[iMinDiffLbl];
						LOG(INFO) << "(" << l << ") ";
					}
				}
			}
		}
		LOG(INFO) << std::endl;

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
			LOG(INFO) << "--> ";
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
				LOG(INFO) << w << " (" << SortedBest[ib].first << " vs. " << v_val << "), ";
			}
			LOG(INFO) << std::endl << "Reordered: ";
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
	if (!classifier.PreInit()) {
		return 1;
	}
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
	while (true) {
		// keep making sentences
		bool b_fatal = false;
		if (!classifier.ComposeSentence(nets,
						word_file_name, 
						word_vector_file_name,
						b_fatal)) {
			if (b_fatal) {
				break;
			}
		}
	}
	
	return 0;
	//classifier.Classify();
	
}
#endif // CAFFE_MULTINET_MAIN
     