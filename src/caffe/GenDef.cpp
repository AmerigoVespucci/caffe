// GenDef.cpp : 
// Calls the functions needed to parse and use GenData data 
//


  
#include <fcntl.h>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <vector>
#include <list>
#include <map>
//#include "stdafx.h"

//#include "MascReader.h"
#include "H5Cpp.h"

//#include "/dev/caffe/include/caffe/proto/GenData.pb.h"
#include "caffe/proto/GenData.pb.h"
#include "caffe/proto/GenDef.pb.h"
#include "caffe/proto/GenDataTbls.pb.h"
#include "caffe/GenData.hpp"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif


#ifdef  _MSC_VER
#pragma warning(disable : 4503)
#endif
	
using namespace std;	

const int c_ordinal_tbl_max = 10; // velo ad bichlal

template <typename T>
string gen_data_to_string ( T Number )
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}

#if 0
string CGenModelRun::GetRecFieldByIdx(	int SRecID, int WID, 
										CaffeGenData_FieldType FieldID, 
										DataAvailType& RetAvail, bool bUseAvail)
{
	RetAvail = datConst;
	WordRec& rec = SentenceRec[SRecID].OneWordRec[WID];		
	SWordRecAvail WordAvail(datConst); // default to all there
	if (bUseAvail) {
		WordAvail = SentenceAvailList[SRecID].WordRecs[WID];
		switch(FieldID) {
			case CaffeGenData::FIELD_TYPE_WORD: 
				RetAvail = WordAvail.Word;
				break;
			case CaffeGenData::FIELD_TYPE_WORD_CORE: 
				RetAvail = WordAvail.WordCore;
				break;
			case CaffeGenData::FIELD_TYPE_POS:
				RetAvail = WordAvail.POS;
				break;
			case CaffeGenData::FIELD_TYPE_WID:
			case CaffeGenData::FIELD_TYPE_RWID:
				RetAvail = datConst;
				break;
			default:
				RetAvail = datInvalid;
				break;
		}
		
		if (RetAvail != datConst) {
			return string();
		}
	}
	switch(FieldID) {
		case CaffeGenData::FIELD_TYPE_WORD: {
			string w = rec.Word;
			bool bWordOK = true;
			for (int iiw = 0; iiw < w.size(); iiw++) {
				char c = w[iiw];
				if (!isalpha(c)) {
					bWordOK = false;
					break;
				}
				if (isupper(c)) {
					w[iiw] = tolower(c);
				}
			}
			if (!bWordOK) {
				RetAvail = datInvalid;
				break;
			}
			return w;
		}
		case CaffeGenData::FIELD_TYPE_WORD_CORE: 
			return rec.WordCore;
		case CaffeGenData::FIELD_TYPE_POS:
			return rec.POS;
//		case fniIndex: I think this is a misunderstanding
//			return gen_data_to_string(FieldID);
		case CaffeGenData::FIELD_TYPE_WID:
			return (gen_data_to_string(WID)) ;
		case CaffeGenData::FIELD_TYPE_RWID:
			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string(WID)) ;
		default:
			RetAvail = datInvalid;
			break;
	}
	return string();
}
#endif // 0

CGenDefTbls::CGenDefTbls(string& sModelProtoName) 
{
	bInitDone = false;
	CaffeGenDataTbls * protoc_data = new CaffeGenDataTbls;

	ifstream proto_ifs(sModelProtoName.c_str());
	if (proto_ifs.is_open()) {
		google::protobuf::io::IstreamInputStream* proto_input 
			= new google::protobuf::io::IstreamInputStream(&proto_ifs);
		bool bGood = true;
		if (!google::protobuf::TextFormat::Parse(proto_input, protoc_data)) {
			cerr << "CGenDefTbls config file Parse failed for " << sModelProtoName << endl;
			bGood = false;
		}
		delete proto_input;
		if (!bGood) {
			return;
		}
	}
	else {
		cerr << "Error in CGenDefTbls. Model file " << sModelProtoName << " not found.\n";
		return;
	}
	
	//int NumVecTbls = protoc_data->vec_tbls_size();
	
	for (int ivt = 0; ivt < protoc_data->vec_tbls_size(); ivt++) {
		CaffeGenDataTbls::VecTbl vec_tbl = protoc_data->vec_tbls(ivt);
		TranslateTblNameMap[vec_tbl.name()] = ivt;
		bool bThisTblIsTheVecTbl = false;
		if (vec_tbl.name() == protoc_data->dep_name_vec_tbl()) {
			bThisTblIsTheVecTbl = true;
			DepNames.clear();
		}
		map<string, int>* pNameMap = new map<string, int>();
		TranslateTblPtrs.push_back(pNameMap);
		vector<vector<float> >* pVecTbl = new vector<vector<float> >();
		VecTblPtrs.push_back(pVecTbl);
		string VecTblPath = protoc_data->vec_tbls_core_path() + vec_tbl.path() + "/OutputVec.txt";
		ifstream VecFile(VecTblPath.c_str());
		if (VecFile.is_open()) {
			string ln;
			int NumVecs;
			int NumValsPerVec;
			VecFile >> NumVecs;
			VecFile >> NumValsPerVec;
//			if (bThisTblIsTheVecTbl) {
//				pDepNames->resize(NumVecs);
//			}

			getline(VecFile, ln); // excess .n
			//while (!VecFile.eof()) {
			for (int ic = 0; ic < NumVecs; ic++) {
				getline(VecFile, ln, ' ');
				string w;
				w = ln;
				if (w.size() == 0) {
					cerr << "There should be as many items in the vec file as stated on the first line of file.\n";
					return;
				}
				if (bThisTblIsTheVecTbl) {
					DepNames.push_back(w);
				}
				(*pNameMap)[w] = ic;
				pVecTbl->push_back(vector<float>());
				vector<float>& OneVec = pVecTbl->back();
				//vector<float>& OneVec = WordsVecs[iw];
				for (int iwv = 0; iwv < NumValsPerVec; iwv++) {
					if (iwv == NumValsPerVec - 1) {
						getline(VecFile, ln);
					}
					else {
						getline(VecFile, ln, ' ');
					}
					float wv;
					wv = ::atof(ln.c_str());
					OneVec.push_back(wv);
				}
			}
		}
	}

	vector<vector<float> > * YesNoTbl = new vector<vector<float> >;
	YesNoTbl->push_back(vector<float>(1, 0.0f));
	YesNoTbl->push_back(vector<float>(1, 1.0f));

	
	int YesNoTblIdx = TranslateTblPtrs.size();
	TranslateTblNameMap["YesNoTbl"] = YesNoTblIdx;
	TranslateTblPtrs.push_back(NULL);
	VecTblPtrs.push_back(YesNoTbl);
	
	vector<vector<float> > * ordinal_vec_tbl = new vector<vector<float> >;
	for (int io = 0; io < c_ordinal_tbl_max; io++) {
		ordinal_vec_tbl->push_back(vector<float>(c_ordinal_tbl_max, 0.0f));
		(*ordinal_vec_tbl)[io][io] = 1.0f;
	}

	int OrdinalTblIdx = TranslateTblPtrs.size();
	TranslateTblNameMap["OrdinalVecTbl"] = OrdinalTblIdx;
	TranslateTblPtrs.push_back(NULL);
	VecTblPtrs.push_back(ordinal_vec_tbl);
	
	
	delete protoc_data;
	bInitDone = true;

}

bool CGenDefTbls::getTableNameIdx(const string& TblName, int& idx)
{
	map<string, int>::iterator itNameMap = TranslateTblNameMap.find(TblName);
	if (itNameMap == TranslateTblNameMap.end()) {
		return false;
	}
	idx = itNameMap->second;
	return true;
}


bool CGenDef::ModelInit(string sModelProtoName ) 
{
	ICanReplaceTbl.clear();
	OCanReplaceTbl.clear();
	NumInstancesTbl[0].clear();
	NumInstancesTbl[1].clear();
	MaxInstancesTbl[0].clear();
	MaxInstancesTbl[1].clear();
	InputTranslateTbl.clear();
	OutputTranslateTbl.clear();
	bCanReplace = false;

	gen_def = new CaffeGenDef;
	
	ifstream proto_ifs(sModelProtoName.c_str());
	if (proto_ifs.is_open()) {
		google::protobuf::io::IstreamInputStream* proto_input 
			= new google::protobuf::io::IstreamInputStream(&proto_ifs);
		bool bGood = true;
		if (!google::protobuf::TextFormat::Parse(proto_input, gen_def)) {
			cerr << "GenDataModelInit config file Parse failed for " << sModelProtoName << endl;
			bGood = false;
		}
		delete proto_input;
		if (!bGood) {
			return false;
		}
	}
	else {
		cerr << "Error in GenDataModelInit. Model file " << sModelProtoName << " not found.\n";
		return false;
	}

	return true;

#if 0	
	InputTranslateTbl.clear();
	OutputTranslateTbl.clear();
	FirstAccessFieldsIdx.clear();
	VarNamesMap.clear();
	DataTranslateTbl.clear();
	DataFilterTbl.clear();
	
	gen_data = new CaffeGenData;
	
	ifstream proto_ifs(sModelProtoName.c_str());
	if (proto_ifs.is_open()) {
		google::protobuf::io::IstreamInputStream* proto_input 
			= new google::protobuf::io::IstreamInputStream(&proto_ifs);
		bool bGood = true;
		if (!google::protobuf::TextFormat::Parse(proto_input, gen_data)) {
			cerr << "GenDataModelInit config file Parse failed for " << sModelProtoName << endl;
			bGood = false;
		}
		delete proto_input;
		if (!bGood) {
			return false;
		}
	}
	else {
		cerr << "Error in GenDataModelInit. Model file " << sModelProtoName << " not found.\n";
		return false;
	}

	return true;
#endif // 0	
}

struct VarTbEl {
	bool bDorW;
	pair<int, int> R_DorW_ID;
};

struct TreeNavEl {
	TreeNavEl(int a_int, bool a_bool, bool a_bool2, int a_distance, bool a_bool3) {
		access_tbl_idx = a_int;
		b_dep_to_gov = a_bool;
		b_already_added = a_bool2 ;
		distance_from_target = a_distance;
		b_dep_not_word = a_bool3;
	}
	void add_to_list(list<TreeNavEl>& tree_list) {
		list<TreeNavEl>::iterator it_tree_list = tree_list.begin();
		bool b_inserted = false;
		int pos = 0;
		for (;it_tree_list != tree_list.end(); it_tree_list++) {
			if (distance_from_target < it_tree_list->distance_from_target) {
				tree_list.insert(it_tree_list, *this);
				b_inserted = true;
				break;
			}
			pos++;
		}
		if (!b_inserted) {
			tree_list.insert(it_tree_list, *this);
		}
		LOG(INFO) << "tree nav el " << access_tbl_idx << " inserted at pos " << pos << " of len " << tree_list.size() << endl;
	}
	int access_tbl_idx;
	bool b_dep_to_gov;
	bool b_already_added;
	int distance_from_target;
	bool b_dep_not_word;
};

const bool cb_dep_to_gov = true;
const bool cb_already_added = true;
const bool cb_dep_not_word = true;

bool CreateAccessOrder(	vector<pair<int, bool> >& AccessFilterOrderTbl, 
						CaffeGenDef * gen_def,
						map<string, int>& var_tbl_idx_map,
						bool b_target_wrec_src)
{
	AccessFilterOrderTbl.clear();
	var_tbl_idx_map.clear();

	vector<bool> filter_inserted_tbl(gen_def->data_filters_size(), false);

	// New Algortihm.
	// First find the access entry that corresponds to the target(s)
	// then work your way up a ladder of wrec, pointed to by dep, next to gov restart
	// If it is a single path tree, we go directly from target to start (root)
	// However, there might be two gov ptrs to the same wrrec (currently it seems that there are never more tha one dep pointer)
	// The other wrec is not on the main branch. So we store the branches and process them at the end
	for (int iaf = 0; iaf < gen_def->access_fields_size(); iaf++) {
		CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
		var_tbl_idx_map[access->var_name()] = iaf;			
		access->set_var_idx(iaf);
	}

	vector<int> targets;
	if (b_target_wrec_src) {
		for (int iaf = 0; iaf < gen_def->access_fields_size(); iaf++) {
			CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
			if (access->var_name() == "WREC_src") {
				targets.push_back(iaf);
				break;
			}
			
		}
	}
	else {
		for (int inv = 0; inv < gen_def->net_values_size(); inv++) {
			const CaffeGenDef::NetValue& net_value = gen_def->net_values(inv);
			if (net_value.var_name_src().length() == 0) {
				continue;
			}
			map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(net_value.var_name_src());
			if (it_var_tbl == var_tbl_idx_map.end()) {
				cerr << "Error: Gen Def proto accesses an entry in access vars that does not exist.\n";
				return false;
			}
			else if (!net_value.b_input()) {
				bool b_found = false;
				for (int it = 0; it < targets.size(); it++) {
					if (targets[it] == it_var_tbl->second) {
						b_found = true;
						break;
					}
				}
				if (!b_found) {
					targets.push_back(it_var_tbl->second);
				}
			}
		}
	}

	vector<bool> access_els_added(gen_def->access_fields_size(), false);
	list<TreeNavEl> tree_nav_stack;
	
	for (int itt = 0; itt < targets.size(); itt++) {
		int iaf = targets[itt];
		//tree_nav_stack.push_back();
		CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
		bool b_dep_not_word = false;
		if (access->access_type() == CaffeGenDef::ACCESS_TYPE_DEP) {
			b_dep_not_word = true;
		} 
		TreeNavEl(iaf, cb_dep_to_gov, !cb_already_added, 0, b_dep_not_word).add_to_list(tree_nav_stack);
	}
	
	while (!tree_nav_stack.empty()) {
		int iaf = tree_nav_stack.front().access_tbl_idx;
		bool b_dep_to_gov = tree_nav_stack.front().b_dep_to_gov;
		bool b_already_added = tree_nav_stack.front().b_already_added;
		int distance = tree_nav_stack.front().distance_from_target;
		bool b_dep_not_word = tree_nav_stack.front().b_dep_not_word;
		tree_nav_stack.pop_front();

		if (!b_already_added) {
			if (access_els_added[iaf]) continue;
			access_els_added[iaf] = true;
			AccessFilterOrderTbl.push_back(make_pair(iaf, true));// true because whether we are adding a dep or word, it's still not a filter
		}
		
		//b_on_main_branch = false; // set back to true later on if the branch continues
		CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
		string access_var_name = access->var_name();
		bool b_filter_found = false;
		//string deptype_var_name = "unfounded";
		string new_var_name = "unfounded";
		string access_gov_var_name = "unfounded";
		//int i_gov_var;
		int iaf_new;
		bool b_check_for_other_govs = false;
		for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
			if (filter_inserted_tbl[idf]) continue;

			CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
			CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
			CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
			CaffeGenDef::DataFilterOneSide* src_side = (b_dep_not_word ? left_side : right_side);
			CaffeGenDef::DataFilterOneSide* new_side = (b_dep_not_word ? right_side : left_side);
			// the condition in the following teriary op is subtle
			// if b_dep_to_gov, we're going up the tree and therefore if we are at a word we want a filter
			// that points to a DEP, so if b_dep_not_word == true, MatchType = GOV and if false, MatchType = DEP
			// if b_dep_to_gov == false, we're going down the tree, so if b_dep_not_word == false, we're at a word
			// and therefore we are looking for a gov, so MatchType = GOV
			// so it's an NXOR, which is simply boolean "equals"
			CaffeGenDef::MatchType Match = (	(b_dep_to_gov ==  b_dep_not_word) 
											?	CaffeGenDef::mtDEP_GOV_RWID 
											:	CaffeGenDef::mtDEP_DEP_RWID);
			// the right side by current convention is the id of wrec and the left side is either the dep or gov pointing to it
			if (right_side->mt() != CaffeGenDef::mtWORD_RWID) continue;
			//if (right_side->var_name_src() != access_var_name) continue;
			if (src_side->var_name_src() != access_var_name) continue;
			//if (left_side->mt() != CaffeGenDef::mtDEP_DEP_RWID) continue;
			if (left_side->mt() != Match) continue;
			//new_var_name = left_side->var_name_src();
			new_var_name = new_side->var_name_src();
			map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(new_var_name);
			if (it_var_tbl == var_tbl_idx_map.end()) {
				cerr << "Error: Gen Def proto accesses an entry in access vars that has ontological problems.\n";
				return false;
			}
			iaf_new = it_var_tbl->second;
			if (access_els_added[iaf_new]) continue;
			if (b_dep_not_word) { // if src was dep, the new is a word, only words get checked for extra govs
				b_check_for_other_govs = true;
			}
			access_els_added[iaf_new] = true;
			AccessFilterOrderTbl.push_back(make_pair(iaf_new, true));	
			AccessFilterOrderTbl.push_back(make_pair(idf, false));
			filter_inserted_tbl[idf] = true;
			left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
			right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
			b_filter_found = true;
			break;
		}
		if (!b_filter_found) {
			continue;
		}
		TreeNavEl(	iaf_new, 
					b_dep_to_gov, // don't change direction here. Only new branch adds can do that
					cb_already_added, // always true, except for the first addd
					distance+1, 
					!b_dep_not_word // toggle dep and word, dep -> word, word -> dep
					).add_to_list(tree_nav_stack);
#if 0		
		if (b_dep_to_gov) {
			if (!b_dep_not_word){
			}
			else { // b_dep_not_word. This option takes a dep and finds a word
				for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
					if (filter_inserted_tbl[idf]) continue;

					CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
					CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
					CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
					// the right side by current convention is the id of wrec and the left side is either the dep or gov pointing to it
					if (right_side->mt() != CaffeGenDef::mtWORD_RWID) continue;
					if (left_side->var_name_src() != access_var_name) continue;
					if (left_side->mt() != CaffeGenDef::mtDEP_GOV_RWID) continue;
					access_gov_var_name = right_side->var_name_src();
					map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(access_gov_var_name);
					if (it_var_tbl == var_tbl_idx_map.end()) {
						cerr << "Error: Gen Def proto accesses an entry in access vars that has ontological problems.\n";
						return false;
					}
					iaf_new = it_var_tbl->second;
					b_check_for_other_govs = true;
					if (access_els_added[iaf_new]) continue;
					access_els_added[iaf_new] = true;
					AccessFilterOrderTbl.push_back(make_pair(iaf_new, true));	
					AccessFilterOrderTbl.push_back(make_pair(idf, false));
					filter_inserted_tbl[idf] = true;
					left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
					right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
					b_filter_found = true;
					break;
				}
				if (!b_filter_found) continue;
				//iaf = i_gov_var;
				//b_on_main_branch = true; // go round for another go
	//			tree_nav_stack.push_front(TreeNavEl(i_gov_var, cb_dep_to_gov, 
	//												cb_already_added, distance+1));
				TreeNavEl(	i_gov_var, cb_dep_to_gov, 
							cb_already_added, distance+1, 
							!cb_dep_not_word).add_to_list(tree_nav_stack);
			}
		}
		else { // going from gov->dep as initiaied by finding a second gov ptr
			//CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
			//string deptype_var_name = "unfounded";
			// for this direction the access gov var name is simply the name we start the search with
			// the var is later used to see if there are any more deps that have this var as a gov
			access_gov_var_name = access_var_name;
			if (access->access_type() == CaffeGenDef::ACCESS_TYPE_DEP) {
				cerr << "Strange. A dep type access should not have appeared in the other_branches_head list.\n";
				deptype_var_name= access_var_name;
			} 
			else {
				for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
					if (filter_inserted_tbl[idf]) continue;

					CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
					CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
					CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
					// the right side by current convention is the id of wrec and the left side is either the dep or gov pointing to it
					if (right_side->mt() != CaffeGenDef::mtWORD_RWID) continue;
					if (right_side->var_name_src() != access_var_name) continue;
					if (left_side->mt() != CaffeGenDef::mtDEP_GOV_RWID) continue;
					deptype_var_name = left_side->var_name_src();
					map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(deptype_var_name);
					if (it_var_tbl == var_tbl_idx_map.end()) {
						cerr << "Error: Gen Def proto accesses an entry in access vars that has ontological problems.\n";
						return false;
					}
					int i_deptype_var = it_var_tbl->second;
					if (access_els_added[i_deptype_var]) continue;
					access_els_added[i_deptype_var] = true;
					AccessFilterOrderTbl.push_back(make_pair(i_deptype_var, true));	
					AccessFilterOrderTbl.push_back(make_pair(idf, false));
					filter_inserted_tbl[idf] = true;
					left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
					right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
					b_filter_found = true;
					break;
				}
				if (!b_filter_found) {
					continue;
				}
			}
			string access_dep_var_name = "unfounded";
			int i_access_dep_var;
			b_filter_found = false;
			for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
				if (filter_inserted_tbl[idf]) continue;
				
				CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
				CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
				CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
				// the right side by current convention is the id of wrec and the left side is either the dep or gov pointing to it
				if (right_side->mt() != CaffeGenDef::mtWORD_RWID) continue;
				if (left_side->var_name_src() != deptype_var_name) continue;
				if (left_side->mt() != CaffeGenDef::mtDEP_DEP_RWID) continue;
				access_dep_var_name = right_side->var_name_src();
				map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(access_dep_var_name);
				if (it_var_tbl == var_tbl_idx_map.end()) {
					cerr << "Error: Gen Def proto accesses an entry in access vars that has ontological problems.\n";
					return false;
				}
				i_access_dep_var = it_var_tbl->second;
				if (access_els_added[i_access_dep_var]) continue;
				access_els_added[i_access_dep_var] = true;
				AccessFilterOrderTbl.push_back(make_pair(i_access_dep_var, true));	
				AccessFilterOrderTbl.push_back(make_pair(idf, false));
				filter_inserted_tbl[idf] = true;
				left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
				right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
				b_filter_found = true;
				break;
			}
			if (!b_filter_found) continue;
			i_gov_var = iaf;
			iaf = i_access_dep_var;
			//b_on_this_branch = true; // go round for another go
//			tree_nav_stack.push_front(TreeNavEl(i_access_dep_var, !cb_dep_to_gov, 
//												cb_already_added, distance+1)); // we came here gov->var so we continue the search gov->var
			TreeNavEl(	i_access_dep_var, !cb_dep_to_gov, 
						cb_already_added, distance+1).add_to_list(tree_nav_stack);


		}
#endif // 0	
		if (b_check_for_other_govs) {
			// this will only be set true when a new word, not dep, has been added
			// we look to add all filters that have this guy as a gov
			CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf_new);
			string access_var_name = access->var_name();
			for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
				if (filter_inserted_tbl[idf]) continue;

				CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
				CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
				CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
				// the right side by current convention is the id of wrec and the left side is either the dep or gov pointing to it
				if (right_side->mt() != CaffeGenDef::mtWORD_RWID) continue;
				if (right_side->var_name_src() != access_var_name) continue;
				if (left_side->mt() != CaffeGenDef::mtDEP_GOV_RWID) continue;
				//deptype_var_name = left_side->var_name_src();
//				map<string, int>::iterator it_var_tbl = var_tbl_idx_map.find(access_gov_var_name);
//				if (it_var_tbl == var_tbl_idx_map.end()) {
//					cerr << "Error: Gen Def proto accesses an entry in access vars that has ontological problems.\n";
//					return false;
//				}
//				i_gov_var = it_var_tbl->second;
				LOG(INFO) << "Note. We need to add a branch here.\n";
	//			tree_nav_stack.push_back(TreeNavEl(	i_gov_var, !cb_dep_to_gov, 
	//												cb_already_added, distance+1)); // what we add is the var position of the access_gov_var_name
				// !cb_dep_not_word because only words can add a new branch
				TreeNavEl(	iaf_new, !cb_dep_to_gov, 
							cb_already_added, distance+1,
							!cb_dep_not_word).add_to_list(tree_nav_stack);

				//break; // just add the extra branch once. For more extra branches see processing of extra branches later
			}
		}

	}
	// iBoth here is adding access fields both:
	// 1. When there is a dep or POS (later word) match field
	// 2. Second time round where there is no match requirement
	// If this order is not kept the search can get very large before failing anyway
//	for (int iBoth = 0; iBoth < 2; iBoth++) {
//		for (int iaf = 0; iaf < gen_def->access_fields_size(); iaf++) {
//			CaffeGenDef::DataAccess* access = gen_def->mutable_access_fields(iaf);
//			if (	iBoth == 0 
//				&&	!(	access->has_pos_to_match() 
//					||	access->has_dep_type_to_match())) {
//				continue;
//			}
//			if (	iBoth == 1 
//				&&	(	access->has_pos_to_match() 
//					||	access->has_dep_type_to_match())) {
//				continue;
//			}
//			AccessFilterOrderTbl.push_back(make_pair(iaf, true));	
//			var_tbl_idx_map[access->var_name()] = iaf;			
//			access->set_var_idx(iaf);
//
//			for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
//				if (filter_inserted_tbl[idf]) continue;
//				
//				CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
//				CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
//				CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
//				bool b_left_found = false;
//				bool b_right_found = false;
//				for (int i_order = 0;i_order < AccessFilterOrderTbl.size(); i_order++) {
//					pair<int, bool>& order_el = AccessFilterOrderTbl[i_order];
//					if (!order_el.second) { // if not an access entry but rather a filter
//						continue;
//					}
//					const CaffeGenDef::DataAccess& access = gen_def->access_fields(order_el.first);
//					if (!b_left_found && (left_side->var_name_src() == access.var_name())) {
//						b_left_found = true;
//					}
//					if (!b_right_found && (right_side->var_name_src() == access.var_name())) {
//						b_right_found = true;
//					}
//					if (b_left_found && b_right_found) {
//						break;
//					}
//				}
//				if (!b_left_found || !b_right_found) {
//					continue;
//				}
//				AccessFilterOrderTbl.push_back(make_pair(idf, false));
//				filter_inserted_tbl[idf] = true;
//				left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
//				right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
//			}
//
//		}
//		
//	}

	for (int i_access = 0; i_access < access_els_added.size(); i_access++) {
		if (!access_els_added[i_access]) {
			cerr << "Note. Current alogorithm did not access every var\n";
			//return false;			
		}
	}

	
	for (int i_inserted = 0; i_inserted < filter_inserted_tbl.size(); i_inserted++) {
		if (!filter_inserted_tbl[i_inserted]) {
			cerr << "Error in GenDef proto file. Data Filter accesses a var name not in var table\n";
			//return false;			
		}
	}

	return true;
}
bool CGenDef::ModelPrep(bool b_prep_from_src)
{

	if (!::CreateAccessOrder(	AccessFilterOrderTbl, gen_def,  
								var_tbl_idx_map, b_prep_from_src)) {
		return false;
	}
	// each pair in the table: first - idx of rec to access. second: idx of VarTbl to put it into - what??
	
//	for (int idf = 0; idf < gen_def->data_filters_size(); idf++) {
//		CaffeGenDef::DataFilter* filter = gen_def->mutable_data_filters(idf);
//		CaffeGenDef::DataFilterOneSide* left_side = filter->mutable_left_side();
//		left_side->set_var_src_idx(var_tbl_idx_map[left_side->var_name_src()]);
//		CaffeGenDef::DataFilterOneSide* right_side = filter->mutable_right_side();
//		right_side->set_var_src_idx(var_tbl_idx_map[right_side->var_name_src()]);
//		bool b_left_found = false;
//		bool b_right_found = false;
//		vector<pair<int, bool> >::iterator itOrderTbl = AccessFilterOrderTbl.begin();
//		for (;itOrderTbl != AccessFilterOrderTbl.end(); itOrderTbl++) {
//			pair<int, bool>& order_el = *itOrderTbl;
//			if (!order_el.second) {
//				continue;
//			}
//			const CaffeGenDef::DataAccess& access = gen_def->access_fields(order_el.first);
//			if (!b_left_found && (left_side->var_name_src() == access.var_name())) {
//				b_left_found = true;
//			}
//			if (!b_right_found && (right_side->var_name_src() == access.var_name())) {
//				b_right_found = true;
//			}
//			if (b_left_found && b_right_found) {
//				break;
//			}
//		}
//		if (!b_left_found || !b_right_found) {
//			cerr << "Error in GenDef proto file. Data Filter accesses a var name not in var table\n";
//			return false;
//		}
//		itOrderTbl++;
//		AccessFilterOrderTbl.insert(itOrderTbl, make_pair(idf, false));
//	}
//	
	int num_input_vals = 0;
	int num_output_vals = 0;
	
	InstanceCountTbl.resize( gen_def->net_values_size());

	for (int inv = 0; inv < gen_def->net_values_size(); inv++) {
		CaffeGenDef::NetValue* net_value = gen_def->mutable_net_values(inv);
		int var_src_idx = ((net_value->vet() == CaffeGenDef::vetDummy) 
				? 0 : var_tbl_idx_map[net_value->var_name_src()]);
		net_value->set_var_src_idx(var_src_idx);
		int table_idx = -1;
		if (!pGenDefTbls->getTableNameIdx((const string)net_value->vec_table_name(), table_idx)) {
			cerr	<< "Error. Table name given in net value " 
					<< (const string)net_value->vec_table_name() 
					<< " not found in table directory\n.";
		}
		net_value->set_vec_table_idx(table_idx);
		if (net_value->b_input()) {
			num_input_vals++;
			InputTranslateTbl.push_back(make_pair(var_src_idx, table_idx));
		}
		else {
			num_output_vals++;
			OutputTranslateTbl.push_back(make_pair(var_src_idx, table_idx));
		}
		if (net_value->has_max_instances()) {
			map<string, int>* pTbl = TranslateTblPtrs[table_idx];
			InstanceCountTbl[inv] = vector<int> (pTbl->size(), 0);
		}
	}

	NumOutputNodesNeeded = -1;
	OutputSetNumNodes.clear();
	if (gen_def->net_end_type() == CaffeGenDef::END_VALID) {
		if (num_output_vals != 1) {
			cerr << "For end type END_VALID, the number of output_field_translates must be exactly 1.\n";	
			return false;
		}
		NumOutputNodesNeeded = 2;
		OutputSetNumNodes.push_back(NumOutputNodesNeeded);
		
	}
	else if (gen_def->net_end_type() == CaffeGenDef::END_ONE_HOT) {
		if (num_output_vals != 1) {
			cerr << "For end type END_ONE_HOT, the number of output_field_translates must be exactly 1.\n";	
			return false;
		}
		map<string, int>* pTbl = TranslateTblPtrs[OutputTranslateTbl[0].second];
		NumOutputNodesNeeded = pTbl->size();
		OutputSetNumNodes.push_back(NumOutputNodesNeeded);
	}
	else if (gen_def->net_end_type() == CaffeGenDef::END_MULTI_HOT) {
		NumOutputNodesNeeded = 0;
		for (int ift = 0; ift < num_output_vals; ift++) {
			vector<vector<float> >* pTbl = VecTblPtrs[OutputTranslateTbl[ift].second];
			int nodes_this_output = (*pTbl)[0].size();
			NumOutputNodesNeeded += nodes_this_output;
			OutputSetNumNodes.push_back(nodes_this_output);
		}
	}
	return true;
	
#if 0
	// each pair in the table: first - idx of rec to access. second: idx of VarTbl to put it into
	for (int idf = 0; idf < gen_data->data_fields_size(); idf++) {
		const CaffeGenData::DataField& Field = gen_data->data_fields(idf);
		const string& VarName = Field.var_name();
		map<string, int>::iterator itvnm = VarNamesMap.find(VarName);
		if (itvnm != VarNamesMap.end()) {
			cerr << "Error parsing prototxt data. " << VarName << " defined twice\n";
			return false;
		}
		int NextVarNamesMapPos = VarNamesMap.size();
		VarNamesMap[VarName] = NextVarNamesMapPos;
		//int FieldID = GetIdxFromFieldName(Field.field_name());
		FirstAccessFieldsIdx.push_back(make_pair(Field.field_type(), NextVarNamesMapPos));
	}


	for (int idt = 0; idt < gen_data->data_translates_size(); idt++) {
		SDataTranslateEntry DataTranslateEntry ;
		DataTranslateEntry.dtet = dtetRWIDToWord;
		const CaffeGenData::DataTranslate& GenDataTranslateEntry 
			= gen_data->data_translates(idt);
		CaffeGenData::DataTranslateType TranslateType = GenDataTranslateEntry.translate_type(); 
		switch (TranslateType) {
			case CaffeGenData::DATA_TRANSLATE_RWID_TO_WORD :
				DataTranslateEntry.dtet = dtetRWIDToWord;
				break;				
			case CaffeGenData::DATA_TRANSLATE_RWID_TO_COREF:
				DataTranslateEntry.dtet = dtetRWIDToCoref;
				break;
			case CaffeGenData::DATA_TRANSLATE_RWID_TO_RDID:
				DataTranslateEntry.dtet = dtetRWIDToRDID;
				break;
 			case CaffeGenData::DATA_TRANSLATE_RDID_TO_DEP_RWID:
				DataTranslateEntry.dtet = dtetRDIDToDepRWID;
				break;
 			case CaffeGenData::DATA_TRANSLATE_RDID_TO_GOV_RWID :
				DataTranslateEntry.dtet = dtetRDIDToGovRWID;
				break;
			case CaffeGenData::DATA_TRANSLATE_RDID_TO_DEP_NAME:
				DataTranslateEntry.dtet = dtetRDIDToDepName;
				break;
		}
		
		const string& MatchName = GenDataTranslateEntry.match_name();
		map<string, int>::iterator itVarNamesMap = VarNamesMap.find(MatchName);
		if (itVarNamesMap == VarNamesMap.end()) {
			cerr << "Error parsing prototxt data. Translate data match_name " << MatchName << " does not exist.\n";
			return false;
		}
		DataTranslateEntry.VarTblMatchIdx = itVarNamesMap->second;

		CaffeGenData_FieldType FieldID = CaffeGenData::FIELD_TYPE_INVALID;
		if (GenDataTranslateEntry.has_field_type() ) {
			// FieldID = GetIdxFromFieldNif Dame(GenDataTranslateEntry.field_type());
			FieldID = GenDataTranslateEntry.field_type();
		}
		else {
			if (DataTranslateEntry.dtet == dtetRWIDToWord) {
				cerr << "field_name is required if DATA_TRANSLATE_RWID_TO_WORD is set.\n";
				return false;
			}
		}
		DataTranslateEntry.TargetTblOutputIdx = FieldID;
		const string& OutputVarName = GenDataTranslateEntry.var_name();
		map<string, int>::iterator itvnm = VarNamesMap.find(OutputVarName);
		if (itvnm != VarNamesMap.end()) {
			cerr << "Error parsing prototxt data. " << OutputVarName << " defined twice\n";
			return false;
		}
		int VarNamesMapSize = VarNamesMap.size();
		VarNamesMap[OutputVarName] = VarNamesMapSize;
		DataTranslateEntry.VarTblIdx = VarNamesMapSize;

		DataTranslateTbl.push_back(DataTranslateEntry);
	}
	// first field the index of the var to be matched in var tbl, second, the string to match
	for (int idf = 0; idf < gen_data->data_filters_size(); idf++) {
		const CaffeGenData::DataFilter& GenDataFilter = gen_data->data_filters(idf);
		const string& MatchName = GenDataFilter.var_name();
		map<string, int>::iterator itVarNamesMap = VarNamesMap.find(MatchName);
		if (itVarNamesMap == VarNamesMap.end()) {
			cerr << "Error parsing prototxt data. Filter data var_name " << MatchName << " does not exist.\n";
			return false;
		}
		DataFilterTbl.push_back(make_pair(itVarNamesMap->second, GenDataFilter.match_string()));
	}
	
	// for each pair in table: first - idx of VarTbl. second - index of translate tbl
	int OTranslateTableSize = -1;
	for (int iBoth=0; iBoth<2; iBoth++) {
		vector<pair<int, int> >* pTranslateTbl = &InputTranslateTbl;
		vector<bool>* pCanReplaceTbl =  &ICanReplaceTbl;
		int TranslateTableSize = gen_data->input_field_translates_size(); 
		if (iBoth == 1) {
			pTranslateTbl = &OutputTranslateTbl;
			pCanReplaceTbl =  &OCanReplaceTbl;
			TranslateTableSize = gen_data->output_field_translates_size(); 
			OTranslateTableSize = TranslateTableSize;
		}
		for (int ift = 0; ift < TranslateTableSize; ift++) {
			const CaffeGenData::FieldTranslate* pTran;
			if (iBoth == 0) {
				pTran = &(gen_data->input_field_translates(ift));
			}
			else {
				pTran = &(gen_data->output_field_translates(ift));
			}
			int VarNameIdx = -1;
			const string& TableName = pTran->table_name();
			if (pTran->has_var_name()) {
				map<string, int>::iterator itvnm = VarNamesMap.end();
				const string& VarName = pTran->var_name();
				itvnm = VarNamesMap.find(VarName);
				if (itvnm == VarNamesMap.end()) {
					cerr << "Error parsing prototxt data. Translate table field " << VarName << " is not defined previously\n";
					return false;
				}
				VarNameIdx = itvnm->second;
			}
			else if (TableName != "YesNoTbl") {
				cerr << "Error parsing prototxt data. Translate table field must be provided unless the table name is \"YesNoTbl\"\n";
				return false;
			}
			map<string, int>::iterator itttnm = TranslateTblNameMap.find(TableName);
			if (itttnm == TranslateTblNameMap.end()) {
				cerr << "Error parsing prototxt data. Translate table name " << TableName << " does not exist.\n";
				return false;
			}
			pTranslateTbl->push_back(make_pair(VarNameIdx, itttnm->second));
			bool bThisCanReplace = pTran->b_can_replace();
			if (bThisCanReplace) bCanReplace = true;
			pCanReplaceTbl->push_back(bThisCanReplace);
			if (pTran->has_max_instances()) {
				// if this field needs a count of instance, add an array with the size of the table
				NumInstancesTbl[iBoth].push_back(vector<int> (TranslateTblPtrs[itttnm->second]->size(), 0));
				MaxInstancesTbl[iBoth].push_back(pTran->max_instances());
			}
			else {
				// else add empty array
				// empty or full, one must be added to keep indicies in sync
				NumInstancesTbl[iBoth].push_back(vector<int> ());
				MaxInstancesTbl[iBoth].push_back(-1);
			}

		}
	}
	gen_data->net_end_type();
	NumOutputNodesNeeded = -1;
	if (gen_data->net_end_type() == CaffeGenData::END_VALID) {
		if (OTranslateTableSize != 1) {
			cerr << "For end type END_VALID, the number of output_field_translates must be exactly 1.\n";	
			return false;
		}
		NumOutputNodesNeeded = 2;
	}
	else if (gen_data->net_end_type() == CaffeGenData::END_ONE_HOT) {
		if (OTranslateTableSize != 1) {
			cerr << "For end type END_ONE_HOT, the number of output_field_translates must be exactly 1.\n";	
			return false;
		}
		const string& TableName = gen_data->output_field_translates(0).table_name();
		map<string, int>* pTbl = TranslateTblPtrs[TranslateTblNameMap[TableName]];
		NumOutputNodesNeeded = pTbl->size();
	}
	else if (gen_data->net_end_type() == CaffeGenData::END_MULTI_HOT) {
		NumOutputNodesNeeded = 0;
		for (int ift = 0; ift < OTranslateTableSize; ift++) {
			const string& TableName = gen_data->output_field_translates(0).table_name();
			vector<vector<float> >* pTbl = VecTblPtrs[TranslateTblNameMap[TableName]];
			NumOutputNodesNeeded += (*pTbl)[0].size();
		}
	}

	return true;
#endif // 0	
}

bool  CGenDef::setReqTheOneOutput(int& OutputTheOneIdx, bool& bIsPOS) {
	if (OutputTranslateTbl.size() != 1) {
		//cerr << "Error in setReqTheOneOutput: only models with 1 output are supported.\n";
		// only where output is word, can the table size be > 0
		OutputTheOneIdx = OutputTranslateTbl[0].first; // thr first of the pair should be the same for both outputs, it is the index into the var table
		bIsPOS = false;
		return true;
	}
	// The index of the var table acceesed in order to create the single output
	// is the index we are looking for. When an access or translation is made whose
	// output is meant to be written to that entry of the var table, it must have
	// the Avail value of datTheOne. That means that the NN built has the output
	// of the field we are looking to add
	pair<int, int>& iott = OutputTranslateTbl[0];
	OutputTheOneIdx = iott.first;
	int PosNumVecTblIdx = TranslateTblNameMap["POSNumTbl"];
	bIsPOS = (iott.second == PosNumVecTblIdx);
	
	return true;
	
}

void  CGenModelRun::setReqTheOneOutput() { 
	bReqTheOneOutput = true; 
	int RetIdx = -1;
	bool bIsPOS = false;
	if (GenDef.setReqTheOneOutput(RetIdx, bIsPOS)) {
		OutputTheOneIdx = RetIdx;
		bTheOneOutputIsPOS = bIsPOS;
	}
}

bool CGenModelRun::DepMatchTest(const CaffeGenDef::DataAccess& data_access_rec, 
								int isr, int DID, int isrBeyond, bool b_use_avail)
{
	if (isr < 0 || isr >= SentenceRec.size()) {
		cerr << "Serious error!\n";
		return false;
	}
	if (DID < 0 || DID >= SentenceRec[isr].Deps.size()) {
		cerr << "Serious error!\n";
		return false;
	}
	bool ret_avail = true;
	DepRec& rec = SentenceRec[isr].Deps[DID];		
	SDepRecAvail dep_avail(datConst); // default to all there
	if (b_use_avail) {
		dep_avail = SentenceAvailList[isr].Deps[DID];
		if (dep_avail.iDep != datConst) {
			if (	(dep_avail.iDep == datTheOne) 
				&&	bReqTheOneOutput 
				&&	(data_access_rec.var_idx() == OutputTheOneIdx)) {
				if (data_access_rec.has_dep_type_to_match()) {
					ret_avail = false; // datTheOne cannot match on a record that requires a specific dep name
				}
				else {
					ret_avail = true;
				}
			}
			else {
				ret_avail = false;
			}
		}
		else {
			if (	bReqTheOneOutput 
				&&	(data_access_rec.var_idx() == OutputTheOneIdx)) {
				ret_avail = false;
			}
			else {
				ret_avail = true;
			}
		}
	}
	
	if (!ret_avail) {
		return false;
	}
	string dep_name = GenDef.DepNames[rec.iDep];
	if (data_access_rec.has_dep_type_to_match()) {
		if 	(dep_name == data_access_rec.dep_type_to_match()) {
			return true;
		}
		else {
			return false;
		}

	}
	else {
		return true;
	}

	// can't get here
	return false;
}

bool CGenModelRun::WordMatchTest(const CaffeGenDef::DataAccess& data_access_rec, 
								int isr, int WID, int isrBeyond, bool b_use_avail)
{
	if (isr < 0 || isr >= SentenceRec.size()) {
		cerr << "Serious error!\n";
		return false;
	}
	if (WID < 0 || WID >= SentenceRec[isr].OneWordRec.size()) {
		cerr << "Serious error!\n";
		return false;
	}
	bool b_avail = true;
	WordRec& rec = SentenceRec[isr].OneWordRec[WID];		
	SWordRecAvail wrec_avail(datConst); // default to all there
	if (data_access_rec.has_word_to_match()) {
		if (b_use_avail) {
			cerr << "Error! b_use_avail == true is not an option for has_word_to_match yet!\n";
			return false;
		}
		if 	(rec.Word != data_access_rec.word_to_match()) {
			return false;
		}
		return true;
	}
	if (data_access_rec.has_pos_to_match()) {
		if (b_use_avail) {
			wrec_avail = SentenceAvailList[isr].WordRecs[WID];
			if (wrec_avail.POS != datConst) {
				if (	(wrec_avail.POS == datTheOne) 
					&&	bReqTheOneOutput 
					&&	(data_access_rec.var_idx() == OutputTheOneIdx)) {
					if (data_access_rec.has_pos_to_match()) {
						b_avail = false; // datTheOne cannot match on a record that requires a specific dep name
					}
					else {
						b_avail = true;
					}
				}
				else {
					b_avail = false;
				}
			}
			else {
				if (	bReqTheOneOutput 
					&&	(data_access_rec.var_idx() == OutputTheOneIdx)
					&&  bTheOneOutputIsPOS	) {
					b_avail = false;
				}
				else {
					b_avail = true;
				}
			}
		}

		if (!b_avail) {
			return false;
		}
		if 	(rec.POS != data_access_rec.pos_to_match()) {
			return false;
		}

	}

	return true;
}

bool CGenModelRun::FilterTest(	const CaffeGenDef::DataFilter& data_filter,
								vector<VarTblEl> var_tbl) {
	string sl, sr; 
	
	for (int i_both = 0; i_both < 2; i_both++) {
		string s;
		const CaffeGenDef::DataFilterOneSide& one_side 
				= (i_both ? data_filter.right_side() : data_filter.left_side());
		VarTblEl& el = var_tbl[one_side.var_src_idx()];
		int isr = el.R_DorW_ID.first;
		if (el.bDorW) {
			int DID = el.R_DorW_ID.second;
			DepRec& rec = SentenceRec[isr].Deps[DID];		
			switch (one_side.mt()) {
				case CaffeGenDef::mtDEP_RDID:
					s = gen_data_to_string(isr) + ":" + gen_data_to_string(DID);
					break;
				case CaffeGenDef::mtDEP_DEP_RWID:
					s = gen_data_to_string(isr) + ":" + gen_data_to_string((int)rec.Dep);
					break;
				case CaffeGenDef::mtDEP_GOV_RWID:
					s = gen_data_to_string(isr) + ":" + gen_data_to_string((int)rec.Gov);
					break;
				default:
					cerr << "Error. Match type " << one_side.mt() << " requested for dep record\n";
					break;
			}
		}
		else {
			int WID = el.R_DorW_ID.second;
			WordRec& rec = SentenceRec[isr].OneWordRec[WID];		
			switch (one_side.mt()) {
				case CaffeGenDef::mtWORD_RWID:
					s = gen_data_to_string(isr) + ":" + gen_data_to_string(WID);
					break;
				case CaffeGenDef::mtWORD_CORE:
					s = rec.WordCore;
					break;
				case CaffeGenDef::mtWORD_WORD:
					s = rec.Word;
					break;
				case CaffeGenDef::mtWORD_POS:
					s = rec.POS;
					break;
				default:
					cerr << "Error. Match type " << one_side.mt() << " requested for word record\n";
					break;
			}
		}
		if (i_both == 0) {
			sl = s;
		}
		else {
			sr = s;
		}
	}
	if (sl == sr) return true;
	
	return false;
}


bool CGenModelRun::DoRun(bool bContinue) {
	if (!bContinue) {
		DataForVecs.clear();
	}
	
//	vector<vector<vector<float> >* >& VecTblPtrs = GenDef.VecTblPtrs;
//	map<string, int>& TranslateTblNameMap = GenDef.TranslateTblNameMap;
	//vector<string>& DepNames = GenDef.DepNames;
	int YesNoTblIdx = GenDef.YesNoTblIdx;
	int OrdinalVecTblIdx = GenDef.OrdinalVecTblIdx;
	vector<map<string, int>*>& TranslateTblPtrs = GenDef.TranslateTblPtrs;
	vector<pair<int, bool> >& AccessFilterOrderTbl
			= GenDef.getAccessFilterOrderTbl(); // if second is true, first is access message index else filter message

	CaffeGenDef* gen_def = GenDef.getGenDef();

	bool bUseAvail = (SentenceAvailList.size() > 0);

	// create a reverse table for the coref data
	
	vector<vector<int> > CorefRevTbl(SentenceRec.size(), vector<int>());
	{
		
		for (int isrec = 0; isrec < SentenceRec.size(); isrec++) {
			SSentenceRec& srec = SentenceRec[isrec];			
			CorefRevTbl[isrec].resize(srec.OneWordRec.size(), -1);
		}

		for (int icrec = 0; icrec < CorefList.size(); icrec++) {
			CorefRec& crec = CorefList[icrec];			
			CorefRevTbl[crec.SentenceID][crec.HeadWordId] = icrec;
		}
	}
	
	max_dep_gov_list.clear();
	{
		for (int isrec = 0; isrec < SentenceRec.size(); isrec++) {
			SSentenceRec& srec = SentenceRec[isrec];
			max_dep_gov_list.push_back(vector<int>(srec.OneWordRec.size(), 0));
			vector<DepRec>& deps = srec.Deps;
			for (int iidep = 0; iidep < deps.size(); iidep++) {
				int igov = (int)(signed char)deps[iidep].Gov;
				if (igov > 0) {
					max_dep_gov_list[isrec][igov]++;
				}
			}
		}
	}
	// step through records creating data based on translation tables just created
	
//	int NumWordsClean = RevWordMapClean.size();
//	int iNA = NumWordsClean - 1;
//	int iPosNA = PosMap.size() - 1; 
	 
	// not the actual vecs, but the integers that will give the vecs
	// random, IData, valid, OData
//	vector<SDataForVecs > DataForVecs;
	//int NumCandidates = 0;


	for (int isr = 0; isr < SentenceRec.size(); isr++) {
		int num_vars = GenDef.gen_def->access_fields_size();
		vector<vector<VarTblEl> > VarTblAllOptions(1, vector<VarTblEl>(num_vars));
		vector<vector <VarTblEl> > VarTblAllNewOptions;

		SSentenceRec rec = SentenceRec[isr];
		for (int iOrder = 0; iOrder < AccessFilterOrderTbl.size(); iOrder++) {
			VarTblAllNewOptions.clear();
			if (AccessFilterOrderTbl[iOrder].second) { // if data access and not a filter
				const CaffeGenDef::DataAccess& data_access_rec 
						= gen_def->access_fields(AccessFilterOrderTbl[iOrder].first);

				if (data_access_rec.access_type() == CaffeGenDef::ACCESS_TYPE_DEP ) {
					for (int idep = 0; idep < rec.Deps.size(); idep++) { // actually loop through this sentence's recs and the next NumSentenceRecsToSearch  sentences using isrecBeyond
						/* const CaffeGenDef::DataAccess& data_access_rec, 
								int isr, int DID, int isrBeyond, bool b_use_avai */
						if (DepMatchTest(data_access_rec, isr, idep, 0, bUseAvail) ) {
							for (int iOption = 0; iOption < VarTblAllOptions.size(); iOption++) {
								vector <VarTblEl>& VarTblSrc = VarTblAllOptions[iOption];
								// go through checking that this node is not already in the var table
								// in the future an option in DataAccess might less us override this check
								bool b_dup = false;
								for (	int iPrevOrder = 0; iPrevOrder < iOrder; 
										iPrevOrder++) {
//								for (int i_var = 0; i_var < data_access_rec.var_idx(); i_var++) {
									if (!AccessFilterOrderTbl[iPrevOrder].second) {
										continue;
									}
									// i_var is the index into the var tbl 
									int i_var = AccessFilterOrderTbl[iPrevOrder].first;
									if (	VarTblSrc[i_var].bDorW 
										&&	VarTblSrc[i_var].R_DorW_ID.first == isr 
										&&	VarTblSrc[i_var].R_DorW_ID.second == idep) {
										b_dup = true;
										break;
									}
								}
								if (b_dup) {
									continue;
								}
								VarTblAllNewOptions.push_back(VarTblSrc);
								vector <VarTblEl>& VarTblNew = VarTblAllNewOptions.back();
								VarTblNew[data_access_rec.var_idx()] = (VarTblEl(true, isr, idep));
							}
						}
					}
				}	
				else if (data_access_rec.access_type() == CaffeGenDef::ACCESS_TYPE_WORD ) {
					// for AccessType WordRec and any other in the same enum 
					// do the same as for deprec
					for (int iwrec = 0; iwrec < rec.OneWordRec.size(); iwrec++) { 					
						if (WordMatchTest(data_access_rec, isr, iwrec, 0, bUseAvail) ) {
							for (int iOption = 0; iOption < VarTblAllOptions.size(); iOption++) {
								vector <VarTblEl>& VarTblSrc = VarTblAllOptions[iOption];
								// go through checking that this node is not already in the var table
								// in the future an option in DataAccess might less us override this check
								bool b_dup = false;
								for (	int iPrevOrder = 0; iPrevOrder < iOrder; 
										iPrevOrder++) {
//								for (int i_var = 0; i_var < data_access_rec.var_idx(); i_var++) {
									if (!AccessFilterOrderTbl[iPrevOrder].second) {
										continue;
									}
									// i_var is the index into the var tbl 
									int i_var = AccessFilterOrderTbl[iPrevOrder].first;
									if (	!VarTblSrc[i_var].bDorW 
										&&	VarTblSrc[i_var].R_DorW_ID.first == isr 
										&&	VarTblSrc[i_var].R_DorW_ID.second == iwrec) {
										b_dup = true;
										break;
									}
								}
								if (b_dup) {
									continue;
								}
								VarTblAllNewOptions.push_back(VarTblSrc);
								vector <VarTblEl>& VarTblNew = VarTblAllNewOptions.back();
								VarTblNew[data_access_rec.var_idx()] = (VarTblEl(false, isr, iwrec));
							}
						}
					}
					
				}
				else {
					cerr << "CGenModelRun::DoRun Error: Unknown access type.\n";
					return false;
				}
			}	
			else { // if (!AccessFilterOrderTbl[iOrder].second) { 
				const CaffeGenDef::DataFilter& filter
						= GenDef.gen_def->data_filters(AccessFilterOrderTbl[iOrder].first);
				for (int iOption = 0; iOption < VarTblAllOptions.size(); iOption++) {
					vector <VarTblEl>& VarTblSrc = VarTblAllOptions[iOption];
					if (FilterTest(filter, VarTblSrc)) {
					//if (access(DataFilterRec.LeftSide, VarTblAllOptions[iOption]) == access(DataFilterRec.RightSide, VarTblAllOptions[iOption])) {
						
						VarTblAllNewOptions.push_back(VarTblSrc);
					}
				}
			}

			VarTblAllOptions.clear();
			VarTblAllOptions = VarTblAllNewOptions;
			if (VarTblAllOptions.size() == 0) {
				break;
			}
			if (VarTblAllOptions.size() > 5000) {
				cerr	<< "Do run options getting big. Now at " << VarTblAllOptions.size() 
						<< ". at isr: " << isr <<  " of " << SentenceRec.size() 
						<< ". index of AccessOrderTbl is at " << iOrder	<< ".\n";
			}
		}	

		for (int i_option = 0; i_option < VarTblAllOptions.size(); i_option++) {
			vector<int> IData;
			vector<int> OData;
			string s;
			vector<VarTblEl>& var_tbl = VarTblAllOptions[i_option];
			bool b_all_found = true;
			bool b_output_eval_done = false;
			for (int inv = 0; inv < GenDef.gen_def->net_values_size(); inv++) {
				const CaffeGenDef::NetValue& val = GenDef.gen_def->net_values(inv);
				vector<int>& Data =  (val.b_input() ? IData : OData);
				if (!val.b_input() && bReqTheOneOutput) {
					// first make sure that even if we are req the one output,
					// we only do this evaluation on one of the outputs
					// this is because word outputs have two outputs, both on the dame var idx
					if (b_output_eval_done) continue;
					b_output_eval_done = true;
					VarTblEl& el = var_tbl[val.var_src_idx()];
					int isr = el.R_DorW_ID.first;
					if (el.bDorW) {
						int DID = el.R_DorW_ID.second;
						SDepRecAvail& dep_avail = SentenceAvailList[isr].Deps[DID];
						if (dep_avail.iDep != datTheOne) {
							b_all_found = false;
							break;
						}
					}
					else {
						int WID = el.R_DorW_ID.second;
						SWordRecAvail& wrec_avail = SentenceAvailList[isr].WordRecs[WID];
						DataAvailType dat;
						switch (val.vet()) {
							case CaffeGenDef::vetPOS:
								dat = wrec_avail.POS;
								break;
							case CaffeGenDef::vetNumDepGovs: // adding this option to vetWord because they are supposed to come together
							case CaffeGenDef::vetWord:
								dat = wrec_avail.Word;
								break;
							case CaffeGenDef::vetWordCore:
								dat = wrec_avail.WordCore;
								break;
							case CaffeGenDef::vetDummy:
								dat = datConst;
								break;
							default:
								cerr << "Error. Requesting a dep field from a word rec.\n";
								return false;

						}
						if (dat != datTheOne) {
							b_all_found = false;
							break;
						}
				}
					
					LOG(INFO) << "Found the one!\n";
					continue;
				}
				if (val.vec_table_idx() == YesNoTblIdx) {
					Data.push_back(1);
				}
				else if (val.vec_table_idx() == OrdinalVecTblIdx) {
					VarTblEl& el = var_tbl[val.var_src_idx()];
					int isr = el.R_DorW_ID.first;
					int WID = el.R_DorW_ID.second;
					if (el.bDorW || (val.vet() != CaffeGenDef::vetNumDepGovs)) {
						cerr << "Error: vet other than word value access of vetNumDepGovs requested table OrdinaVec.\n";
						break;
					}
					int num_govs = max_dep_gov_list[isr][WID];
					if (num_govs >= c_ordinal_tbl_max) {
						num_govs = c_ordinal_tbl_max - 1;
					}
					Data.push_back(num_govs);
				}

				else {
					VarTblEl& el = var_tbl[val.var_src_idx()];
					int isr = el.R_DorW_ID.first;
					if (el.bDorW) {
						int DID = el.R_DorW_ID.second;
						if (!bUseAvail || val.b_input()) {
							if ((DID < 0) || (DID >= SentenceRec[isr].Deps.size())) {
								cerr << "Error: Unxepected access of dep record out of bounds.\n";
								return false;
							}
							DepRec& rec = SentenceRec[isr].Deps[DID];		
							switch (val.vet()) {
								case CaffeGenDef::vetDepName:
									s = GenDef.DepNames[rec.iDep];
									break;
								case CaffeGenDef::vetDummy:
									s = "1";
									break;
								default:
									cerr << "Error. Requesting a word field from a dep rec.\n";
									return false;

							}
						}
					}
					else { // if (word)
						int WID = el.R_DorW_ID.second;
						if (!bUseAvail || val.b_input()) {
							if ((WID < 0) || (WID >= SentenceRec[isr].OneWordRec.size())) {
								cerr << "Error: Unxepected access of word record out of bounds.\n";
								return false;
							}
							WordRec& rec = SentenceRec[isr].OneWordRec[WID];		
							switch (val.vet()) {
								case CaffeGenDef::vetPOS:
									s = rec.POS;
									break;
								case CaffeGenDef::vetWord:
									s = rec.Word;
									break;
								case CaffeGenDef::vetWordCore: 
									s = rec.WordCore;
									break;
								case CaffeGenDef::vetDummy:
									s = "1";
									break;
								default:
									cerr << "Error. Requesting a dep field from a word rec.\n";
									return false;

							}
						}
					}
					map<string, int>* mapp = TranslateTblPtrs[val.vec_table_idx()];
					if (mapp == NULL) {
						cerr << "Error: NULL translate table access.\n";
						break;
					}
					map<string, int>::iterator itm = mapp->find(s);

					if (itm == mapp->end()) {
						b_all_found = false;
						break;
					}
					if (val.has_max_instances()) {
						if (GenDef.InstanceCountTbl[inv][itm->second] >= val.max_instances()) {
							b_all_found = false;
							break;
						}
						GenDef.InstanceCountTbl[inv][itm->second]++;
					}

					Data.push_back(itm->second);
				}
			}
			if (!b_all_found) {
				continue;
			}
			DataForVecs.push_back(SDataForVecs(rand(), IData, true, OData));
		}
	}
		

#if 0	
	vector<map<string, int>*>& TranslateTblPtrs = GenDef.TranslateTblPtrs;
	CaffeGenData* gen_data = GenDef.getGenData();

	//	int * po = &(OutputTranslateTbl[0].second);
	vector<vector<string> > VarTblsForGo; 

	bool bContinueWithMainLoop = true;
	int isr = 0;
	while (bContinueWithMainLoop) {
		VarTblsForGo.clear();
		if (gen_data->iterate_type() == CaffeGenData::ITERATE_DEP) {
			const int cNumSentenceRecsPerDepGo = 1;
			for (int iisr = 0; iisr < cNumSentenceRecsPerDepGo; iisr++, isr++) {
				if (isr >= SentenceRec.size()) {
					bContinueWithMainLoop = false;
					break;
				}
				SSentenceRec Rec = SentenceRec[isr];
//				if (Rec.Deps.size() < cMinRealLength) {
//					continue;
//				}
				vector<DepRec>& DepRecs = Rec.Deps;
				for (int idrec = 0; idrec < DepRecs.size(); idrec++) { 
//					DepRec& drec = DepRecs[idrec];
					vector<string> VarTbl = vector<string> (GenDef.VarNamesMap.size()); 
					bool bAllFieldsFound = true;
					for (int ia = 0; ia < GenDef.FirstAccessFieldsIdx.size(); ia++ ) {
						pair<CaffeGenData_FieldType, int>& access = GenDef.FirstAccessFieldsIdx[ia];
						DataAvailType RetAvail = datConst;
						if (access.second >= VarTbl.size()) {
							cerr << "Serious error!\n";
							return false;
						}
						VarTbl[access.second] 
								= GetDepRecField(	isr, idrec, access.first, 
													RetAvail, bUseAvail);
						if (RetAvail != datConst) {
							if (	(RetAvail == datTheOne) 
								&&	bReqTheOneOutput 
								&&	(access.second == OutputTheOneIdx)) {
							}
							else {
								bAllFieldsFound = false;
								break;
							}
						}
						else if (	bReqTheOneOutput 
								&&	(access.second == OutputTheOneIdx)) {
							bAllFieldsFound = false;
							break;
						}
					}
					if (bAllFieldsFound) {
						VarTblsForGo.push_back(VarTbl);
					}
					NumCandidates++;
				}
			}
		}	
		else if (gen_data->iterate_type() == CaffeGenData::ITERATE_WORD) {
			const int cNumSentenceRecsPerGo = 1;
			for (int iisr = 0; iisr < cNumSentenceRecsPerGo; iisr++, isr++) {
				if (isr >= SentenceRec.size()) {
					bContinueWithMainLoop = false;
					break;
				}
				SSentenceRec Rec = SentenceRec[isr];
				vector<WordRec>& WordRecs = Rec.OneWordRec;
//				if (Rec.OneWordRec.size() < cMinRealLength) {
//					continue;
//				}
				 for (int iwrec = 0; iwrec < WordRecs.size(); iwrec++) {
					VarTblsForGo.push_back(vector<string> (GenDef.VarNamesMap.size()));
					vector<string>& VarTbl = VarTblsForGo.back(); 
					bool bAllFieldsFound = true;
					for (int ia = 0; ia < GenDef.FirstAccessFieldsIdx.size(); ia++ ) {
						pair<CaffeGenData_FieldType, int>& access = GenDef.FirstAccessFieldsIdx[ia];
						DataAvailType RetAvail = datConst;
						if (access.second >= VarTbl.size()) {
							cerr << "Serious error!\n";
							return false;
						}
						VarTbl[access.second] 
								= GetRecFieldByIdx(	isr, iwrec, access.first, 
													RetAvail, bUseAvail);
						if (RetAvail != datConst) {
							if (	(RetAvail == datTheOne) 
								&&	bReqTheOneOutput 
								&&	(access.second == OutputTheOneIdx)) {
							}
							else {
								bAllFieldsFound = false;
								break;
							}
						}
						else if (	bReqTheOneOutput 
								&&	(access.second == OutputTheOneIdx)) {
							bAllFieldsFound = false;
							break;
						}
					}
					if (!bAllFieldsFound) {
						continue;
					}
					NumCandidates++;
				}
			}
		}
		
		if (GenDef.DataTranslateTbl.size() > 0) {
			vector<vector<string> > VarTblsForGoTranslated; 
			for (int ivt = 0; ivt < VarTblsForGo.size(); ivt++) {
				vector<string>& VarTbl = VarTblsForGo[ivt];
				bool bAllGood = true;
				for (int idte = 0; idte < GenDef.DataTranslateTbl.size(); idte++) {
					SDataTranslateEntry& DataTranslateEntry =  GenDef.DataTranslateTbl[idte];
					string VarNameForMatch = VarTbl[DataTranslateEntry.VarTblMatchIdx];
					// Phase 1. Access the current record
					int WID = -1; // Assumes the translation was from WID, the id of the record in the WordRecs Tbl
					int RecID = -1;
					int DID = -1;
					bool bGov = true;
					if (	DataTranslateEntry.dtet == dtetRWIDToWord 
						||	DataTranslateEntry.dtet == dtetRWIDToCoref
						||	DataTranslateEntry.dtet == dtetRWIDToRDID) {
						int ColonPos = VarNameForMatch.find(":");
						if (ColonPos == -1) {
							cerr << "Error: RWID result is not formatted with a \":\". WID must be formatted as RecID::idx where idx is the index of the word in the sentence\n";
							// this is really a parsing error not a data error so we return
							return false;
						}
						RecID = atoi(VarNameForMatch.substr(0, ColonPos).c_str());
						if (RecID >= SentenceRec.size()) {
							cerr << "Error: Record number stored from dep result is greater than the number of SentenceRecs\n";
							bAllGood = false;
							break;
						}
						WID = atoi(VarNameForMatch.substr(ColonPos+1).c_str());
					}
					if (	( DataTranslateEntry.dtet == dtetRDIDToDepRWID) 
						||	(DataTranslateEntry.dtet == dtetRDIDToGovRWID) 
						||	(DataTranslateEntry.dtet == dtetRDIDToDepName) ) {
						int ColonPos = VarNameForMatch.find(":");
						if (ColonPos == -1) {
							cerr << "Error: RDID result is not formatted with even a single \":\". DID must be formatted as RecID::idx:g/v where idx is the index of the dep rec in the sentence\n";
							// this is really a parsing error not a data error so we return
							return false;
						}
						RecID = atoi(VarNameForMatch.substr(0, ColonPos).c_str());
						VarNameForMatch = VarNameForMatch.substr(ColonPos+1);
						ColonPos = VarNameForMatch.find(":");
						if (ColonPos == -1) {
							cerr << "Error: RDID result is not formatted with its second \":\". DID must be formatted as RecID::idx:g/v where idx is the index of the dep rec in the sentence\n";
							// this is really a parsing error not a data error so we return
							return false;
						}
						DID = atoi(VarNameForMatch.substr(0, ColonPos).c_str());
						string sDorG = VarNameForMatch.substr(ColonPos+1);
						if (sDorG == "d") {
							bGov = false;
						}
					}
					// Phase 2. Apply the link to another set of records
					if (	DataTranslateEntry.dtet == dtetRWIDToWord) {
						SSentenceRec SRec = SentenceRec[RecID];
						vector<WordRec>& WordRecs = SRec.OneWordRec;
						// for the following, should be using DataTranslateEntry.TargetTblMatchIdx 
						// and then loop through the records, till the return value matches WID
						// but for dep records, they are sorted by the index anyway.
						if (WID == 255) {
							// ..but 255 is the special case of root
							VarTbl[DataTranslateEntry.VarTblIdx] = "<na>";
						}
						else {
							if (WID >= WordRecs.size()) {
								//cerr << "Error: Word number stored from dep result is greater than the number of words in sentence\n";
								bAllGood = false;
								break;
							}
							DataAvailType RetAvail = datConst;
							VarTbl[DataTranslateEntry.VarTblIdx] 
									= GetRecFieldByIdx(	RecID, WID, 
														DataTranslateEntry.TargetTblOutputIdx, 
														RetAvail, bUseAvail );
							if (RetAvail != datConst) {
								if (	(RetAvail == datTheOne) 
									&&	bReqTheOneOutput 
									&&	(DataTranslateEntry.VarTblIdx == OutputTheOneIdx)) {
								}
								else {
									bAllGood = false;
									break;
								}
							}
							else if (	bReqTheOneOutput 
									&&	(DataTranslateEntry.VarTblIdx == OutputTheOneIdx)) {
								bAllGood = false;
								break;
							}
						}
					}
					else if (	DataTranslateEntry.dtet == dtetRWIDToRDID) {
						SSentenceRec SRec = SentenceRec[RecID];
						vector<DepRec>& DepRecs = SRec.Deps;
						for (int idid = 0; idid < DepRecs.size() ; idid++) {
							DepRec& drec = DepRecs[idid];
							DataAvailType RetAvail = datConst;
							if (drec.Gov == WID) { 
#pragma message("replace setting the last time by addding a new rVarTbl each time")								
								VarTbl[DataTranslateEntry.VarTblIdx] 
										= GetDepRecField(	RecID, idid, 
															CaffeGenData::FIELD_TYPE_GOV_RDID, 
															RetAvail, bUseAvail);

							}
							if (drec.Dep == WID) { 
								VarTbl[DataTranslateEntry.VarTblIdx] 
										= GetDepRecField(	RecID, idid, 
															CaffeGenData::FIELD_TYPE_DEP_RDID, 
															RetAvail, bUseAvail);

							}
							
							if (RetAvail != datConst) {
								if (	(RetAvail == datTheOne) 
									&&	bReqTheOneOutput 
									&&	(DataTranslateEntry.VarTblIdx == OutputTheOneIdx)) {
								}
								else {
									bAllGood = false;
									break;
								}
							}
							else if (	bReqTheOneOutput 
									&&	(DataTranslateEntry.VarTblIdx == OutputTheOneIdx)) {
								bAllGood = false;
								break;
							}
							
						}
						
						
					}
					else if (	DataTranslateEntry.dtet == dtetRWIDToCoref) {
						if (	(RecID < 0) || (WID < 0 )
							||	(CorefRevTbl.size() <= RecID) || (CorefRevTbl[RecID].size() <= WID)) {
							bAllGood = false;
						}
						else if (CorefRevTbl[RecID][WID] == -1) {
							bAllGood = false;
						}
						else {
							int icrec = CorefRevTbl[RecID][WID];
							if (icrec < CorefList.size() && icrec > 0) {
								CorefRec& crec = CorefList[icrec];
								CorefRec& crecPrev = CorefList[icrec-1];
								if (crec.GovID != crecPrev.GovID) {
									bAllGood = false;
								}
								else {
//									CorefRec& crecGov = CorefList[crec.GovID];
//									cerr << "Found coref of gov: "
//										<< SentenceRec[crecGov.SentenceID].OneWordRec[crecGov.HeadWordId].Word 
//										<< " from "
//										<< SentenceRec[crecPrev.SentenceID].OneWordRec[crecPrev.HeadWordId].Word 
//										<< " and "
//										<< SentenceRec[crec.SentenceID].OneWordRec[crec.HeadWordId].Word 
//										<< endl;
									VarTbl[DataTranslateEntry.VarTblIdx] 
										=		(gen_data_to_string(crecPrev.SentenceID) + ":" 
											+	gen_data_to_string(crecPrev.HeadWordId)) ;
								}
							}
							else {
								bAllGood = false;
							}
						}
					}
					else if (	(DataTranslateEntry.dtet == dtetRDIDToDepRWID) 
							||	(DataTranslateEntry.dtet == dtetRDIDToGovRWID) 
							||	(DataTranslateEntry.dtet == dtetRDIDToDepName) ){
						if ((RecID < 0) ||  (RecID > SentenceRec.size())) {
							bAllGood = false;
						}
						else {
							SSentenceRec SRec = SentenceRec[RecID];
							vector<DepRec>& DepRecs = SRec.Deps;
							if ((DID < 0) ||  (DID > DepRecs.size())) {
								bAllGood = false;
							}
							else {
								DepRec& drec = DepRecs[DID];
								switch (DataTranslateEntry.dtet) {
									case dtetRDIDToDepRWID:
										VarTbl[DataTranslateEntry.VarTblIdx] 
											=		(gen_data_to_string(RecID) + ":" 
												+	gen_data_to_string((int)(drec.Dep))) ;
										break;
									case dtetRDIDToGovRWID:
										VarTbl[DataTranslateEntry.VarTblIdx] 
											=		(gen_data_to_string(RecID) + ":" 
												+	gen_data_to_string((int)(drec.Gov))) ;
										break;
									case dtetRDIDToDepName:
										VarTbl[DataTranslateEntry.VarTblIdx] =
											(DepNames[drec.iDep] + (bGov ? ":g" : ":d"));
										break;
									default:
										cerr << "Error on translate data entry.\n";
										break;
								}
								
							}
						}
					}

				}
				//add all the other dtet options
				if (bAllGood) {
					VarTblsForGoTranslated.push_back(VarTbl);
				}

			}
			VarTblsForGo.clear();
			VarTblsForGo = VarTblsForGoTranslated;
		}
		if (GenDef.DataFilterTbl.size() > 0) {
			vector<vector<string> > VarTblsForGoTranslated; 
			for (int ivt = 0; ivt < VarTblsForGo.size(); ivt++) {
				vector<string>& VarTbl = VarTblsForGo[ivt];
				for (int idf = 0; idf < GenDef.DataFilterTbl.size(); idf++) {
					pair<int, string>& DataFilter =  GenDef.DataFilterTbl[idf];
					string FieldVal = VarTbl[DataFilter.first];
					if (FieldVal == DataFilter.second) {
						VarTblsForGoTranslated.push_back(VarTbl);
					}
				}
			}
			
			VarTblsForGo.clear();
			VarTblsForGo = VarTblsForGoTranslated;
		}
		for (int ivt = 0; ivt < VarTblsForGo.size(); ivt++) {
			vector<string>& VarTbl = VarTblsForGo[ivt];
			bool bAllFieldsFound = true;
			vector<int> IData;
			vector<int> OData;
			for (int iBoth=0; bAllFieldsFound && iBoth<2; iBoth++) {
				vector<pair<int, int> >* pTranslateTbl = &GenDef.InputTranslateTbl;
				vector<int>* pData =  &IData;
				if (iBoth == 1) {
					pData = &OData;
					pTranslateTbl = &GenDef.OutputTranslateTbl;
					if (bReqTheOneOutput) {
						cerr << "Found the one!\n";
						continue;
					}
				}
				
				for (int iitt = 0;iitt<pTranslateTbl->size();iitt++) {
					pair<int, int>& itt = (*pTranslateTbl)[iitt];
					
					if (itt.second == YesNoTblIdx) {
						pData->push_back(1);
					}
					else {
						string& FirstAccessVal = VarTbl[itt.first];
						map<string, int>* mapp = TranslateTblPtrs[itt.second];
						map<string, int>::iterator itm = mapp->find(FirstAccessVal);

						if (itm == mapp->end()) {
							bAllFieldsFound = false;
							break;
						}
						if (GenDef.MaxInstancesTbl[iBoth][iitt] >= 0) {
							if (GenDef.NumInstancesTbl[iBoth][iitt][itm->second] >= GenDef.MaxInstancesTbl[iBoth][iitt]) {
								bAllFieldsFound = false;
								break;
							}
							GenDef.NumInstancesTbl[iBoth][iitt][itm->second]++;
						}
						pData->push_back(itm->second);
					}
					
				}
			}
			if (!bAllFieldsFound) {
				continue;
			}
			DataForVecs.push_back(SDataForVecs(rand(), IData, true, OData));
			if (GenDef.bCanReplace) {
				vector<int> IDataRepl(IData.size());
				vector<int> ODataRepl(OData.size());
				for (int iBoth=0; iBoth<2; iBoth++) {
					vector<int>* pDataRepl = &IDataRepl;
					vector<pair<int, int> >* pTranslateTbl = &GenDef.InputTranslateTbl;
					vector<int>* pData = &IData;
					vector<bool>* pCanReplaceTbl =  &GenDef.ICanReplaceTbl;
					if (iBoth == 1) {
						pDataRepl = &ODataRepl;
						pTranslateTbl = &GenDef.OutputTranslateTbl;
						pData = &OData; 
						pCanReplaceTbl =  &GenDef.OCanReplaceTbl;
					}
					for (int iitt = 0;iitt<pTranslateTbl->size();iitt++) {
						pair<int, int>& itt = (*pTranslateTbl)[iitt];
						if (itt.second == YesNoTblIdx) {
							(*pDataRepl)[iitt] = 0;
						}
						else {
							map<string, int>* mapp = TranslateTblPtrs[itt.second];
							if ((*pCanReplaceTbl)[iitt]) {
								int ReplaceVal = rand() % mapp->size();
								(*pDataRepl)[iitt] = ReplaceVal;
							}
							else {
								(*pDataRepl)[iitt] = (*pData)[iitt];
							}
						}
					}
				}
				DataForVecs.push_back(SDataForVecs(rand(), IDataRepl, false, ODataRepl));
			}
		}
	} // end loop over sentence recs

//		{
//			// this goes outside Rec Loop
//			//something is wrong here. Too few records get here. Count the candidates and see why this is happenining
//			sort(DataForVecs.begin(), DataForVecs.end());
//			for (auto idata : DataForVecs) {
//				for (int iBoth=0; iBoth<2; iBoth++) {
//					auto pTranslateTbl = &InputTranslateTbl;
//					auto pData = &(get<1>(idata));
//					if (iBoth == 1) {
//						pTranslateTbl = &OutputTranslateTbl;
//						pData = &(get<3>(idata));
//					}
//					int ii = 0;
//					for (auto itt : (*pTranslateTbl)) {
//						vector<float>& vec = (*VecTblPtrs[itt.second])[(*pData)[ii]];
//						ii++;
//					}
//				}
//				bool bValid = get<2>(idata);
//			}
//		}

	cerr << "Num candidates " << NumCandidates << " resulting in " << DataForVecs.size() << endl;
	// use the random int in the first field to sort
	sort(DataForVecs.begin(), DataForVecs.end(), SDataForVecs::SortFn);

//	int NumRecords = DataForVecs.size() / 2;
//	int NumLabelVals = 0;
//	int NumItemsPerRec = 0;
//	for (int iitt = 0;iitt<InputTranslateTbl.size();iitt++) {
//		pair<int, int>& itt = InputTranslateTbl[iitt];
//		NumItemsPerRec += (*VecTblPtrs[itt.second])[0].size();
//	}
////	for (auto ott : OutputTranslateTbl) {
//	for (int iott = 0;iott<OutputTranslateTbl.size();iott++) {
//		pair<int, int>& ott = OutputTranslateTbl[iott];
//		NumLabelVals += (*VecTblPtrs[ott.second])[0].size();
//	}

#endif // 0
	return true;
}

 