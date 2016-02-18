// GenDef.cpp : 
// Calls the functions needed to parse and use GenData data 
//


 
#include <fcntl.h>
#include <string>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <vector>
#include <map>
//#include "stdafx.h"

//#include "MascReader.h"
#include "H5Cpp.h"

//#include "/dev/caffe/include/caffe/proto/GenData.pb.h"
#include "caffe/proto/GenData.pb.h"
#include "caffe/proto/GenDataTbls.pb.h"
#include "caffe/GenData.hpp"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif


#ifdef  _MSC_VER
#pragma warning(disable : 4503)
#endif
	
using namespace std;	

template <typename T>
string gen_data_to_string ( T Number )
{
	ostringstream ss;
	ss << Number;
	return ss.str();
}

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

//DataAvailType GetDepRecAvail(SDepRecAvail& DepRec, CaffeGenData_FieldType FieldID) 
//{
//	if (DepRec.Dep == datConst && DepRec.Gov == datConst && DepRec.iDep == datConst) {
//		return datConst;
//	}
//	
//	switch(FieldID) {
//		case CaffeGenData::FIELD_TYPE_DEP_NAME: 
//			return (DepRec.iDep) ;
//		case CaffeGenData::FIELD_TYPE_GOV_WID: 
//		case CaffeGenData::FIELD_TYPE_GOV_RWID: 
//			return (DepRec.Gov) ;
//		case CaffeGenData::FIELD_TYPE_DEP_WID: 
//		case CaffeGenData::FIELD_TYPE_DEP_RWID: 
//			return (DepRec.Dep) ;
//		case CaffeGenData::FIELD_TYPE_GOV_RDID:
//		case CaffeGenData::FIELD_TYPE_DEP_RDID:
//		case CaffeGenData::FIELD_TYPE_DEP_NAME_G:
//		case CaffeGenData::FIELD_TYPE_DEP_NAME_D:
//		default:
//			cerr << "Error: TBD as yet. Don't know what to do in these cases.\n";
//			return (datNotSetTooFar) ;
//	}
//	return (datNotSetTooFar) ;
//}

string	CGenModelRun::GetDepRecField(	int SRecID, int DID, 
										CaffeGenData_FieldType FieldID, 
										DataAvailType& RetAvail, bool bUseAvail) {
	RetAvail = datConst;
	DepRec& rec = SentenceRec[SRecID].Deps[DID];		
	SDepRecAvail DepAvail(datConst); // default to all there
	if (bUseAvail) {
		DepAvail = SentenceAvailList[SRecID].Deps[DID];
		switch(FieldID) {
			case CaffeGenData::FIELD_TYPE_DEP_NAME: 
				RetAvail = DepAvail.iDep;
				break;
			case CaffeGenData::FIELD_TYPE_GOV_WID: 
			case CaffeGenData::FIELD_TYPE_GOV_RWID: 
				RetAvail = DepAvail.Gov;
				break;
			case CaffeGenData::FIELD_TYPE_DEP_WID: 
			case CaffeGenData::FIELD_TYPE_DEP_RWID: 
				RetAvail = DepAvail.Dep;
				break;
			case CaffeGenData::FIELD_TYPE_GOV_RDID:
			case CaffeGenData::FIELD_TYPE_DEP_RDID:
			case CaffeGenData::FIELD_TYPE_DEP_NAME_G:
			case CaffeGenData::FIELD_TYPE_DEP_NAME_D:
			default:
				cerr << "Error: TBD as yet. Don't know what to do in these cases.\n";
				RetAvail = datInvalid ;
		}
		
		if (RetAvail != datConst) {
			return string();
		}
	}
	
	switch(FieldID) {
		case CaffeGenData::FIELD_TYPE_DEP_NAME: 
			return (GenDef.DepNames[rec.iDep]) ;
		case CaffeGenData::FIELD_TYPE_GOV_WID: 
			return (gen_data_to_string((int)rec.Gov)) ;
		case CaffeGenData::FIELD_TYPE_DEP_WID: 
			return (gen_data_to_string((int)rec.Dep)) ; 
		case CaffeGenData::FIELD_TYPE_GOV_RWID: 
			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string((int)rec.Gov)) ;
		case CaffeGenData::FIELD_TYPE_DEP_RWID: 
			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string((int)rec.Dep)) ; 
		case CaffeGenData::FIELD_TYPE_GOV_RDID:
			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string(DID) + ":g");
		case CaffeGenData::FIELD_TYPE_DEP_RDID:
			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string(DID) + ":d");
		case CaffeGenData::FIELD_TYPE_DEP_NAME_G:
			return (GenDef.DepNames[rec.iDep] + ":g") ;
		case CaffeGenData::FIELD_TYPE_DEP_NAME_D:
			return (GenDef.DepNames[rec.iDep] + ":d") ;
		default:
			RetAvail = datInvalid;
			break;
	}
	return string();
}

//string GetDepRecFieldByIdx(	int SRecID, int DID, vector<string>& DepNames, DepRec& rec, 
//							CaffeGenData_FieldType FieldID, bool& bRetValid)
//{
//	bRetValid = true;
//	switch(FieldID) {
//		case CaffeGenData::FIELD_TYPE_DEP_NAME: 
//			return (DepNames[rec.iDep]) ;
//		case CaffeGenData::FIELD_TYPE_GOV_WID: 
//			return (gen_data_to_string((int)rec.Gov)) ;
//		case CaffeGenData::FIELD_TYPE_DEP_WID: 
//			return (gen_data_to_string((int)rec.Dep)) ; 
//		case CaffeGenData::FIELD_TYPE_GOV_RWID: 
//			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string((int)rec.Gov)) ;
//		case CaffeGenData::FIELD_TYPE_DEP_RWID: 
//			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string((int)rec.Dep)) ; 
//		case CaffeGenData::FIELD_TYPE_GOV_RDID:
//			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string(DID) + ":g");
//		case CaffeGenData::FIELD_TYPE_DEP_RDID:
//			return (gen_data_to_string(SRecID) + ":" + gen_data_to_string(DID) + ":d");
//		case CaffeGenData::FIELD_TYPE_DEP_NAME_G:
//			return (DepNames[rec.iDep] + ":g") ;
//		case CaffeGenData::FIELD_TYPE_DEP_NAME_D:
//			return (DepNames[rec.iDep] + ":d") ;
//		default:
//			bRetValid = false;
//			break;
//	}
//	return string();
//}
	
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
			cerr << "GenDataModelInit config file Parse failed for " << sModelProtoName << endl;
			bGood = false;
		}
		delete proto_input;
		if (!bGood) {
			return;
		}
	}
	else {
		cerr << "Error in GenDataModelInit. Model file " << sModelProtoName << " not found.\n";
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
	delete protoc_data;
	bInitDone = true;

}



bool CGenDef::ModelInit(string sModelProtoName ) 
{
	//CaffeFnInitData * InitData = new CaffeFnInitData;

//	CaffeFnHandle = NULL;
//	CaffeFnOutHandle = NULL;
//	TranslateTblPtrs.clear();
//	VecTblPtrs.clear();
//	TranslateTblNameMap.clear();
	InputTranslateTbl.clear();
	OutputTranslateTbl.clear();
	FirstAccessFieldsIdx.clear();
	VarNamesMap.clear();
	DataTranslateTbl.clear();
	DataFilterTbl.clear();
	ICanReplaceTbl.clear();
	OCanReplaceTbl.clear();
	NumInstancesTbl[0].clear();
	NumInstancesTbl[1].clear();
	MaxInstancesTbl[0].clear();
	MaxInstancesTbl[1].clear();
	bCanReplace = false;
	
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

//	const string& CoreDir = gen_data->files_core_dir();
//	string H5TrainListFileName = CoreDir + "data/train_list.txt";
//	string H5TestListFileName = CoreDir + "data/test_list.txt";
//	
//	ofstream test_list(H5TestListFileName.c_str(), ofstream::trunc);
//	ofstream train_list(H5TrainListFileName.c_str(), ofstream::trunc);

//	NumVecTbls = gen_data->vec_tbls_size();
//	
//	for (int ivt = 0; ivt < gen_data->vec_tbls_size(); ivt++) {
//		CaffeGenData::VecTbl vec_tbl = gen_data->vec_tbls(ivt);
//		TranslateTblNameMap[vec_tbl.name()] = ivt;
//		bool bThisTblIsTheVecTbl = false;
//		if (vec_tbl.name() == gen_data->dep_name_vec_tbl()) {
//			bThisTblIsTheVecTbl = true;
//			DepNames.clear();
//		}
//		map<string, int>* pNameMap = new map<string, int>();
//		TranslateTblPtrs.push_back(pNameMap);
//		vector<vector<float> >* pVecTbl = new vector<vector<float> >();
//		VecTblPtrs.push_back(pVecTbl);
//		string VecTblPath = gen_data->vec_tbls_core_path() + vec_tbl.path() + "/OutputVec.txt";
//		ifstream VecFile(VecTblPath.c_str());
//		if (VecFile.is_open()) {
//			string ln;
//			int NumVecs;
//			int NumValsPerVec;
//			VecFile >> NumVecs;
//			VecFile >> NumValsPerVec;
////			if (bThisTblIsTheVecTbl) {
////				pDepNames->resize(NumVecs);
////			}
//
//			getline(VecFile, ln); // excess .n
//			//while (!VecFile.eof()) {
//			for (int ic = 0; ic < NumVecs; ic++) {
//				getline(VecFile, ln, ' ');
//				string w;
//				w = ln;
//				if (w.size() == 0) {
//					cerr << "There should be as many items in the vec file as stated on the first line of file.\n";
//					return false;
//				}
//				if (bThisTblIsTheVecTbl) {
//					DepNames.push_back(w);
//				}
//				(*pNameMap)[w] = ic;
//				pVecTbl->push_back(vector<float>());
//				vector<float>& OneVec = pVecTbl->back();
//				//vector<float>& OneVec = WordsVecs[iw];
//				for (int iwv = 0; iwv < NumValsPerVec; iwv++) {
//					if (iwv == NumValsPerVec - 1) {
//						getline(VecFile, ln);
//					}
//					else {
//						getline(VecFile, ln, ' ');
//					}
//					float wv;
//					wv = ::atof(ln.c_str());
//					OneVec.push_back(wv);
//				}
//			}
//		}
//	}
//
//	vector<vector<float> > * YesNoTbl = new vector<vector<float> >;
//	YesNoTbl->push_back(vector<float>(1, 0.0f));
//	YesNoTbl->push_back(vector<float>(1, 1.0f));
//
//	
//	YesNoTblIdx = TranslateTblPtrs.size();
//	TranslateTblNameMap["YesNoTbl"] = YesNoTblIdx;
//	TranslateTblPtrs.push_back(NULL);
//	VecTblPtrs.push_back(YesNoTbl);
	return true;
}

#if 0
void GenDataModelComplete(CGenDef * InitData)
{
	
	//if (!CaffeFnHandle || !CaffeFnOutHandle) {
	if (!InitData || !(InitData->gen_data))	 {
		cerr << "GenDataModelComplete can only work if preceeded by a succesful call to GenDataModelInit.\n";
		return;
		
	}

	delete InitData;
}
#endif // 0

bool CGenDef::ModelPrep()
{
	//CaffeGenData* gen_data = gen_data; // (CaffeGenData *)CaffeFnHandle;
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
//	vector<pair<int, int> > InputTranslateTbl;
//	vector<pair<int, int> > OutputTranslateTbl;
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
}

bool  CGenDef::setReqTheOneOutput(int& OutputTheOneIdx) {
	if (OutputTranslateTbl.size() != 1) {
		cerr << "Error in setReqTheOneOutput: only models with 1 output are supported.\n";
		return false;
	}
	// The index of the var table acceesed in order to create the single output
	// is the index we are looking for. When an access or translation is made whose
	// output is meant to be written to that entry of the var table, it must have
	// the Avail value of datTheOne. That means that the NN built has the output
	// of the field we are looking to add
	pair<int, int>& iott = OutputTranslateTbl[0];
	OutputTheOneIdx = iott.first;
	
	return true;
	
}

void  CGenModelRun::setReqTheOneOutput() { 
	bReqTheOneOutput = true; 
	int RetIdx = -1;
	if (GenDef.setReqTheOneOutput(RetIdx)) {
		OutputTheOneIdx = RetIdx;
	}
}

bool CGenModelRun::DoRun() {
	DataForVecs.clear();
	
	vector<map<string, int>*>& TranslateTblPtrs = GenDef.TranslateTblPtrs;
//	vector<vector<vector<float> >* >& VecTblPtrs = GenDef.VecTblPtrs;
//	map<string, int>& TranslateTblNameMap = GenDef.TranslateTblNameMap;
	vector<string>& DepNames = GenDef.DepNames;
	int YesNoTblIdx = GenDef.YesNoTblIdx;
	CaffeGenData* gen_data = GenDef.getGenData();

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
	// step through records creating data based on translation tables just created
	
//	int NumWordsClean = RevWordMapClean.size();
//	int iNA = NumWordsClean - 1;
//	int iPosNA = PosMap.size() - 1; 
	 
	// not the actual vecs, but the integers that will give the vecs
	// random, IData, valid, OData
//	vector<SDataForVecs > DataForVecs;
	int NumCandidates = 0;
	
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
						cerr << "Founf the one!\n";
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


	return true;
}

