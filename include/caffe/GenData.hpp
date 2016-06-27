/* 
 * File:   GenData.hpp
 * Author: eli
 *
 * Created on February 4, 2016, 11:32 AM
 */

#ifndef GENDATA_HPP
#define GENDATA_HPP
#include <cstdlib>
#include <iostream>
//#include <minmax.h>
#include <fstream>
#include <sstream>
#include <cstring>

using namespace std;

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long long u64;

typedef unsigned char uchar;

// used for sentence rec
struct WordRec {
    WordRec() {
            bCap = false;
    }
    WordRec(std::string& aWord) {
            Word = aWord;
            bCap = false;
    }
    std::string RegionName;
    std::string Word;
    std::string WordCore;
    std::string POS;
    std::string NER;
    bool bCap; // only true if the first letter only was capitalized. We store WordRecin that case as all lowercase
    void Store(std::ofstream& fout);
    void Load(std::ifstream& fin);
};

struct DepRec {
    uchar iDep;
    uchar Gov;
    uchar Dep;
    void Store(std::ofstream& fout);
    void Load(std::ifstream& fin);
    bool operator == (DepRec& Other);
};

struct CorefRec {
    int SentenceID; // zero based index into SSentenceRec array
    int StartWordID; // zb index into WordRecArray of OneWordRec of SSentenceRec
    int EndWordID; // a coref mention can include a number of words
    int HeadWordId; // refers to the dependencey head of a phrase that makes up the mention
    int GovID; // zb index into the CorefRec array itself. Who is the first mention. Now all will point to him
    void Store(std::ofstream& fout);
    void Load(std::ifstream& fin);
    bool operator == (CorefRec& Other);
};

struct VarTblEl {
	VarTblEl() {
		bDorW = false; // but invalid
		R_DorW_ID = make_pair(-1, -1);
	}
	VarTblEl(bool b_deprec, int isr, int w_or_d_id) {
		bDorW = b_deprec;
		R_DorW_ID = make_pair(isr, w_or_d_id);
	}
	bool bDorW;
	pair<int, int> R_DorW_ID; // SRec ID, WID/DID
};



struct SSentenceRec {
    static bool lt(const SSentenceRec& r0, const SSentenceRec& r1) {
            return (r0.TextIDStart < r1.TextIDStart);
    }
    void Store(std::ofstream& fout);
    void Load(std::ifstream& fout);
    std::string Name;
    std::string Sentence;
    int TextIDStart;
    int TextIDEnd;
    int ParaNum;
    int SentInParaNum;
    std::string Label;
    std::vector<WordRec> OneWordRec;
    std::vector<DepRec> Deps; // not collapsed dependencies
};

enum DataAvailType {
    datConst, // record has a real value
    datNotSet, 
    datVar, // value not set but linked to some other record using name of variable. Not the same as GenData variable
    datTheOne, // used when initializing the deprec/wordrec/coref structure to find a specific new value. See gmail 12th Feb 2016
    datNotSetTooFar,
    datInvalid, // used for return from access functions only. use datNotSet in the record itslef if the value has not been set
};

struct SWordRecAvail {
    SWordRecAvail() {
        RegionName = datNotSet;
        Word = datNotSet;
        WordCore = datNotSet;
        POS = datNotSet;
        NER = datNotSet;
//		max_dep_govs = 0;
    }
    SWordRecAvail(DataAvailType InitVal) {
        RegionName = InitVal;
        Word = InitVal;
        WordCore = InitVal;
        POS = InitVal;
        NER = InitVal;
//		max_dep_govs = 0;
    }
    DataAvailType RegionName;
    DataAvailType Word;
    DataAvailType WordCore;
    DataAvailType POS;
    DataAvailType NER;
//	int max_dep_govs;
    
};

struct SDepRecAvail {
    SDepRecAvail() {
        iDep = datNotSet;
        Gov = datNotSet;
        Dep = datNotSet;
    }
    SDepRecAvail(DataAvailType InitVal) {
        iDep = InitVal;
        Gov = InitVal;
        Dep = InitVal;
    }
    DataAvailType iDep;
    DataAvailType Gov;
    DataAvailType Dep;
};

struct SSentenceRecAvail {
    std::vector<SWordRecAvail> WordRecs;
    std::vector<SDepRecAvail> Deps; // not collapsed dependencies
};

struct SVarCntrlEl {
    DataAvailType SrcStatus;
};

struct SDataForVecs {
    SDataForVecs(int aiRandom, std::vector<int>& aIData, bool abValid, std::vector<int>& aOData) {
        iRandom = aiRandom;
        IData  = aIData;
        bValid  = abValid;
        OData  = aOData;
    }
    int iRandom;
    std::vector<int> IData;
    bool bValid;
    std::vector<int> OData;
    static bool SortFn (const SDataForVecs& i,const SDataForVecs j) { return (i.iRandom<j.iRandom); }
};

enum DataTranslateEntryType {
        dtetRWIDToWord,
        dtetRWIDToCoref,
        dtetRWIDToRDID,
        dtetRDIDToDepRWID,
        dtetRDIDToGovRWID,
        dtetRDIDToDepName,
};
struct SDataTranslateEntry {
        DataTranslateEntryType dtet;
        int VarTblIdx; // index of field of var tbl to write result to 
        int VarTblMatchIdx; // index of search field to retrieve from var tbl
        int TargetTblMatchIdx; // index of field in target to match
        CaffeGenData_FieldType TargetTblOutputIdx; // idx of field in target table to output
};


class CGenModelRun;
class CGenDef;

class CGenDefTbls {
	friend CGenModelRun;
	friend CGenDef;
public:	
    CGenDefTbls(string& sModelProtoName);
	~CGenDefTbls() {
        for (int i=0; i < TranslateTblPtrs.size(); i++) {
            std::map<std::string, int>* p = TranslateTblPtrs[i];
            if (p != NULL) {
                delete p;
            }
        }
        for (int i=0; i < VecTblPtrs.size(); i++) {
            std::vector<std::vector<float> >* p = VecTblPtrs[i];
            if (p != NULL) {
                delete p;
            }
        }
		
	}
    
    vector<vector<vector<float> >* >& getVecTblPtrs() { return VecTblPtrs; }
    vector<map<string, int>*>& getTranslateTblPtrs() { return TranslateTblPtrs; }
    vector<string>& getDepNamesTbl() { return DepNames; };
	bool getTableNameIdx(const string& TblName, int& idx);

    bool bInitDone;
    
private:    
    std::vector<std::map<std::string, int>*> TranslateTblPtrs;
    std::vector<std::vector<std::vector<float> >* > VecTblPtrs;
    std::map<std::string, int> TranslateTblNameMap;
    vector<string> DepNames;
    
};

class CGenDef {
    friend CGenModelRun;
    
public:
    CGenDef(CGenDefTbls * apGenDefTbls, bool abYouOwnTblsData) :
			TranslateTblPtrs(apGenDefTbls->TranslateTblPtrs), 
			VecTblPtrs(apGenDefTbls->VecTblPtrs),
			TranslateTblNameMap(apGenDefTbls->TranslateTblNameMap),
			DepNames(apGenDefTbls->DepNames)
					
	{
        gen_def = NULL;
		pGenDefTbls = apGenDefTbls;
		bYouOwnTblsData = abYouOwnTblsData;
	    pGenDefTbls->getTableNameIdx(string("YesNoTbl"), YesNoTblIdx);
	    pGenDefTbls->getTableNameIdx(string("OrdinalVecTbl"), OrdinalVecTblIdx);

		
        //test = atest;
    }
    ~CGenDef() {
        if (gen_def != NULL) {
            delete gen_def;
        }
		if (bYouOwnTblsData) {
			delete pGenDefTbls;
		}
    }
    bool ModelInit(string sModelProtoName);
    bool ModelPrep(bool b_prep_from_src = false);
    //CaffeGenData* getGenData() {return gen_data; }
    CaffeGenDef* getGenDef() {return gen_def; }
    vector<pair<int, int> >& getInputTranslateTbl() { return InputTranslateTbl; }
    vector<pair<int, int> >& getOutputTranslateTbl() { return OutputTranslateTbl; }
    vector<vector<vector<float> >* >& getVecTblPtrs() { return VecTblPtrs; }
    vector<map<string, int>*>& getTranslateTblPtrs() { return TranslateTblPtrs; }
    vector<string>& getDepNamesTbl() { return DepNames; };
    bool setReqTheOneOutput(int& OutputTheOneIdx, bool& bIsPOS) ;
    int getNumOutputNodesNeeded() { return NumOutputNodesNeeded; }
    vector<int>& getOutputSetNumNodes() { return OutputSetNumNodes; }
	vector<pair<int, bool> >& getAccessFilterOrderTbl() { return AccessFilterOrderTbl; }
	int getYesNoTblIdx() { return YesNoTblIdx; }
	int getOrdinalVecTblIdx() { return OrdinalVecTblIdx; }
	//bool CreateAccessOrder();
//    void DoTest(int atest) {
//        test = atest;
//    }
private:
    //CaffeGenData* gen_data;
	CaffeGenDef * gen_def;
	CGenDefTbls * pGenDefTbls;
	bool bYouOwnTblsData;
    //CaffeGenSeed* gen_seed_config;
    int NumVecTbls;// remove
    std::vector<std::map<std::string, int>*>& TranslateTblPtrs;
    std::vector<std::vector<std::vector<float> >* >& VecTblPtrs;
    std::map<std::string, int>& TranslateTblNameMap;
    vector<string>& DepNames;
    int YesNoTblIdx; 
    int OrdinalVecTblIdx; 
	// for the following two tables the first is the index in the var table (access elements index)
	// and the second is the index of the table in the VecTblPtrs member variable of this class
    std::vector<std::pair<int, int> > InputTranslateTbl;
    std::vector<std::pair<int, int> > OutputTranslateTbl;
    int NumOutputNodesNeeded;
	vector<int> OutputSetNumNodes;
    vector<pair<CaffeGenData_FieldType, int> > FirstAccessFieldsIdx; 
    map<string, int> VarNamesMap;
    vector<SDataTranslateEntry> DataTranslateTbl; 
    vector<pair<int, string> > DataFilterTbl;
    vector<bool> ICanReplaceTbl;
    vector<bool> OCanReplaceTbl;
    // Num number of instances of that word in that data field
    // one for in and one for out
    // combining old and vew arrays
    // indexed the same as the translate tbl
    vector<vector<int> > NumInstancesTbl[2];
	// A new revision of NumInstancesTbl concept
	// has the same dimentionality as net_values and indexed as it is
    vector<vector<int> > InstanceCountTbl;
	vector<int> MaxInstancesTbl[2];
    bool bCanReplace;
	vector<pair<int, bool> > AccessFilterOrderTbl; // if second is true, first is access message index else filter message
	map<string, int> var_tbl_idx_map; // map of name of vars to their index in the AccessFilterOrderTbl
};

bool CreateAccessOrder(	vector<pair<int, bool> >& AccessFilterOrderTbl, 
						CaffeGenDef * gen_def,
						map<string, int>& var_tbl_idx_map,
						bool b_target_wrec_src = false);


class CGenModelRun {
public:
    CGenModelRun(   CGenDef& aGenDef,
                    std::vector<SSentenceRec>& aSentenceRec,
                    std::vector<CorefRec>& aCorefList,
                    std::vector<SSentenceRecAvail>& aSentenceAvailList,
                    std::vector<DataAvailType>& aCorefAvail,
					std::vector<SDataForVecs >& aDataForVecs) :
        GenDef(aGenDef), SentenceRec(aSentenceRec), CorefList(aCorefList),
        SentenceAvailList(aSentenceAvailList), CorefAvail(aCorefAvail),
		DataForVecs(aDataForVecs) {
            OutputTheOneIdx = -1;
            bReqTheOneOutput = false;
			bTheOneOutputIsPOS = false;
    }
    
    bool DoRun(bool bContinue = false);
    void setReqTheOneOutput() ;
	bool DepMatchTest(	const CaffeGenDef::DataAccess& data_access_rec, 
						int isr, int DID, int isrBeyond, bool b_use_avail);
	bool WordMatchTest(	const CaffeGenDef::DataAccess& data_access_rec, 
						int isr, int WID, int isrBeyond, bool b_use_avail);
	bool FilterTest(	const CaffeGenDef::DataFilter& data_filter,
						vector<VarTblEl> var_tbl);
	bool CreateAccessOrder();
    
private:
    string  GetDepRecField(	int SRecID, int DID, CaffeGenData_FieldType FieldID, 
                                DataAvailType& RetAvail, bool bUseAvail) {
		return string(); 
	}
    string GetRecFieldByIdx(int SRecID, int WID, 
                            CaffeGenData_FieldType FieldID, 
                            DataAvailType& RetAvail, bool bUseAvail); 

    CGenDef& GenDef;
    std::vector<SSentenceRec>& SentenceRec; 
    std::vector<CorefRec>& CorefList;
    std::vector<SSentenceRecAvail>& SentenceAvailList; 
	vector<vector<int> > max_dep_gov_list;
    std::vector<DataAvailType>& CorefAvail;        
    std::vector<SDataForVecs >& DataForVecs;
    bool bReqTheOneOutput;
    int OutputTheOneIdx;
	bool bTheOneOutputIsPOS;
   
};


std::string GetRecFieldByIdx(int SRecID, int WID, WordRec& rec, 
                            CaffeGenData_FieldType FieldID, bool& bRetValid);
std::string GetDepRecFieldByIdx( int SRecID, int DID, std::vector<std::string>& DepNames, DepRec& rec, 
                                CaffeGenData_FieldType FieldID, bool& bRetValid);
bool GenDataModelInit(std::string sModelProtoName, CGenDef * InitData );


#endif /* GENDATA_HPP */

