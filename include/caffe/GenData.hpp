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
    }
    SWordRecAvail(DataAvailType InitVal) {
        RegionName = InitVal;
        Word = InitVal;
        WordCore = InitVal;
        POS = InitVal;
        NER = InitVal;
    }
    DataAvailType RegionName;
    DataAvailType Word;
    DataAvailType WordCore;
    DataAvailType POS;
    DataAvailType NER;
    
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
//			std::vector<std::map<std::string, int>*> aTranslateTblPtrs,
//			std::vector<std::vector<std::vector<float> >* > aVecTblPtrs,
//			std::map<std::string, int> aTranslateTblNameMap,
//			vector<string> aDepNames) :
			TranslateTblPtrs(apGenDefTbls->TranslateTblPtrs), 
			VecTblPtrs(apGenDefTbls->VecTblPtrs),
			TranslateTblNameMap(apGenDefTbls->TranslateTblNameMap),
			DepNames(apGenDefTbls->DepNames)
					
	{
        gen_data = NULL;
		pGenDefTbls = apGenDefTbls;
		bYouOwnTblsData = abYouOwnTblsData;
		
        //test = atest;
    }
    ~CGenDef() {
        if (gen_data != NULL) {
            delete gen_data;
        }
		if (bYouOwnTblsData) {
			delete pGenDefTbls;
		}
    }
    bool ModelInit(string sModelProtoName);
    bool ModelPrep();
    CaffeGenData* getGenData() {return gen_data; }
    vector<pair<int, int> >& getInputTranslateTbl() { return InputTranslateTbl; }
    vector<pair<int, int> >& getOutputTranslateTbl() { return OutputTranslateTbl; }
    vector<vector<vector<float> >* >& getVecTblPtrs() { return VecTblPtrs; }
    vector<map<string, int>*>& getTranslateTblPtrs() { return TranslateTblPtrs; }
    vector<string>& getDepNamesTbl() { return DepNames; };
    bool setReqTheOneOutput(int& OutputTheOneIdx) ;
    int getNumOutputNodesNeeded() { return NumOutputNodesNeeded; }
//    void DoTest(int atest) {
//        test = atest;
//    }
private:
    CaffeGenData* gen_data;
	CGenDefTbls * pGenDefTbls;
	bool bYouOwnTblsData;
    //CaffeGenSeed* gen_seed_config;
    int NumVecTbls;// remove
    std::vector<std::map<std::string, int>*>& TranslateTblPtrs;
    std::vector<std::vector<std::vector<float> >* >& VecTblPtrs;
    std::map<std::string, int>& TranslateTblNameMap;
    vector<string>& DepNames;
    int YesNoTblIdx; // remove
    std::vector<std::pair<int, int> > InputTranslateTbl;
    std::vector<std::pair<int, int> > OutputTranslateTbl;
    int NumOutputNodesNeeded;
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
    vector<int> MaxInstancesTbl[2];
    bool bCanReplace;



};

class CGenModelRun {
public:
    CGenModelRun(   CGenDef& aGenDef,
                    std::vector<SSentenceRec>& aSentenceRec,
                    std::vector<CorefRec>& aCorefList,
                    std::vector<SSentenceRecAvail>& aSentenceAvailList,
                    std::vector<DataAvailType>& aCorefAvail) :
        GenDef(aGenDef), SentenceRec(aSentenceRec), CorefList(aCorefList),
        SentenceAvailList(aSentenceAvailList), CorefAvail(aCorefAvail) {
            OutputTheOneIdx = -1;
            bReqTheOneOutput = false;
    }
    
    bool DoRun();
    vector<SDataForVecs >& getDataForVecs() { return DataForVecs; }
    void setReqTheOneOutput() ;
    
private:
    string  GetDepRecField(	int SRecID, int DID, CaffeGenData_FieldType FieldID, 
                                DataAvailType& RetAvail, bool bUseAvail) ;
    string GetRecFieldByIdx(int SRecID, int WID, 
                            CaffeGenData_FieldType FieldID, 
                            DataAvailType& RetAvail, bool bUseAvail);

    CGenDef& GenDef;
    std::vector<SSentenceRec>& SentenceRec; 
    std::vector<CorefRec>& CorefList;
    std::vector<SSentenceRecAvail>& SentenceAvailList; 
    std::vector<DataAvailType>& CorefAvail;        
    std::vector<SDataForVecs > DataForVecs;
    bool bReqTheOneOutput;
    int OutputTheOneIdx;
   
};


std::string GetRecFieldByIdx(int SRecID, int WID, WordRec& rec, 
                            CaffeGenData_FieldType FieldID, bool& bRetValid);
std::string GetDepRecFieldByIdx( int SRecID, int DID, std::vector<std::string>& DepNames, DepRec& rec, 
                                CaffeGenData_FieldType FieldID, bool& bRetValid);
bool GenDataModelInit(std::string sModelProtoName, CGenDef * InitData );


#endif /* GENDATA_HPP */

