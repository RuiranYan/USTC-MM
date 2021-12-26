// implementation of class DArray
#include "DArray.h"

#include <iostream>

#include <cassert>

#include <cstring>

using namespace std;

// default constructor
DArray::DArray() {
	Init();
}

// set an array with default values
DArray::DArray(int nSize, double dValue) {
	//TODO
    m_nSize = nSize;
    m_pData = new double[nSize];
    for (int i = 0; i < nSize; ++i) {
        m_pData[i] = dValue;
    }
}

DArray::DArray(const DArray& arr) {
	//TODO
    m_nSize = arr.m_nSize;
    m_pData = new double[arr.m_nSize];
    for(int i = 0; i < arr.m_nSize; ++i){
        m_pData[i] = arr.m_pData[i];
    }
}

// deconstructor
DArray::~DArray() {
	Free();
}

// display the elements of the array
void DArray::Print() const {
	//TODO
    cout << "the size of the array is: " << m_nSize << "\nthe data is:";
    for(int i = 0; i < m_nSize; ++i){
        cout << " " << m_pData[i];
    }
    cout << endl;
}

// initilize the array
void DArray::Init() {
	//TODO
    m_pData = nullptr;
    m_nSize = 0;
    m_nMax = 0;
}

// free the array
void DArray::Free() {
	//TODO
    delete[] m_pData;
    m_pData = nullptr;
    m_nSize = 0;
    m_nMax = 0;
}

// get the size of the array
int DArray::GetSize() const {
	//TODO
	return m_nSize; // you should return a correct value
}

void DArray::Reserve(int nSize) {
    if (m_nMax >= nSize)
        return;

    while (m_nMax < nSize)
        m_nMax = m_nMax * 2 + 1;

    double* pData = new double[m_nMax];
    memcpy(pData, m_pData, m_nSize * sizeof(double));

    delete[] m_pData;
    m_pData = pData;
}

// set the size of the array
void DArray::SetSize(int nSize) {
	//TODO
    if (m_nSize == nSize)return;
    Reserve(nSize);
    for (int i = m_nSize; i < nSize; i++)
        m_pData[i] = 0.;
    m_nSize = nSize;
}

// get an element at an index
const double& DArray::GetAt(int nIndex) const {
	//TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// set the value of an element 
void DArray::SetAt(int nIndex, double dValue) {
	//TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    m_pData[nIndex] = dValue;
}

// overload operator '[]'
double& DArray::operator[](int nIndex) {
	// TODO
	static double ERROR; // you should delete this line
	return ERROR; // you should return a correct value
}

// overload operator '[]'
const double& DArray::operator[](int nIndex) const {
	//TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// add a new element at the end of the array
void DArray::PushBack(double dValue) {
	//TODO
    Reserve(m_nSize + 1);
    m_pData[m_nSize] = dValue;
    m_nSize++;
}

// delete an element at some index
void DArray::DeleteAt(int nIndex) {
	//TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    for (int i = nIndex + 1; i < m_nSize; i++)
        m_pData[i - 1] = m_pData[i];
    m_nSize--;
}

// insert a new element at some index
void DArray::InsertAt(int nIndex, double dValue) {
	//TODO
    assert(nIndex >= 0 && nIndex <= m_nSize); // nIndex == m_nSize is legal
    Reserve(m_nSize + 1);
    for (int i = m_nSize; i > nIndex; i--)
        m_pData[i] = m_pData[i - 1];
    m_pData[nIndex] = dValue;
    m_nSize++;
}

// overload operator '='
DArray& DArray::operator = (const DArray& arr) {
	//TODO
    Reserve(arr.m_nSize);
    m_nSize = arr.m_nSize;
    memcpy(m_pData, arr.m_pData, arr.m_nSize * sizeof(double));
	return *this;
}
