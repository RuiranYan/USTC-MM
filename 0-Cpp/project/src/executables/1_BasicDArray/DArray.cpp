// implementation of class DArray
#include "DArray.h"
#include <iostream>
#include <cassert>

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

//copy constructor
DArray::DArray(const DArray &arr) {
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
}

// free the array
void DArray::Free() {
    //TODO
    delete[] m_pData;
    m_pData = nullptr;
    m_nSize = 0;
}

// get the size of the array
int DArray::GetSize() const {
    //TODO
    return m_nSize; // you should return a correct value
}

// set the size of the array
void DArray::SetSize(int nSize) {
    //TODO
    if (m_nSize == nSize) return;
    double* data = new double[nSize];
    if (m_nSize > nSize){
        for(int i = 0; i < nSize; ++i){
            data[i] = m_pData[i];
        }
        for (int i = nSize; i < m_nSize; ++i) {
            data[i] = 0.0;
        }
    }
    else {
        for (int i = 0; i < nSize; ++i) {
            data[i] = m_pData[i];
        }
    }
    m_nSize = nSize;
    delete[] m_pData;
    m_pData = data;
    data = nullptr;
}

// get an element at an index
const double &DArray::GetAt(int nIndex) const {
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
const double &DArray::operator[](int nIndex) const {
    //TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// add a new element at the end of the array
void DArray::PushBack(double dValue) {
    //TODO
    double * data = new double[m_nSize+1];
    for (int i = 0; i < m_nSize; ++i) {
        data[i] = m_pData[i];
    }
    data[m_nSize] = dValue;
    delete[] m_pData;
    m_pData = data;
    m_nSize++;
    data = nullptr;
}

// delete an element at some index
void DArray::DeleteAt(int nIndex) {
    //TODO
    assert(nIndex >= 0 && nIndex < m_nSize);
    double * data = new double[m_nSize-1];
    for (int i = 0; i < nIndex; ++i) {
        data[i] = m_pData[i];
    }
    for (int i = nIndex; i < m_nSize - 1; ++i) {
        data[i] = m_pData[i+1];
    }
    delete[] m_pData;
    m_pData = data;
    data = nullptr;
    m_nSize--;
}

// insert a new element at some index
void DArray::InsertAt(int nIndex, double dValue) {
    //TODO
    assert(nIndex >= 0 && nIndex <= m_nSize);
    double * data = new double[m_nSize + 1];
    for (int i = 0; i < nIndex; ++i) {
        data[i] = m_pData[i];
    }
    data[nIndex] = dValue;
    for (int i = nIndex; i < m_nSize ; ++i) {
        data[i+1] = m_pData[i];
    }
    delete[] m_pData;
    m_pData = data;
    data = nullptr;
    m_nSize++;
}

// overload operator '='
DArray &DArray::operator=(const DArray &arr) {
    //TODO
    delete[] m_pData;
    m_nSize = arr.m_nSize;
    m_pData = new double[m_nSize];
    for (int i = 0; i < m_nSize; i++)
        m_pData[i] = arr[i];
    return *this;
}
