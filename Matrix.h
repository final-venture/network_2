#pragma once

#include <cstdint>

#include <vector>

#include <ostream>

#include <functional>

#include <stdexcept>

#include <sstream>

class Matrix;

Matrix operator*(const Matrix &m, double scalar);

Matrix operator*(double scalar, const Matrix &m);

Matrix operator*(const Matrix &m1, const Matrix &m2);

Matrix operator+(const Matrix &m1, const Matrix &m2);

Matrix operator-(const Matrix &m1, const Matrix &m2);

class Matrix

{

public:
    Matrix();

    Matrix(uint32_t r, uint32_t c);

    Matrix(const std::vector<std::vector<double>> &data);

    Matrix(std::vector<std::vector<double>> &&data);

    bool addRow(const std::vector<double> &row);

    bool addRow(std::vector<double> &&row);

    bool addCol(const std::vector<double> &col);

    uint32_t getRowCount() const;

    uint32_t getColCount() const;

    double operator()(uint32_t r, uint32_t c) const;

    double &operator()(uint32_t r, uint32_t c);

    Matrix &operator+=(const Matrix &other);

    Matrix &operator-=(const Matrix &other);

    Matrix &operator*=(double scalar);

    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

    void apply(std::function<double(double)> func);

private:
    bool addRowImpl(std::vector<double> &&row);

    bool isValid(uint32_t r, uint32_t c) const;

    std::vector<std::vector<double>> matrix_;
};