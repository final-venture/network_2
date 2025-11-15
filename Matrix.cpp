#include "Matrix.h"

Matrix operator*(const Matrix &m1, const Matrix &m2)
{
    uint32_t m1Rows = m1.getRowCount(), m1Cols = m1.getColCount();
    uint32_t m2Rows = m2.getRowCount(), m2Cols = m2.getColCount();
    if (m1Cols != m2Rows)
    {
        std::ostringstream oss;
        oss << "Cannot multiply " << m1Rows << " x " << m1Cols << " and" << " " << m2Rows << " x " << m2Cols;
        throw std::invalid_argument(oss.str());
    }
    Matrix resultMatrix = Matrix{m1Rows, m2Cols};
    for (uint32_t r = 0; r < m1Rows; ++r)
    {
        for (uint32_t c = 0; c < m1Cols; ++c)
        {
            for (uint32_t k = 0; k < m2Cols; ++k)
            {
                resultMatrix(r, k) = resultMatrix(r, k) + (m1(r, c) * m2(c, k));
            }
        }
    }
    return resultMatrix;
}

Matrix operator+(const Matrix &m1, const Matrix &m2)
{
    Matrix result(m1);
    result += m2;
    return result;
}

Matrix operator-(const Matrix &m1, const Matrix &m2)
{
    Matrix result(m1);
    result -= m2;
    return result;
}

Matrix operator*(const Matrix &m, double scalar)
{
    Matrix result(m);
    result *= scalar;
    return result;
}

Matrix operator*(double scalar, const Matrix &m) { return m * scalar; }

void Matrix::apply(std::function<double(double)> func)
{
    for (uint32_t r = 0; r < getRowCount(); ++r)
    {
        for (uint32_t c = 0; c < getColCount(); ++c)
        {
            matrix_[r][c] = func(matrix_[r][c]);
        }
    }
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    uint32_t m1Rows = getRowCount(), m1Cols = getColCount();
    uint32_t m2Rows = other.getRowCount(), m2Cols = other.getColCount();
    if ((m1Rows != m2Rows) || (m1Cols != m2Cols))
    {
        std::ostringstream oss;
        oss << "Cannot add " << m1Rows << " x " << m1Cols << " and " << m2Rows << " x " << m2Cols;
        throw std::invalid_argument(oss.str());
    }
    for (uint32_t r = 0; r < m1Rows; ++r)
    {
        for (uint32_t c = 0; c < m1Cols; ++c)
        {
            matrix_[r][c] += other(r, c);
        }
    }
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    uint32_t m1Rows = getRowCount(), m1Cols = getColCount();
    uint32_t m2Rows = other.getRowCount(), m2Cols = other.getColCount();
    if ((m1Rows != m2Rows) || (m1Cols != m2Cols))
    {
        std::ostringstream oss;
        oss << "Cannot subtract " << m2Rows << " x " << m2Cols << " from " << m1Rows << " x " << m1Cols;
        throw std::invalid_argument(oss.str());
    }
    for (uint32_t r = 0; r < m1Rows; ++r)
    {
        for (uint32_t c = 0; c < m1Cols; ++c)
        {
            matrix_[r][c] -= other(r, c);
        }
    }
    return *this;
}

Matrix &Matrix::operator*=(double scalar)
{
    for (uint32_t r = 0; r < getRowCount(); ++r)
    {
        for (uint32_t c = 0; c < getColCount(); ++c)
        {
            matrix_[r][c] *= scalar;
        }
    }
    return *this;
}

Matrix::Matrix() {}

Matrix::Matrix(uint32_t r, uint32_t c)
{
    if (r == 0 || c == 0)
    {
        std::ostringstream oss;
        oss << "Cannot create matrix with zero dimension: " << r << " x " << c;
        throw std::invalid_argument(oss.str());
    }
    matrix_.resize(r, std::vector<double>(c, 0));
}

Matrix::Matrix(const std::vector<std::vector<double>> &data) : matrix_(data)
{
    if (getRowCount() == 0 || getColCount() == 0)
    {
        std::ostringstream oss;
        oss << "Cannot create matrix from data with zero dimension: " << getRowCount() << " x " << getColCount();
        throw std::invalid_argument(oss.str());
    }
}

Matrix::Matrix(std::vector<std::vector<double>> &&data) : matrix_(std::move(data))
{
    if (getRowCount() == 0 || getColCount() == 0)
    {
        std::ostringstream oss;
        oss << "Cannot create matrix from data with zero dimension: " << getRowCount() << " x " << getColCount();
        throw std::invalid_argument(oss.str());
    }
}

bool Matrix::addRow(const std::vector<double> &row) { return addRowImpl(std::vector<double>(row)); }

bool Matrix::addRow(std::vector<double> &&row) { return addRowImpl(std::move(row)); }

bool Matrix::addCol(const std::vector<double> &col)
{
    uint32_t colSize = col.size();
    if (colSize != getRowCount())
    {
        std::ostringstream oss;
        oss << "Cannot add column of size " << colSize << " to " << getRowCount() << " x " << getColCount() << " matrix!";
        throw std::invalid_argument(oss.str());
    }
    for (size_t i = 0; i < colSize; ++i)
    {
        matrix_[i].push_back(col[i]);
    }
    return true;
}

uint32_t Matrix::getRowCount() const { return matrix_.size(); }

uint32_t Matrix::getColCount() const { return (matrix_.empty() ? 0 : matrix_[0].size()); }

double Matrix::operator()(uint32_t r, uint32_t c) const
{
    if (!isValid(r, c))
    {
        std::ostringstream oss;
        oss << "Cannot access element at (" << r << ", " << c << ") in " << getRowCount() << " x " << getColCount() << " matrix!";
        throw std::out_of_range(oss.str());
    }
    return matrix_[r][c];
}

double &Matrix::operator()(uint32_t r, uint32_t c)
{
    if (!isValid(r, c))
    {
        std::ostringstream oss;
        oss << "Cannot access element at (" << r << ", " << c << ") in " << getRowCount() << " x " << getColCount() << " matrix!";
        throw std::out_of_range(oss.str());
    }
    return matrix_[r][c];
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix)
{
    for (uint32_t r = 0; r < matrix.getRowCount(); ++r)
    {
        for (uint32_t c = 0; c < matrix.getColCount(); ++c)
        {
            os << matrix(r, c) << "  ";
        }
        os << '\n';
    }
    return os;
}

// PRIVATE
bool Matrix::isValid(uint32_t r, uint32_t c) const { return (r < getRowCount()) && (c < getColCount()); }

bool Matrix::addRowImpl(std::vector<double> &&row)
{
    uint32_t rowSize = row.size();
    if (rowSize != getColCount() && !matrix_.empty())
    {
        std::ostringstream oss;
        oss << "Cannot add row of size " << rowSize << " to " << getRowCount() << " x " << getColCount() << " matrix!";
        throw std::invalid_argument(oss.str());
    }
    if (rowSize == 0)
    {
        std::ostringstream oss;
        oss << "Cannot initialize matrix with an empty row!";
        throw std::invalid_argument(oss.str());
    }
    matrix_.push_back(std::move(row));
    return true;
}
