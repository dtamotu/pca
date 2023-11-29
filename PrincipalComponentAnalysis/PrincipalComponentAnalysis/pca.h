#ifndef PCA_H
#define PCA_H

#include <iostream>
#include <assert.h>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;
using namespace std;

class PCA
{
public:

    /*
     * Computes the princcipal component of a given matrix. Computation steps:
     * Assert that the input matrix is square matrix.
     * Compute the mean image.
     * Subtract mean image from the data set to get mean centered data vector
     * Compute the covariance matrix from the mean centered data matrix
     * Calculate the eigenvalues and eigenvectors for the covariance matrix
     * Normalize the eigen vectors
     * Find out an eigenvector with the largest eigenvalue
     * 
     * @input MatrixXd D the data samples matrix.
     * 
     * @returns VectorXd The principal component vector
     */
    static VectorXd Compute(MatrixXd D)
    {
        // The matrix must be square matrix.
        assert(D.rows() == D.cols());
        int N = D.rows();

        // 1. Asegurarse de que la Matriz sea Cuadrada
        MatrixXd mean(1, N);
        mean.setZero();

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                mean(0, j) += D(i, j) / N;
            }
        }

        // 2 Cálculo de la Media y Centrado de Datos: Calcula la media
        // de cada columna de D y luego centra los datos restando esta media.
        MatrixXd U = D;

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                U(i, j) -= mean(0, j);
            }
        }

        // 3. Cálculo de la Matriz de Covarianza a partir de los datos centrados.
        MatrixXd covariance = (U.transpose() * U) / (double)(N);

        cout <<"covarianza:  "<< endl<<covariance << endl << endl << endl;

        // 4. Cálculo de Autovalores y Autovectores
        EigenSolver<MatrixXd> solver(covariance);
        MatrixXd eigenVectors = solver.eigenvectors().real();
        VectorXd eigenValues = solver.eigenvalues().real();

        // 5. Normalización de Autovectores: Normaliza los autovectores obtenidos.
        eigenVectors.normalize();

        cout <<"Autovectores" << endl << eigenVectors << endl;
        cout << "Autovalores"<<endl<< eigenValues << endl << endl << endl << endl;

        // 6. Selección del Componente Principal: Ordena los autovalores y 
        // selecciona el autovector asociado con el mayor autovalor.
        sort(eigenValues.derived().data(), eigenValues.derived().data() + eigenValues.derived().size());
        short index = eigenValues.size() - 1;
        VectorXd featureVector = eigenVectors.row(index);

        return featureVector;
    }
};

#endif // PCA_H
