#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {



    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(!estimations.size() || estimations.size() != ground_truth.size()){
        cout << "Error on vectors sizes" << endl;
        exit(-1);
    }

    //accumulate squared residuals
    cout << "point 1: " << estimations.size() << endl;

    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd ac = (estimations[i]-ground_truth[i]);
        ac = ac.array() * ac.array();

        rmse += ac;
        cout << i << ": "<< rmse << endl;
    }

    //calculate the mean
    rmse /= estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3,4);
    //recover state parameters
    float px = static_cast<float>(x_state(0));
    float py = static_cast<float>(x_state(1));
    float vx = static_cast<float>(x_state(2));
    float vy = static_cast<float>(x_state(3));

    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);

    //check division by zero
    if(fabs(c1) < 0.0001){
        cerr << "CalculateJacobian () - Error - Division by Zero: " << Hj << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << static_cast<double>((px/c2)), static_cast<double>((py/c2)), 0, 0,
            static_cast<double>(-(py/c1)), static_cast<double>((px/c1)), 0, 0,
            static_cast<double>(py*(vx*py - vy*px)/c3), static_cast<double>(px*(px*vy - py*vx)/c3), static_cast<double>(px/c2), static_cast<double>(py/c2);

    return Hj;
}
