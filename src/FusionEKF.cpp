#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    //measurement matrix
    H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

    Hj_ << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
   *  Initialization
   ****************************************************************************/
    if (!is_initialized_) {

        previous_timestamp_ = 0;

        //create a 4D state vector, we don't know yet the values of the x state
        ekf_.x_ = VectorXd(4);

        //state covariance matrix P
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

        //measurement matrix
        ekf_.H_ = MatrixXd(2, 4);
        ekf_.H_ << 1, 0, 0, 0,
                0, 1, 0, 0;

        //the initial transition matrix F_
        ekf_.F_ = MatrixXd(4, 4);
        ekf_.F_ << 1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;

        ekf_.Q_ = MatrixXd(4, 4);

        // first measurement
        cout << "EKF: " << endl;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            double theta = measurement_pack.raw_measurements_[1];
            double r = measurement_pack.raw_measurements_[0];
            double v = measurement_pack.raw_measurements_[2];
            double x = r * cos( theta );
            double y = r * sin( theta );
            double vx = v * cos( theta );
            double vy = v * sin( theta );

            ekf_.x_ << x, y, vx, vy;
        }

        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
   *  Prediction
   ****************************************************************************/


    //compute the time elapsed between the current and previous measurements
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    //set the acceleration noise components
    double noise_ax = 9;
    double noise_ay = 9;

    //1. Modify the F matrix so that the time is integrated
    ekf_.F_(0,2) = dt;
    ekf_.F_(1,3) = dt;

    //2. Set the process covariance matrix Q

    ekf_.Q_ << (pow(dt, 4) / 4) * noise_ax, 0 , (pow(dt, 3)/2) * noise_ax, 0,
            0, (pow(dt,4) / 4) * noise_ay, 0, (pow(dt, 3)/2) * noise_ay,
            (pow(dt, 3)/2) * noise_ax, 0, pow(dt, 2) * noise_ax, 0,
            0, (pow(dt, 3)/2) * noise_ay, 0, pow(dt, 2) * noise_ay;

    ekf_.Predict();

    /*****************************************************************************
   *  Update
   ****************************************************************************/


    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        ekf_.R_ = R_radar_;
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
