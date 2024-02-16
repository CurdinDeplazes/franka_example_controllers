// Started by Curdin on 31.01.2024

// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>           

#include <rclcpp/rclcpp.hpp>
#include "rclcpp/subscription.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigen>

#include <controller_interface/controller_interface.hpp>

#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <franka_hardware/franka_hardware_interface.hpp>
#include <franka_hardware/model.hpp>

#include "franka_msgs/msg/franka_robot_state.hpp"
#include "franka_msgs/msg/errors.hpp"

#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"


#define IDENTITY Eigen::MatrixXd::Identity(6, 6)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using Vector7d = Eigen::Matrix<double, 7, 1>;

namespace franka_example_controllers {

class JointImpedanceExampleController : public controller_interface::ControllerInterface {

public:

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;


private:
    //Nodes
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr franka_state_subscriber = nullptr;

    //Functions
    void topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg);
    void updateJointStates();
    void update_stiffness_and_references();
    void arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix);
    void arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix);
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d);  

    //State vectors and matrices
    std::array<double, 7> q_subscribed;
    std::array<double, 7> tau_J_d;
    std::array<double, 6> O_F_ext_hat_K;
    Eigen::Matrix<double, 7, 1> q_subscribed_M;
    Eigen::Matrix<double, 7, 1> tau_J_d_M;
    Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M;
    Eigen::Matrix<double, 7, 1> q_;
    Eigen::Matrix<double, 7, 1> dq_;
    Eigen::MatrixXd jacobian_transpose_pinv;  


    //Robot parameters
    const int num_joints = 7;
    const std::string state_interface_name_{"robot_state"};
    const std::string robot_name_{"panda"};
    const std::string k_robot_state_interface_name{"robot_state"};
    const std::string k_robot_model_interface_name{"robot_model"};
    franka_hardware::FrankaHardwareInterface interfaceClass;
    std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
    const double delta_tau_max_{1.0};
    const double dt = 0.001;
                
    //Impedance control variables              
    Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;                                           // operational space mass matrix
    Eigen::Matrix<double, 6, 6> Sm = IDENTITY;                                               // task space selection matrix for positions and rotation
    Eigen::Matrix<double, 6, 6> Sf = Eigen::MatrixXd::Zero(6, 6);                            // task space selection matrix for forces
    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 250,   0,   0,   0,   0,   0,
                                                                0, 250,   0,   0,   0,   0,
                                                                0,   0, 250,   0,   0,   0,  // impedance stiffness term
                                                                0,   0,   0,  80,   0,   0,
                                                                0,   0,   0,   0,  80,   0,
                                                                0,   0,   0,   0,   0,  10).finished();

    Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  55,   0,   0,   0,   0,   0,
                                                                0,  55,   0,   0,   0,   0,
                                                                0,   0,  55,   0,   0,   0,  // impedance damping term
                                                                0,   0,   0,  18,   0,   0,
                                                                0,   0,   0,   0,  18,   0,
                                                                0,   0,   0,   0,   0,   6).finished();

    Eigen::Matrix<double, 6, 6> T = IDENTITY;                                                // impedance inertia term

    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;                                 // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_;                                   // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_inertia_target_;                                   // impedance damping term

    Eigen::Matrix<double, 6, 1> F_impedance;  
    Eigen::Matrix<double, 6, 1> F_contact_des = Eigen::MatrixXd::Zero(6, 1);                 // desired contact force
    Eigen::Matrix<double, 6, 1> F_contact_target = Eigen::MatrixXd::Zero(6, 1);              // desired contact force used for filtering
    Eigen::Matrix<double, 6, 1> F_ext = Eigen::MatrixXd::Zero(6, 1);                         // external forces
    Eigen::Matrix<double, 6, 1> F_cmd = Eigen::MatrixXd::Zero(6, 1);                         // commanded contact force
    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Matrix<double, 6, 1> error;                                                       // pose error (6d)
    double nullspace_stiffness_{0.001};
    double nullspace_stiffness_target_{0.001};

    //Logging
    int outcounter = 0;
    const int update_frequency = 1; //frequency for update outputs

    //Integrator
    Eigen::Matrix<double, 6, 1> I_error = Eigen::MatrixXd::Zero(6, 1);                      // pose error (6d)
    Eigen::Matrix<double, 6, 1> I_F_error = Eigen::MatrixXd::Zero(6, 1);                    // force error integral
    Eigen::Matrix<double, 6, 1> integrator_weights = 
      (Eigen::MatrixXd(6,1) << 150.0, 150.0, 150.0, 150.0, 150.0, 4.0).finished();
    Eigen::Matrix<double, 6, 1> max_I = 
      (Eigen::MatrixXd(6,1) << 15.0, 15.0, 15.0, 8.0, 8.0, 1.0).finished();

    //Goal position
    Eigen::Vector3d position_d_; 
    Eigen::Quaterniond orientation_d_;
    Eigen::Vector3d position_d_target_;
    Eigen::Quaterniond orientation_d_target_;
    std::mutex position_and_orientation_d_target_mutex_;

    //Flags
    bool config_control = false;           // sets if we want to control the configuration of the robot in nullspace
    bool do_logging = false;               // set if we do log values

    //Filter-parameters
    double filter_params_{0.005};
};
}  // namespace franka_example_controllers