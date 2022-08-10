#include "rot_estimator/ros_estimator.h"

uint count = 0;

// Standard C++ entry point
int main(int argc, char** argv) {
    // Announce this program to the ROS master as a "node" called "hello_world_node"
    ros::init(argc, argv, "rot_estimator_node");

    // init Ros Node Handle
    ros::NodeHandle nh;

    // init rot_estimator
    std::unique_ptr<RotEstimator> rot_estimator (new RotEstimator(nh));

    // Process ROS callbacks until receiving a SIGINT (ctrl-c)
    ros::spin();
    // Exit tranquilly
    return 0;
}