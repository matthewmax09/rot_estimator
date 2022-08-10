#pragma once

// Include the ROS C++ APIs
#include <ros/ros.h>
#include <ros/package.h>

// messages
#include <dvs_msgs/EventImage.h>
#include <std_msgs/Float32.h>

// torch
#include <torch/torch.h>
#include <torch/script.h>

class RotEstimator{
    public:
        RotEstimator(ros::NodeHandle &nh);

    private:
        ros::NodeHandle nh_;
        std::string ns;

        ros::Publisher rotation_pub;
        std_msgs::Float32 rotation_msg;

        ros::Subscriber eventImage_sub;
        void eventImageCallback(const dvs_msgs::EventImage::ConstPtr &msg);
        void warmup();

        c10::InferenceMode guard;
        torch::jit::script::Module model;
        std::vector<torch::jit::IValue> inputs;
        void loadModel();
        float median = 0.0f;
        torch::Tensor prev_img;
        bool prev_img_empty = true;

        template<typename T>
        int sgn(T val) {
		    return (T(0) < val) - (val < T(0));
        }
        
};