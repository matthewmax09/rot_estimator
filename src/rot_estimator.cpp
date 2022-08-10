#include "rot_estimator/ros_estimator.h"

RotEstimator::RotEstimator(ros::NodeHandle &nh) : nh_(nh) {

    // Load torchscript model
    loadModel();
    // warm up model
    warmup();
    // Check and attach node to dvs namespace
    ns = ros::this_node::getNamespace();
	if (ns == "/") {
		ns = "/dvs";
	}
    // Advertise rotation
    rotation_pub = nh_.advertise<std_msgs::Float32>(ns+ "/rotation",10);
    // Subsribe to eventImage message
    eventImage_sub = nh_.subscribe("/dvs/eventImage", 1, &RotEstimator::eventImageCallback, this);

}

void RotEstimator::eventImageCallback(const dvs_msgs::EventImage::ConstPtr &msg) {

    // Received image from dvs driver, convert to tensor and transfer to CUDA memory
    torch::Tensor t1 = torch::from_blob((void *)msg->data.data(),{400,400},torch::kFloat32)
                            .to(torch::kCUDA);

    if (!prev_img_empty){

        inputs.clear();
        inputs.push_back(prev_img);
        inputs.push_back(t1);
        float theta = model.forward(inputs).toTensor().item<float>();
        median += 0.2 * sgn(theta - median);
        rotation_msg.data = median;
        rotation_pub.publish(rotation_msg);

        
    }
    else{
        // Check that tensor is really empty
        ROS_ASSERT_MSG(prev_img.numel()==0, "Seems like tensor is not empty! Expected tensor to be empty");
        // Set boolean to false
        prev_img_empty = false;
    }
    // Else store latest img in tensor
    prev_img = t1;
    
}

void RotEstimator::warmup(){

    torch::Tensor a = torch::rand({400,400});

    for(int i = 0; i < 1000; ++i){

        // Transfer from CPU to CUDA
        auto b = a.to(torch::kCUDA);
        // Load tensors into input vector
        inputs.push_back(b);
        inputs.push_back(b);
        // Compute rotation from input tensors
        torch::Tensor theta = model.forward(inputs).toTensor();
        // Clear vector
        inputs.clear();
    }
    ROS_INFO("Warm up completed");
    ros::Time start = ros::Time::now();
    torch::Tensor b = a.to(torch::kCUDA);
    inputs.push_back(b); inputs.push_back(b);
    torch::Tensor theta = model.forward(inputs).toTensor();
    double duration = (ros::Time::now()- start).toSec()*1000;
    ROS_INFO_STREAM("Time taken to compute forward after warm up: "<< duration << " milliseconds");
    ROS_INFO_STREAM("Inference mode activated: " << std::boolalpha << torch::is_inference(b));
    inputs.clear();

}

void RotEstimator::loadModel(){
    // Get full path to torchscript model
    std::string path = ros::package::getPath("rot_estimator");
    path += "/scripts/fmt_scripted.pt";
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        RotEstimator::model = torch::jit::load(path);
        model.eval();
        ROS_INFO("Successfully loaded model");
    }
    catch (const c10::Error& e) {
        ROS_ERROR("Error loading the model\n");
    }

}
