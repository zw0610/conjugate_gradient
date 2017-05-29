#include "CG.hpp"

#include "CL/cl.hpp"



int main(int argc, char const *argv[]) {
    // Setup OpenCL Environment
    constexpr size_t platform_id = 0;
    constexpr size_t device_id = 0;
    // Platform
    auto platforms = std::vector<cl::Platform>();
    cl::Platform::get(&platforms);
    if (platforms.size() == 0 || platform_id >= platforms.size()) {
        return 1;
    }
    cl::Platform default_platform = platforms[platform_id];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    // Device
    std::vector<cl::Device> devices_list;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_list);
    if ( devices_list.size() == 0 ) {
        std::cout << "No available device found." << std::endl;
        return 1;
    } else {
        std::cout << "There are " << devices_list.size() << " devices(s) in total." << std::endl;
        for (size_t num = 0; num<devices_list.size(); num++) {
            std::cout << "Device No." << num << " : " << devices_list[num].getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }
    cl::Device default_device = devices_list[device_id];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
    // Context, Queue
    auto context = cl::Context(devices_list);
    auto dcQueue = cl::CommandQueue( context, default_device);


    // Initialize A, b, x, ld
    constexpr size_t ld = 2;
    auto host_x = std::vector<float>(ld, 0.0f);
    std::vector<float> host_b{ 2.0f, -8.0f };
    std::vector<float> host_a{ 3.0f, 2.0f, 2.0f, 6.0f};

    auto buffer_a = cl::Buffer(context, CL_MEM_READ_WRITE, ld*ld*sizeof(float));
    dcQueue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, ld*ld*sizeof(float), host_a.data());
    auto buffer_b = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(float));
    dcQueue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, ld*sizeof(float), host_b.data());
    auto buffer_x = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(float));
    dcQueue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, ld*sizeof(float), host_x.data());

    cg<float>(buffer_a, ld,
        buffer_b, buffer_x,
        default_device, context, dcQueue);

    dcQueue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, ld*sizeof(float), host_x.data());

    for (auto i : host_x) {
        std::cout << i << std::endl;
    }


    return 0;
}
