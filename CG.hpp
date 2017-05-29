// =================================================================================================
// This file is a header file for implementing Conjugate Gradient Method
// Conjugate Gradient Method, cg<typename>(T a, Tb)
//
//
// Author:
//   Wang Zhang zw199006@gmail.com
//
//
// CLBlast is used for accelerating Matrix Opration
// =================================================================================================

#include <chrono>
#include <vector>
#include <iostream>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.hpp>

#include <clblast.h>

template< class T >
void cg(const cl::Buffer &buffer_a, const size_t ld,
    const cl::Buffer &buffer_b, cl::Buffer &buffer_x,
    const cl::Device &device, const cl::Context &context, const cl::CommandQueue &queue,
    const size_t it_max = 100, const T eps = 0.02) {

    std::vector<T> temp(ld, 0.0);

    auto queue_plain = queue();

    constexpr T alpha = -1.0;
    constexpr T beta  =  1.0;
    constexpr T alpha2 = 1.0;
    constexpr T beta2  = 0.0;

    T host_delta_new = 0.0, host_delta_old = 0.0;
    auto device_delta_new = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) );

    T host_al;
    auto device_al = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) );

    // 0.0 Set i = 0
    size_t it = 0;
    auto event = cl_event{nullptr};

    //auto host_r   = std::vector<T>(ld, 0.0);
    auto device_r = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(T) );
    //auto host_d   = std::vector<T>(ld, 0.0);
    auto device_d = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(T) );

    auto device_q = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(T) );
    auto device_q_zero = cl::Buffer(context, CL_MEM_READ_WRITE, ld*sizeof(T) );
    auto host_q_zero = std::vector<T>(ld, 0.0);
    queue.enqueueWriteBuffer(device_q_zero, CL_TRUE, 0, ld*sizeof(T), host_q_zero.data() );

    // 0.1 Calculate r = - A x + b
    // r = b;
    clblast::Copy<T>(ld,
                    buffer_b(), 0, 1,
                    device_r(), 0, 1,
                    &queue_plain);
    // r = - A x + r
    clblast::Gemv<T>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                      ld, ld,
                      alpha,
                      buffer_a(), 0, ld,
                      device_d(), 0, 1,
                      beta,
                      device_r(), 0, 1,
                      &queue_plain, &event);
    // 0.2 Copy r to d
    queue.enqueueCopyBuffer(device_r, device_d, 0, 0, ld*sizeof(T) );

    // 0.3 Calculate delta_new = dot(r, r);
    clblast::Dot<T>(ld,
                    device_delta_new(), 0,
                    device_r(), 0, 1,
                    device_r(), 0, 1,
                    &queue_plain);
    queue.enqueueReadBuffer(device_delta_new, CL_TRUE, 0, sizeof(T), &host_delta_new );
    // 0.4 Copy delta_new to delta_zero
    T host_delta_zero = host_delta_new;

    while ( it<it_max && host_delta_new>(eps*eps*host_delta_zero)) {

        // 1.0 Clear q with zeros, with beta2 = 0.0, not necesssary
        // queue.enqueueCopyBuffer(device_q_zero, device_q, 0, 0, ld*sizeof(T) );
        // 1.1 Calculate q = A d
        clblast::Gemv<T>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                          ld, ld,
                          1.0,
                          buffer_a(), 0, ld,
                          device_d(), 0, 1,
                          0.0,
                          device_q(), 0, 1,
                          &queue_plain, &event);
        // 2.0 Calculate dot(d, q)
        clblast::Dot<T>(ld,
                        device_al(), 0,
                        device_d(), 0, 1,
                        device_q(), 0, 1,
                        &queue_plain);
        queue.enqueueReadBuffer(device_al, CL_TRUE, 0, sizeof(T), &host_al);
        // 2.1 Calculate al = delta_new / al*
        host_al = host_delta_new / host_al;

        // 3.0 Calculate x = al*d + x
        clblast::Axpy<T>(ld,
                        host_al,
                        device_d(), 0, 1,
                        buffer_x(), 0, 1,
                        &queue_plain);

        // 4.0 If-Else regarding i%50 for updating r
        if ( (it%50) == 0 ) {
            // r = - A x + b
            // Step 1: r = b
            clblast::Copy<T>(ld,
                            buffer_b(), 0, 1,
                            device_r(), 0, 1,
                            &queue_plain);
            // Step 2: r = -Ax + r
            clblast::Gemv<T>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                              ld, ld,
                              alpha,
                              buffer_a(), 0, ld,
                              buffer_x(), 0, 1,
                              beta,
                              device_r(), 0, 1,
                              &queue_plain, &event);

        } else {
            // r = (-al)*q + r
            host_al = -host_al;
            clblast::Axpy<T>(ld,
                            host_al,
                            device_q(), 0, 1,
                            device_r(), 0, 1,
                            &queue_plain);

        }

        // 5.0 update delta_old
        host_delta_old = host_delta_new;

        //  6.0 Update delta_new = dot(r, r)
        clblast::Dot<T>(ld,
                        device_delta_new(), 0,
                        device_r(), 0, 1,
                        device_r(), 0, 1,
                        &queue_plain);
        // 6.1 Read delta_new
        queue.enqueueReadBuffer(device_delta_new, CL_TRUE, 0, sizeof(T), &host_delta_new);

        // 7.0 Update bt = delta_new / delta_old
        const T host_bt = host_delta_new / host_delta_old;

        // 8.0 Update d = r + bt * d
        // 8.1 Scale d = bt * d
        clblast::Scal<T>(ld,
                        host_bt,
                        device_d(), 0, 1,
                        &queue_plain);
        // 8.2 Calculate d = 1.0 * r + d
        clblast::Axpy<T>(ld,
                        1.0,
                        device_r(), 0, 1,
                        device_d(), 0, 1,
                        &queue_plain);

        // 8.3 Update iteration it
        it++;
    }

    queue.enqueueWaitForEvents({event});
    //clReleaseEvent(event);

    return;
}
