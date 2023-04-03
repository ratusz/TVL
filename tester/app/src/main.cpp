#include <tvlintrin.hpp>
#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <tuple>
#include <utility>
#include <vector>

// Time
#include <sys/time.h>
// Sleep
#include <unistd.h>

//#include <sycl/CL/sycl/INTEL/ac_types/ac_int.hpp>

using namespace sycl;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
//// Board globals. Can be changed from command line.
// default to values in pac_s10_usm BSP
#ifndef DDR_CHANNELS
#define DDR_CHANNELS 4
#endif

#ifndef DDR_WIDTH
#define DDR_WIDTH 64  // bytes (512 bits)
#endif

#ifndef PCIE_WIDTH
#define PCIE_WIDTH 64  // bytes (512 bits)
#endif

#ifndef DDR_INTERLEAVED_CHUNK_SIZE
#define DDR_INTERLEAVED_CHUNK_SIZE 4096  // bytes
#endif

// constexpr size_t kDDRChannels = DDR_CHANNELS;
// constexpr size_t kDDRWidth = DDR_WIDTH;
constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;
// constexpr size_t kPCIeWidth = PCIE_WIDTH;
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Forward declare functions

using Type = float;  // type to use for the test
using Type2 = unsigned int;
const size_t lower_bound = 5;
const size_t upper_bound = 15;

size_t aggregation_kernel_128(queue& q, Type* in_host, long* out_host, size_t size);
size_t aggregation_kernel_256(queue& q, Type* in_host, long* out_host, size_t size);
size_t aggregation_kernel_512(queue& q, Type* in_host, long* out_host, size_t size);
size_t aggregation_kernel_1024(queue& q, Type* in_host, long* out_host, size_t size);
size_t aggregation_kernel_2048(queue& q, Type* in_host, long* out_host, size_t size);

size_t lzc_kernel_128(queue& q, Type2* in_host, uint32_t* out_host, size_t size);
size_t lzc_kernel_256(queue& q, Type2* in_host, uint32_t* out_host, size_t size);
size_t lzc_kernel_512(queue& q, Type2* in_host, uint32_t* out_host, size_t size);
size_t lzc_kernel_1024(queue& q, Type2* in_host, uint32_t* out_host, size_t size);
size_t lzc_kernel_2048(queue& q, Type2* in_host, uint32_t* out_host, size_t size);

template <typename T>
bool validate(T* in_host, T* out_host, size_t size);

void exception_handler(exception_list exceptions);

// Function prototypes

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Forward declar kernel names to reduce name mangling
class aggregationKernel_tvl_128;
class aggregationKernel_tvl_256;
class aggregationKernel_tvl_512;
class aggregationKernel_tvl_1024;
class aggregationKernel_tvl_2048;

class lzcKernel_tvl_128;
class lzcKernel_tvl_256;
class lzcKernel_tvl_512;
class lzcKernel_tvl_1024;
class lzcKernel_tvl_2048;

////////////////////////////////////////////////////////////////////////////////

template <typename MyVec, typename T, typename U>
uint32_t agg_kernel_free_func(T in, size_t size, U lower, U upper) {
    using namespace tvl;
    using CountVec = tvl::simd<uint32_t, typename MyVec::target_extension, MyVec::vector_size_b()>;

    auto result_vec = set1<CountVec>(0);
    const auto increment_vec = set1<CountVec>(1);

    const auto lower_vec = set1<MyVec>(lower);
    const auto upper_vec = set1<MyVec>(upper);
    for (size_t i = 0; i < size; i += MyVec::vector_element_count()) {
        const auto data_vec = loadu<MyVec>(&in[i]);
        const auto result_mask = between_inclusive<MyVec>(data_vec, lower_vec, upper_vec);

        const auto increment_result_vec = binary_and<CountVec>(reinterpret<MyVec, CountVec>(to_vector<MyVec>(result_mask)), increment_vec);
        result_vec = add<CountVec>(result_vec, increment_result_vec);
    }

    return hadd<CountVec>(result_vec);
}

template <typename MyVec, typename T>
void lzc_kernel_free_func(T in, T out, size_t size) {
    using namespace tvl;
    using CountVec = tvl::simd<unsigned int, typename MyVec::target_extension, MyVec::vector_size_b()>;

    auto result_vec = set1<CountVec>(0);

    for (size_t i = 0; i < size; i += MyVec::vector_element_count()) {
        const auto data_vec = loadu<MyVec>(&in[i]);
         result_vec = lzc<CountVec>(data_vec);
         storeu<MyVec>(out,result_vec);//not defined for FPGA TODO copy from load
	 out+=MyVec::vector_element_count();
    }
}


template <typename MyVec, typename T, typename U>
double run(T in, size_t size, U lower, U upper) {
    double speed = 0;
    volatile size_t results = 0;
    for (size_t its = 0; its < 10; ++its) {
        auto start = std::chrono::high_resolution_clock::now();
        results += agg_kernel_free_func<MyVec>(in, size, lower, upper);
        auto end = std::chrono::high_resolution_clock::now();
        speed += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "Result for " << tvl::type_name<MyVec>() << ": " << results / 10 << std::endl;
    return speed / 10;
}

// main
int main(int argc, char* argv[]) {
    // make default input size enough to hide overhead
#ifdef FPGA_EMULATOR
    size_t size = kDDRInterleavedChunkSize * 4;//
#else
    size_t size = kDDRInterleavedChunkSize * 16384;
#endif

std::cout << "size defined " << size << std::endl;
    // the device selector
#ifdef FPGA_EMULATOR
    ext::intel::fpga_emulator_selector selector;
#else
    ext::intel::fpga_selector selector;
#endif

std::cout << "selector defined " << size << std::endl;    

// create the device queue
    auto props = property_list{property::queue::enable_profiling()};
    queue q(selector, exception_handler, props);

    std::cout << "queue created.. " << size << std::endl;


    // make sure the device supports USM device allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_device_allocations>()) {
        std::cerr << "ERROR: The selected device does not support USM device"
                  << " allocations" << std::endl;
        std::terminate();
    }
    if (!d.get_info<info::device::usm_host_allocations>()) {
        std::cerr << "ERROR: The selected device does not support USM host"
                  << " allocations" << std::endl;
        std::terminate();
    }

    std::cout << "usm allocation tested... " << size << std::endl;

    if (argc != 2)  // argc should be 2 for correct execution
    {
        size = 2048;
    } else {
        size = atoll(argv[1]);
    }

    std::cout << "Element count: " << size << std::endl;

    // Define for Allocate input/output data in pinned host memory
    // Used in all three tests, for convenience
    Type* in;
    Type2* in2;
    // int *out;
    long* out_aggr;
    long* out_aggr_2;
    uint32_t* out_lzc;	
    ////////////////
    // aggregation
    ////////////////

    std::cout << std::endl
              << "### aggregation###" << std::endl
              << std::endl;

    size_t number_CL = 0;


    const size_t vec_elems = 2048 / 8 / sizeof(Type);
    if (size % vec_elems == 0) {
        number_CL = size / vec_elems;
    } else {
        number_CL = size / vec_elems + 1;
    }

    std::cout << "Number CLs: " << number_CL << std::endl;
    std::cout << "size: " << size <<" vec_elems:" << vec_elems << " Type size:" <<sizeof(Type) << std::endl;


    // Allocate input/output data in pinned host memory
    // Used in both tests, for convenience

    if ((in = malloc_host<Type>(number_CL * vec_elems, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'" << std::endl;
        std::terminate();
    }
    if ((out_aggr = malloc_host<long>(5, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'" << std::endl;
        std::terminate();
    }

    std::mt19937 engine(0xc01dbadc00ffee);
    std::uniform_real_distribution<float> dist(0, 20);

    // Init input buffer
    out_aggr[0] = 0;
    for (int i = 0; i < (number_CL * vec_elems); ++i) {
        if (i < size) {
            in[i] = dist(engine);
            if (in[i] >= lower_bound && in[i] <= upper_bound) out_aggr[0]++;
        } else {
            in[i] = 0;
        }
    }

    // Init output buffer

    std::cout << "Buffer filled completely with '1'" << std::endl;
    std::cout << "in[0], in[1], in[2] ... : " << in[0] << " " << in[1] << " " << in[2] << std::endl;
    std::cout << "... in[size-1]: " << in[(number_CL * vec_elems) - 1] << std::endl;

    std::cout << "Serial aggregation: " << out_aggr[0] << std::endl;
    out_aggr[0] = 0;
    out_aggr[1] = 0;
    out_aggr[2] = 0;
    out_aggr[3] = 0;
    out_aggr[4] = 0;

    // track timing information, in ms
    size_t pcie_time = 0.0;
    size_t pcie_time_128 = 0.0;
    size_t pcie_time_256 = 0.0;
    size_t pcie_time_512 = 0.0;
    size_t pcie_time_1024 = 0.0;
    size_t pcie_time_2048 = 0.0;

    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout << "Running HOST-Aggregation test "
                  << "with " << size << " values" << std::endl;

        pcie_time = aggregation_kernel_128(q, in, out_aggr, vec_elems);   // dummy run to program FPGA, dont care first run for measurement
        pcie_time_128 = aggregation_kernel_128(q, in, out_aggr + 0, number_CL * vec_elems);
        pcie_time_256 = aggregation_kernel_256(q, in, out_aggr + 1, number_CL * vec_elems);
        pcie_time_512 = aggregation_kernel_512(q, in, out_aggr + 2, number_CL * vec_elems);
        pcie_time_1024 = aggregation_kernel_1024(q, in, out_aggr + 3, number_CL * vec_elems);
        pcie_time_2048 = aggregation_kernel_2048(q, in, out_aggr + 4, number_CL * vec_elems);

        ////////////////////////////////////////////////////////////////////////////
    } catch (exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }

    std::cout << "FPGA Aggregation Result [0]:\t" << out_aggr[0] << std::endl;
    std::cout << "FPGA Aggregation Result [1]:\t" << out_aggr[1] << std::endl;
    std::cout << "FPGA Aggregation Result [2]:\t" << out_aggr[2] << std::endl;
    std::cout << "FPGA Aggregation Result [3]:\t" << out_aggr[3] << std::endl;
    std::cout << "FPGA Aggregation Result [4]:\t" << out_aggr[4] << std::endl;
    std::cout << "pcie_time_128  [us]:\t" << pcie_time_128 << "us" << std::endl;
    std::cout << "pcie_time_256  [us]:\t" << pcie_time_256 << "us" << std::endl;
    std::cout << "pcie_time_512  [us]:\t" << pcie_time_512 << "us" << std::endl;
    std::cout << "pcie_time_1024 [us]:\t" << pcie_time_1024 << "us" << std::endl;
    std::cout << "pcie_time_2048 [us]:\t" << pcie_time_2048 << "us" << std::endl;

    // print result

    double input_size_mb = static_cast<double>(size * sizeof(Type)) / (1024 * 1024);

    std::cout << "Size of input data: " << input_size_mb << " MiB" << std::endl;
    std::cout << "HOST-DEVICE (128 ) Throughput: " << (input_size_mb / (pcie_time_128 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (256 ) Throughput: " << (input_size_mb / (pcie_time_256 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (512 ) Throughput: " << (input_size_mb / (pcie_time_512 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (1024) Throughput: " << (input_size_mb / (pcie_time_1024 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (2048) Throughput: " << (input_size_mb / (pcie_time_2048 * 1e-6)) << " MiB/s" << std::endl;

/////////////////////////////    
///////LZC counter ///////// 
////////////////////////////   
    size_t number_CL_lzc = 0;

    std::cout << std::endl
              << "### lzc ###" << std::endl
              << std::endl;

    
    const size_t vec_elems_lzc = 2048 / 8 / sizeof(Type2);
    if (size % vec_elems_lzc == 0) {
        number_CL_lzc = size / vec_elems_lzc;
    } else {
        number_CL_lzc = size / vec_elems_lzc + 1;
    }

    std::cout << "Number CLs LZC: " << number_CL_lzc << std::endl;
    std::cout << "size: " << size <<" vec_elems:" << vec_elems_lzc << " Type size:" <<sizeof(Type2) << std::endl;
    if ((in2 = malloc_host<Type2>(number_CL * vec_elems, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'" << std::endl;
        std::terminate();
    }
 
   if ((out_lzc = malloc_host<uint32_t>(size, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'" << std::endl;
        std::terminate();
    }
    for (int i = 0; i < (number_CL * vec_elems); ++i) {
            in2[i] = i;
    }


    pcie_time = lzc_kernel_128(q, in2, out_lzc, vec_elems_lzc);   // dummy run to program FPGA, dont care first run for measurement
 

    std::cout<<"Input values:"<<in2[0]<<", "<<in2[1]<<", "<<in2[2]<<", "<<in2[3]<<" ..."<<std::endl;    
    std::cout<<"Output values:"<<out_lzc[0]<<", "<<out_lzc[1]<<", "<<out_lzc[2]<<", "<<out_lzc[3]<<" ..."<<std::endl;    

 
    

    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout << "Running HOST-LZC test "
                  << "with " << size << " values" << std::endl;

//      pcie_time = lzc_kernel_128(q, in, out_lzc, vec_elems);   // dummy run to program FPGA, dont care first run for measurement
        pcie_time_128 = lzc_kernel_128(q, in2, out_lzc, number_CL * vec_elems);
        std::cout << "FPGA LZC Result 128:\t" << out_lzc[0] << ", " << out_lzc[1] <<", "<< out_lzc[2]<<", "<<out_lzc[3]<<" ..."<< std::endl;
        pcie_time_256 = lzc_kernel_256(q, in2, out_lzc, number_CL * vec_elems);
        std::cout << "FPGA LZC Result 256:\t" << out_lzc[0] << ", " << out_lzc[1] <<", "<< out_lzc[2]<<", "<<out_lzc[3]<<" ..."<< std::endl;
        pcie_time_512 = lzc_kernel_512(q, in2, out_lzc , number_CL * vec_elems);
        std::cout << "FPGA LZC Result 512:\t" << out_lzc[0] << ", " << out_lzc[1] <<", "<< out_lzc[2]<<", "<<out_lzc[3]<<" ..."<< std::endl;
        pcie_time_1024 = lzc_kernel_1024(q, in2, out_lzc , number_CL * vec_elems);
        std::cout << "FPGA LZC Result 1024:\t" << out_lzc[0] << ", " << out_lzc[1] <<", "<< out_lzc[2]<<", "<<out_lzc[3]<<" ..."<< std::endl;
        pcie_time_2048 = lzc_kernel_2048(q, in2, out_lzc , number_CL * vec_elems);
        std::cout << "FPGA LZC Result 2048:\t" << out_lzc[0] << ", " << out_lzc[1] <<", "<< out_lzc[2]<<", "<<out_lzc[3]<<" ..."<< std::endl;
 
        ////////////////////////////////////////////////////////////////////////////
    } catch (exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }

    std::cout << "pcie_time_128  [us]:\t" << pcie_time_128 << "us" << std::endl;
    std::cout << "pcie_time_256  [us]:\t" << pcie_time_256 << "us" << std::endl;
    std::cout << "pcie_time_512  [us]:\t" << pcie_time_512 << "us" << std::endl;
    std::cout << "pcie_time_1024 [us]:\t" << pcie_time_1024 << "us" << std::endl;
    std::cout << "pcie_time_2048 [us]:\t" << pcie_time_2048 << "us" << std::endl;

    // print result

    input_size_mb = static_cast<double>(size * sizeof(Type2)) / (1024 * 1024);

    std::cout << "Size of input data: " << input_size_mb << " MiB" << std::endl;
    std::cout << "HOST-DEVICE (128 ) Throughput: " << (input_size_mb / (pcie_time_128 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (256 ) Throughput: " << (input_size_mb / (pcie_time_256 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (512 ) Throughput: " << (input_size_mb / (pcie_time_512 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (1024) Throughput: " << (input_size_mb / (pcie_time_1024 * 1e-6)) << " MiB/s" << std::endl;
    std::cout << "HOST-DEVICE (2048) Throughput: " << (input_size_mb / (pcie_time_2048 * 1e-6)) << " MiB/s" << std::endl;


    // size_t res1 = run<tvl::simd<float, tvl::sse>>(in, (number_CL * vec_elems), lower_bound, upper_bound);
    // size_t res2 = run<tvl::simd<float, tvl::avx2>>(in, (number_CL * vec_elems), lower_bound, upper_bound);
    // size_t res3 = run<tvl::simd<float, tvl::avx512>>(in, (number_CL * vec_elems), lower_bound, upper_bound);

    // std::cout << "## Simd Testing ##" << std::endl;
    // std::cout << "SSE Tput: " << (input_size_mb / (res1 * 1e-6)) << " MiB/s "
    //           << "(" << res1 << " us)" << std::endl;
    // std::cout << "AVX2 Tput: " << (input_size_mb / (res2 * 1e-6)) << " MiB/s "
    //           << "(" << res2 << " us)" << std::endl;
    // std::cout << "AVX512 Tput: " << (input_size_mb / (res3 * 1e-6)) << " MiB/s "
    //           << "(" << res3 << " us)" << std::endl;

    // free USM memory
    sycl::free(in, q);
    sycl::free(out_aggr, q);
}

size_t aggregation_kernel_128(queue& q, Type* in_host, long* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<aggregationKernel_tvl_128>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type> in(in_host);
             host_ptr<long> out(out_host);
             out[0] = agg_kernel_free_func<tvl::simd<Type, tvl::fpga_generic, 128>>(in, size, static_cast<float>(lower_bound), static_cast<float>(upper_bound));
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


size_t lzc_kernel_128(queue& q, Type2* in_host, unsigned int* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<lzcKernel_tvl_128>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type2> in(in_host);
             host_ptr<unsigned int> out(out_host);
             lzc_kernel_free_func<tvl::simd<Type2, tvl::fpga_native, 128>>(in,out, size); //?
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


size_t aggregation_kernel_256(queue& q, Type* in_host, long* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<aggregationKernel_tvl_256>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type> in(in_host);
             host_ptr<long> out(out_host);
             out[0] = agg_kernel_free_func<tvl::simd<Type, tvl::fpga_generic, 256>>(in, size, static_cast<float>(lower_bound), static_cast<float>(upper_bound));
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}

size_t lzc_kernel_256(queue& q, Type2* in_host, unsigned int* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<lzcKernel_tvl_256>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type2> in(in_host);
             host_ptr<unsigned int> out(out_host);
             lzc_kernel_free_func<tvl::simd<Type2, tvl::fpga_native, 256>>(in,out, size); //?
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


size_t aggregation_kernel_512(queue& q, Type* in_host, long* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<aggregationKernel_tvl_512>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type> in(in_host);
             host_ptr<long> out(out_host);
             out[0] = agg_kernel_free_func<tvl::simd<Type, tvl::fpga_generic, 512>>(in, size, static_cast<float>(lower_bound), static_cast<float>(upper_bound));
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}

size_t lzc_kernel_512(queue& q, Type2* in_host, unsigned int* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<lzcKernel_tvl_512>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type2> in(in_host);
             host_ptr<unsigned int> out(out_host);
             lzc_kernel_free_func<tvl::simd<Type2, tvl::fpga_native, 512>>(in,out, size); //?
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


size_t aggregation_kernel_1024(queue& q, Type* in_host, long* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<aggregationKernel_tvl_1024>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type> in(in_host);
             host_ptr<long> out(out_host);
             out[0] = agg_kernel_free_func<tvl::simd<Type, tvl::fpga_generic, 1024>>(in, size, static_cast<float>(lower_bound), static_cast<float>(upper_bound));
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}

size_t lzc_kernel_1024(queue& q, Type2* in_host, unsigned int* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<lzcKernel_tvl_1024>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type2> in(in_host);
             host_ptr<unsigned int> out(out_host);
             lzc_kernel_free_func<tvl::simd<Type2, tvl::fpga_native, 1024>>(in,out, size); //?
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


size_t aggregation_kernel_2048(queue& q, Type* in_host, long* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<aggregationKernel_tvl_2048>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type> in(in_host);
             host_ptr<long> out(out_host);
             out[0] = agg_kernel_free_func<tvl::simd<Type, tvl::fpga_generic, 2048>>(in, size, static_cast<float>(lower_bound), static_cast<float>(upper_bound));
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}

size_t lzc_kernel_2048(queue& q, Type2* in_host, unsigned int* out_host, size_t size) {
    auto start = high_resolution_clock::now();
    q.submit([&](handler& h) {
         h.single_task<lzcKernel_tvl_2048>([=]() [[intel::kernel_args_restrict]] {
             host_ptr<Type2> in(in_host);
             host_ptr<unsigned int> out(out_host);
             lzc_kernel_free_func<tvl::simd<Type2, tvl::fpga_native, 2048>>(in,out, size); //?
         });
     }).wait();
    auto end = high_resolution_clock::now();

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return diff;
}


void exception_handler(exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
        }
    }
}

