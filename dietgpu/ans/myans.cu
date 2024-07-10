/**
 * @file myans.cpp
 * @brief A simple program that demonstrates the use of the ANS codec.
 *
 * This program demonstrates basic utilization of rANS codec from dietgpu that reads data from the disk.
 *
 * @author Boyuan Zhang
 * @date July 9, 2024
 *
 * @version 1.0
 * @license MIT
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

std::vector<std::vector<uint8_t>> readFileToVector(const std::string &filePath)
{
    auto out = std::vector<std::vector<uint8_t>>();

    // Create an input file stream in binary mode
    std::ifstream file(filePath, std::ios::binary);

    // Check if the file was opened successfully
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Seek to the end of the file to determine the file size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Create a vector with the appropriate size
    std::vector<uint8_t> buffer(fileSize);

    // Read the file into the vector
    if (!file.read(reinterpret_cast<char *>(buffer.data()), fileSize))
    {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    out.push_back(buffer);
    
    return out;
}

std::vector<GpuMemoryReservation<uint8_t>> toDevice(
    StackDeviceMemory &res,
    const std::vector<std::vector<uint8_t>> &vs,
    cudaStream_t stream)
{
    auto out = std::vector<GpuMemoryReservation<uint8_t>>();

    for (auto &v : vs)
    {
        out.emplace_back(res.copyAlloc(stream, v, AllocType::Permanent));
    }

    return out;
}

std::vector<std::vector<uint8_t>> toHost(
    StackDeviceMemory &res,
    const std::vector<GpuMemoryReservation<uint8_t>> &vs,
    cudaStream_t stream)
{
    auto out = std::vector<std::vector<uint8_t>>();

    for (auto &v : vs)
    {
        out.emplace_back(v.copyToHost(stream));
    }

    return out;
}

std::vector<GpuMemoryReservation<uint8_t>> buffersToDevice(
    StackDeviceMemory &res,
    const std::vector<uint32_t> &sizes,
    cudaStream_t stream)
{
    auto out = std::vector<GpuMemoryReservation<uint8_t>>();

    for (auto &s : sizes)
    {
        out.emplace_back(res.alloc<uint8_t>(stream, s, AllocType::Permanent));
    }

    return out;
}

void runRansCodec(
    const std::string &filePath)
{
    // run on a different stream to test stream assignment
    auto stream = CudaStream::makeNonBlocking();

    int prec = 10;

    auto res = makeStackMemory();

    auto batch_host = readFileToVector(filePath);
    auto batch_dev = toDevice(res, batch_host, stream);

    std::vector<uint32_t> batchSizes(1);

    batchSizes[0] = batch_host[0].size();

    int numInBatch = batchSizes.size();
    uint32_t maxSize = 0;
    for (auto v : batchSizes)
    {
        maxSize = std::max(maxSize, v);
    }

    auto outBatchStride = getMaxCompressedSize(maxSize);

    auto inPtrs = std::vector<const void *>(batchSizes.size());
    {
        for (int i = 0; i < inPtrs.size(); ++i)
        {
            inPtrs[i] = batch_dev[i].data();
        }
    }

    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

    auto encPtrs = std::vector<void *>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i)
    {
        encPtrs[i] = (uint8_t *)enc_dev.data() + i * outBatchStride;
    }

    auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    cudaStreamSynchronize(stream);
    auto cStart = std::chrono::high_resolution_clock::now();

    ansEncodeBatchPointer(
        res,
        ANSCodecConfig(prec, true),
        numInBatch,
        inPtrs.data(),
        batchSizes.data(),
        nullptr,
        encPtrs.data(),
        outCompressedSize_dev.data(),
        stream);
    
    cudaStreamSynchronize(stream);
    auto cEnd = std::chrono::high_resolution_clock::now();

    auto encSize = outCompressedSize_dev.copyToHost(stream);
    
    // report the compression ratio
    std::cout << "Compression ratio: " << (double)batch_host[0].size() / encSize[0] << std::endl;

    // report the time taken and compression throughput in GB/s
    std::chrono::duration<double, std::milli> cElapsed = cEnd - cStart;
    std::cout << "Time taken: " << cElapsed.count() << " ms" << std::endl;
    std::cout << "Compression throughput: " << (double)batch_host[0].size() / (cElapsed.count() * 1e6) << " GB/s" << std::endl;

    for (auto v : encSize)
    {
        // Reported compressed sizes in bytes should be a multiple of 16 for aligned
        // packing
        EXPECT_EQ(v % 16, 0);
    }

    // Decode data
    auto dec_dev = buffersToDevice(res, batchSizes, stream);

    auto decPtrs = std::vector<void *>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i)
    {
        decPtrs[i] = dec_dev[i].data();
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    cudaStreamSynchronize(stream);
    auto dStart = std::chrono::high_resolution_clock::now();

    ansDecodeBatchPointer(
        res,
        ANSCodecConfig(prec, true),
        numInBatch,
        (const void **)encPtrs.data(),
        decPtrs.data(),
        batchSizes.data(),
        outSuccess_dev.data(),
        outSize_dev.data(),
        stream);

    cudaStreamSynchronize(stream);
    auto dEnd = std::chrono::high_resolution_clock::now();

    // report the time taken in and decompression throughput in GB/s
    std::chrono::duration<double, std::milli> dElapsed = dEnd - dStart;
    std::cout << "Time taken: " << dElapsed.count() << " ms" << std::endl;
    std::cout << "Decompression throughput: " << (double)batch_host[0].size() / (dElapsed.count() * 1e6) << " GB/s" << std::endl;

    auto outSuccess = outSuccess_dev.copyToHost(stream);
    auto outSize = outSize_dev.copyToHost(stream);

    for (int i = 0; i < outSuccess.size(); ++i)
    {
        EXPECT_TRUE(outSuccess[i]);
        EXPECT_EQ(outSize[i], batchSizes[i]);
    }

    auto dec_host = toHost(res, dec_dev, stream);
    EXPECT_EQ(batch_host, dec_host);
}

int main(int argc, char **argv)
{
    // process the argv
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    }

    // run the codec
    runRansCodec(argv[1]);

    return 0;
}
