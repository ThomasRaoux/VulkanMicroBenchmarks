/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <chrono>
#include <string.h>

#include <vulkan/vulkan.h>

using std::vector;

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))

const unsigned int subgroupsize = 8;

#define CHECK_RESULT(r) do {    \
    if ((r) != VK_SUCCESS) {    \
        printf("result = %d, line = %d\n", (r), __LINE__);  \
        throw;  \
    }   \
} while (0)

// pasted from Vulkan spec
int32_t findProperties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
                       uint32_t memoryTypeBitsRequirement,
                       VkMemoryPropertyFlags requiredProperties) {
    const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
        const uint32_t memoryTypeBits = (1 << memoryIndex);
        const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

        const VkMemoryPropertyFlags properties =
            pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
        const bool hasRequiredProperties =
            (properties & requiredProperties) == requiredProperties;

        if (isRequiredMemoryType && hasRequiredProperties)
            return static_cast<int32_t>(memoryIndex);
    }

    // failed to find memory type
    return -1;
}

struct {
    const char *typeName;
    uint32_t bits;
} componentTypeInfo[] =
{
    { "float32_t",  32 },
    { "uint32_t",   32 },
};

enum class MatrixType {
    FLOAT_TYPE,
    INT32_TYPE,
    UNKNOWN_TYPE,
};

struct TestCase
{
    MatrixType inputType;
    MatrixType outputType;

    // MxNxK is the size of the full matrix multiply
    uint32_t M;
    uint32_t N;
    uint32_t K;

    // Each cooperative matrix multiply is lMxlNxlK
    uint32_t lM;
    uint32_t lN;
    uint32_t lK;

    // size of workgroup tile in destination matrix
    uint32_t TILE_M;
    uint32_t TILE_N;
    uint32_t TILE_K;

    bool BColMajor;
    uint32_t ARowLen;
    uint32_t ANumRows;
    uint32_t BRowLen;
    uint32_t BNumRows;
};

struct MatrixDesc
{
    struct
    {
        uint32_t rows, cols;
    } dims;
    MatrixType dataType;
    size_t elementSize;
    VkDeviceSize bufferSize;
    uint32_t totalElements;

    // Create a host- and device-local buffer for each input and output.
    // Descriptors point at the device buffers.
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkDeviceMemory deviceMemory;
    void *ptr;

    bool isFloatType() const
    {
        return dataType == MatrixType::FLOAT_TYPE;
    }

    void setDataFloat(uint32_t i, float value)
    {
        ((float *)ptr)[i] = value;
    }

    float getDataFloat(uint32_t i) const
    {
        return ((float *)ptr)[i];
    }

    float getDataFloat(int m, int n, bool colMajor) const
    {
        return getDataFloat(colMajor ? (n * dims.rows + m) : (m * dims.cols + n));
    }

    void setDataInt(uint32_t i, uint32_t value)
    {
        ((int32_t  *)ptr)[i] = (int32_t)value;
    }

    uint32_t getDataInt(uint32_t i) const
    {
	    return ((int32_t  *)ptr)[i];
    }

    uint32_t getDataInt(int m, int n, bool colMajor) const
    {
        return getDataInt(colMajor ? (n * dims.rows + m) : (m * dims.cols + n));
    }
};

// create storage for a matrix
void createMatrixDesc(VkDevice device, VkPhysicalDeviceMemoryProperties &memoryProperties, MatrixDesc &m, MatrixType dt, int rows, int cols)
{
    VkResult result;

    m.dims.rows = rows;
    m.dims.cols = cols;
    m.dataType = dt;
    m.elementSize = componentTypeInfo[(int)m.dataType].bits / 8;
    m.totalElements = m.dims.cols * m.dims.rows;
    m.bufferSize = m.totalElements * m.elementSize;

    VkBufferCreateInfo bufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        NULL,
        0,
        m.bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
        VK_SHARING_MODE_EXCLUSIVE,
        0u,
        NULL,
    };

    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.hostBuffer);
    CHECK_RESULT(result);
    result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.deviceBuffer);
    CHECK_RESULT(result);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, m.hostBuffer, &memReqs);

    int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    int32_t deviceIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo memAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        NULL,
        memReqs.size,
        (uint32_t)hostIndex,
    };

    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.hostMemory);
    CHECK_RESULT(result);

    memAllocateInfo.memoryTypeIndex = deviceIndex;
    result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.deviceMemory);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, m.hostBuffer, m.hostMemory, 0);
    CHECK_RESULT(result);

    result = vkBindBufferMemory(device, m.deviceBuffer, m.deviceMemory, 0);
    CHECK_RESULT(result);

    result = vkMapMemory(device, m.hostMemory, 0, m.bufferSize, 0, &m.ptr);
    CHECK_RESULT(result);
}

// destroy storage for a matrix
void destroyMatrixDesc(VkDevice device, MatrixDesc &m)
{
    vkDestroyBuffer(device, m.hostBuffer, NULL);
    vkDestroyBuffer(device, m.deviceBuffer, NULL);
    vkFreeMemory(device, m.hostMemory, NULL);
    vkFreeMemory(device, m.deviceMemory, NULL);
}

int main(int argc, char *argv[])
{
    bool correctness = true;

   // printf("usage: vk_cooperative_matrix_perf.exe [--correctness]\n\n");

    for (int arg = 1; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--correctness") == 0) {
            correctness = true;
        }
    }

    // Initialize Vulkan
    VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "Cooperative matrix performance test",
        1,
        "none",
        0,
        VK_MAKE_VERSION(1, 1, 0),
    };

    const char *enabledInstanceExtensions[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        NULL,
        0,
        &applicationInfo,
        0,
        NULL,
        1,
        enabledInstanceExtensions,
    };

    VkResult result;
    VkInstance instance;
    result = vkCreateInstance(&instanceCreateInfo, NULL, &instance);
    CHECK_RESULT(result);

    uint32_t numPhysicalDevices = 0;
    vector<VkPhysicalDevice> physicalDevices;

    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, NULL);
    CHECK_RESULT(result);

    physicalDevices.resize(numPhysicalDevices);
    result = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, &physicalDevices[0]);
    CHECK_RESULT(result);

    // pick the first device.
    int physicalDeviceIndex = 0;
    /*
    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {

        uint32_t numExtensions = 0;
        vector<VkExtensionProperties> extensions;

        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, NULL);
        CHECK_RESULT(result);

        extensions.resize(numExtensions);
        result = vkEnumerateDeviceExtensionProperties(physicalDevices[i], NULL, &numExtensions, &extensions[0]);
        CHECK_RESULT(result);

    }

    if (physicalDeviceIndex == -1) {
        printf("couldn't find physical device that supports VK_NV_cooperative_matrix\n");
        return 0;
    }*/
    VkPhysicalDevice physicalDevice = physicalDevices[physicalDeviceIndex];


    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    uint32_t numQueueFamilies = 0;
    vector<VkQueueFamilyProperties> queueFamilies;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, NULL);

    queueFamilies.resize(numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, &queueFamilies[0]);

    int queueFamilyIndex = -1;

    for (uint32_t i = 0; i < numPhysicalDevices; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == -1) {
        printf("couldn't find compute queue\n");
        return 0;
    }

    VkPhysicalDeviceBufferAddressFeaturesEXT bufferDeviceAddressFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
      NULL,
      VK_TRUE, // bufferDeviceAddress
      VK_FALSE, // bufferDeviceAddressCaptureReplay
      VK_FALSE, // bufferDeviceAddressMultiDevice
    };

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        NULL,
        0,
        (uint32_t)queueFamilyIndex,
        1,
        &queuePriority,
    };

    const char *enabledDeviceExtensions[] = { VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME };
    VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &bufferDeviceAddressFeatures,
        0,
        1,
        &deviceQueueCreateInfo,
        0,
        NULL,
        1,
        enabledDeviceExtensions,
        NULL,
    };

    VkDevice device;
    result = vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
    CHECK_RESULT(result);

    VkQueue queue;
    vkGetDeviceQueue(device, (uint32_t)queueFamilyIndex, 0, &queue);

    // The shaders use one UBO to pass in all the buffer addresses
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &layoutBinding,
    };

    VkDescriptorSetLayout descriptorSetLayout;
    result = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout);
    CHECK_RESULT(result);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        NULL,
        0,
        1,
        &descriptorSetLayout,
        0,
        NULL
    };

    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout);
    CHECK_RESULT(result);

    VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        NULL,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        1,
        ARRAY_LENGTH(poolSizes),
        poolSizes,
    };

    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool);
    CHECK_RESULT(result);

    VkDescriptorSetAllocateInfo setAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        NULL,
        descriptorPool,
        1,
        &descriptorSetLayout,
    };

    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet);
    CHECK_RESULT(result);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        (uint32_t)queueFamilyIndex,
    };

    VkCommandPool commandPool;
    result = vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
    CHECK_RESULT(result);

    // The command buffers, one for initializing buffers, one for compute, one
    // for reading back the results. This lets us time the compute work more
    // precisely.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        NULL,
        commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        3,
    };

    VkCommandBuffer commandBuffers[3];
    result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers);
    CHECK_RESULT(result);
    for(int ver =0; ver < 6; ver++)
    {
    {
        unsigned int elPerThread = 4;
        std::string fileName;
        if (ver == 0) {
            elPerThread = 4;
            fileName = "shaders/copy_vec4.spv";
        }
        else if (ver == 1) {
            elPerThread = 4;
            fileName = "shaders/copy_scalar_4.spv";
        }
        else if (ver == 2) {
            elPerThread = 1;
            fileName = "shaders/copy_scalar_1.spv";
        }
        else if (ver == 3) {
            elPerThread = 2;
            fileName = "shaders/copy_vec2.spv";
        }
        else if (ver == 4) {
            elPerThread = 8;
            fileName = "shaders/copy_vec4_2.spv";
        }
        else if (ver == 5) {
            // dummy info
            elPerThread = 1;
            fileName = "shaders/copy_scalar_1.spv";
        }

        if (ver == 5) {
            printf("\n hardware copy\n");
        } else {
        printf("\nshader: %s\n", fileName.c_str());
        }
        // Load and create the shader module.
        std::ifstream spirvfile(fileName.c_str(), std::ios::binary | std::ios::ate);
        std::streampos spirvsize = spirvfile.tellg();
        if ((int)spirvsize == -1) {
            printf("%s not found!\n", fileName.c_str());
            throw;
        }
        spirvfile.seekg(0, std::ios::beg);

        vector<char> spirv(spirvsize);
        spirvfile.read(&spirv[0], spirvsize);

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            NULL,
            0,
            spirv.size(),
            (const uint32_t *)&spirv[0],
        };

        VkShaderModule shaderModule;
        result = vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule);
        CHECK_RESULT(result);

        uint32_t MSize = subgroupsize;
        uint32_t NSize = subgroupsize;
        uint32_t KSize = 4;

        MatrixType MatType = MatrixType::FLOAT_TYPE;

        /*printf("\ncooperativeMatrixProps = %dx%dx%d   A = %s B = %s C = %s D = %s\n",
                MSize,
                NSize,
                KSize,
                componentTypeInfo[(int)MatType].typeName,
                componentTypeInfo[(int)MatType].typeName,
                componentTypeInfo[(int)MatType].typeName,
                componentTypeInfo[(int)MatType].typeName);
                */
        // For performance, test a 4096x4096x4096 multiply. For correctness,
        // test 256x256x256 (because the CPU reference computation is so slow).
       // uint32_t defaultDim = correctness ? 256 : 4096;
        uint32_t defaultDim = 4096;
        uint32_t defaultM = defaultDim;
        uint32_t defaultN = defaultDim;
        uint32_t defaultK = defaultDim;

        typedef struct {
            unsigned int maxTILE_M;
            unsigned int maxTILE_N;
            unsigned int granularityTILE_M;
            unsigned int granularityTILE_N;
        } SubTestParams;

        // TT_SHARED requires a multiple of 128x128 to satisfy the assumptions
        // of its SSBO->shared memory copy code.
        SubTestParams subTestParams = { 128, 128, subgroupsize, subgroupsize };
        bool BColMajor = false;
        SubTestParams* params = &subTestParams;

        //for (unsigned int TILE_M_size = params->granularityTILE_M; TILE_M_size <= params->maxTILE_M; TILE_M_size *= 2) {
        unsigned int TILE_M_size = params->granularityTILE_M; {
        double maxPerfThisIter = 0;
        //for (unsigned int TILE_N_size = params->granularityTILE_N; TILE_N_size <= params->maxTILE_N; TILE_N_size *= 2) {
        unsigned int TILE_N_size = params->granularityTILE_N; {
        //for (unsigned int bcolmajor = 0; bcolmajor <= 1; ++bcolmajor) {
            unsigned int bcolmajor = 0; {

            TestCase testCase = {
                MatType, // VkComponentTypeNV inputType;
                MatType, // VkComponentTypeNV outputType;

                // MxNxK is the size of the full matrix multiply
                defaultM, // uint32_t M;
                defaultN, // uint32_t N;
                defaultK, // uint32_t K;

                // Each cooperative matrix multiply is lMxlNxlK
                MSize, // uint32_t lM;
                NSize, // uint32_t lN;
                KSize, // uint32_t lK;

                // size of workgroup tile in destination matrix
                TILE_M_size, // uint32_t TILE_M;
                TILE_N_size, // uint32_t TILE_N;
                KSize, // uint32_t TILE_K;

                BColMajor, // bool BColMajor;
            };
            float alpha = 2.0f, beta = 3.0f;

            // For non-power of two tile sizes, round up the matrix size to
            // be an even multiple of the tile size.
            testCase.M = (testCase.M + testCase.TILE_M - 1) / testCase.TILE_M * testCase.TILE_M;
            testCase.N = (testCase.N + testCase.TILE_N - 1) / testCase.TILE_N * testCase.TILE_N;
            testCase.K = (testCase.K + testCase.TILE_K - 1) / testCase.TILE_K * testCase.TILE_K;

            testCase.ARowLen = testCase.TILE_K;
            testCase.ANumRows = testCase.TILE_M;
            testCase.BRowLen = BColMajor ? testCase.TILE_K : testCase.TILE_N;
            testCase.BNumRows = BColMajor ? testCase.TILE_N : testCase.TILE_K;

            enum {MAT_A = 0, MAT_B = 1, MAT_C = 2, MAT_D = 3, NUM_MATS = 4};

            MatrixDesc matrices[NUM_MATS];

            createMatrixDesc(device, memoryProperties, matrices[MAT_A], MatType, testCase.M, testCase.K);
            createMatrixDesc(device, memoryProperties, matrices[MAT_B], MatType, testCase.K, testCase.N);
            createMatrixDesc(device, memoryProperties, matrices[MAT_C], MatType, testCase.M, testCase.N);
            createMatrixDesc(device, memoryProperties, matrices[MAT_D], MatType, testCase.M, testCase.N);

            // Allocate buffer to hold device addresses for the four matrices
            VkBuffer paramBuffer;
            VkDeviceMemory paramMemory;
            void *paramPtr;

            VkBufferCreateInfo bufferCreateInfo = {
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                NULL,
                0,
                4*sizeof(VkDeviceAddress),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_SHARING_MODE_EXCLUSIVE,
                0u,
                NULL,
            };

            result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &paramBuffer);
            CHECK_RESULT(result);

            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(device, paramBuffer, &memReqs);

            int32_t hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

            VkMemoryAllocateInfo memAllocateInfo = {
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                NULL,
                memReqs.size,
                (uint32_t)hostIndex,
            };

            result = vkAllocateMemory(device, &memAllocateInfo, NULL, &paramMemory);
            CHECK_RESULT(result);

            result = vkBindBufferMemory(device, paramBuffer, paramMemory, 0);
            CHECK_RESULT(result);

            result = vkMapMemory(device, paramMemory, 0, bufferCreateInfo.size, 0, &paramPtr);
            CHECK_RESULT(result);

            PFN_vkGetBufferDeviceAddressEXT pfn_vkGetBufferDeviceAddressEXT =
                (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressEXT");

            for (int i = 0; i < NUM_MATS; ++i) {
                MatrixDesc &m = matrices[i];

                VkBufferDeviceAddressInfoEXT info = {
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
                    NULL,
                    0,
                };
                VkDeviceAddress *addrsInMemory = (VkDeviceAddress *)paramPtr;
                info.buffer = m.deviceBuffer;
                VkDeviceAddress addr = pfn_vkGetBufferDeviceAddressEXT(device, &info);
                addrsInMemory[i] = addr;
            }

            VkDescriptorBufferInfo bufferDescriptor;
            bufferDescriptor.buffer = paramBuffer;
            bufferDescriptor.offset = 0;
            bufferDescriptor.range = bufferCreateInfo.size;

            VkWriteDescriptorSet writeDescriptorset = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                NULL,
                descriptorSet,
                0, // dstBinding,
                0, // dstArrayElement
                1, // descriptorCount
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                NULL,
                &bufferDescriptor,
                NULL,
            };

            vkUpdateDescriptorSets(device, 1, &writeDescriptorset, 0, NULL);

            // Initialize input buffers to random values. These are relatively
            // small and have few mantissa bits set so we don't lose precision
            // in fp16 mode when running the correctness test.
            // Initialize the output buffer to an obvious value.
            for (uint32_t i = 0; i < NUM_MATS; ++i) {
                MatrixDesc &m = matrices[i];
                for (uint32_t j = 0; j < m.totalElements; ++j) {
                    if (m.isFloatType()) {
                        m.setDataFloat(j, ((float)(rand() & 0x3) - 1.0f) / 2.0f);
                        if (i == 3) {
                            m.setDataFloat(j, 1234.0f);
                        }
                    } else {
                        m.setDataInt(j, (rand() & 0xFF) - 128);
                        if (i == 3) {
                            m.setDataInt(j, 1234);
                        }
                    }
                }
            }

            // Specialize the shader with the matrix sizes, strides, and constants.
            const uint32_t specData[] = {
                testCase.lM,
                testCase.lN,
                testCase.lK,
                testCase.TILE_M,
                testCase.TILE_N,
                testCase.TILE_K,
                testCase.K,
                testCase.K, // stride0
                testCase.BColMajor ? testCase.K : testCase.N, // stride1
                testCase.N, // stride2
                testCase.N, // stride3
                *(uint32_t *)&alpha,
                *(uint32_t *)&beta,
                testCase.BColMajor,
                testCase.ARowLen,
                testCase.ANumRows,
                testCase.BRowLen,
                testCase.BNumRows,
            };

#if 0
            for (int i = 0; i < ARRAY_LENGTH(specData); ++i) {
                printf("specdata[%d] = %d\n", i, specData[i]);
            }
#endif

            VkPipelineShaderStageCreateInfo shaderCreateInfo = {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                NULL,
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                shaderModule,
                "main",
                NULL,
            };

            VkComputePipelineCreateInfo pipelineCreateInfo = {
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                NULL,
                0,
                shaderCreateInfo,
                pipelineLayout,
                VK_NULL_HANDLE,
                0
            };

            VkPipeline pipeline;
            result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
            CHECK_RESULT(result);

            VkCommandBufferBeginInfo commandBufferBeginInfo = {
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                NULL,
                0,
                NULL,
            };

            // Download input buffers to device memory.
            result = vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo);
            CHECK_RESULT(result);

            for (uint32_t i = 0; i < 4; ++i) {
                MatrixDesc &m = matrices[i];
                VkBufferCopy copy = { 0, 0, m.bufferSize };
                vkCmdCopyBuffer(commandBuffers[0], m.hostBuffer, m.deviceBuffer, 1, &copy);
            }

            result = vkEndCommandBuffer(commandBuffers[0]);
            CHECK_RESULT(result);

            VkSubmitInfo submitInfo = {
                VK_STRUCTURE_TYPE_SUBMIT_INFO,
                NULL,
                0,
                NULL,
                NULL,
                1,
                &commandBuffers[0],
                0,
                NULL,
            };

            submitInfo.pCommandBuffers = &commandBuffers[0];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            // Run the shader.
            result = vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo);
            CHECK_RESULT(result);
            //uint32_t repeatCount = correctness ? 1 : 10;
            uint32_t repeatCount = 10;
            if (ver == 5) {
                MatrixDesc& src = matrices[0];
                MatrixDesc& dst = matrices[3];
                VkBufferCopy copy = { 0, 0, src.bufferSize };
                vkCmdCopyBuffer(commandBuffers[1], src.deviceBuffer, dst.deviceBuffer, 1, &copy);
            }
            else {
                vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
                vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);



                for (uint32_t i = 0; i < repeatCount; ++i) {
                    //  vkCmdDispatch(commandBuffers[1], testCase.N / testCase.TILE_N, testCase.M / testCase.TILE_M, 1);
                    vkCmdDispatch(commandBuffers[1], (testCase.M / 32) / elPerThread, testCase.N, 1);
                }
            }
            result = vkEndCommandBuffer(commandBuffers[1]);
            CHECK_RESULT(result);

            // Time the submit/wait time for this command buffer.
            auto beginTime = std::chrono::high_resolution_clock::now();

            submitInfo.pCommandBuffers = &commandBuffers[1];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            auto endTime = std::chrono::high_resolution_clock::now();
            uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count();
            uint64_t flops = 2ULL * (uint64_t)testCase.M * (uint64_t)testCase.N * (uint64_t)testCase.K * (uint64_t)repeatCount;
            double tflops = (double)flops / (double)(elapsedUs / 1000000.0) / (1000.0*1000.0*1000.0*1000.0);

            uint64_t buffersizeinbytes = (uint64_t)testCase.M * (uint64_t)testCase.N * sizeof(float) * (uint64_t)repeatCount;
            double bandwidth = ((double)buffersizeinbytes / (double)elapsedUs)/1000.0; // in Gb/s
            // printf("TILE_M=%d TILE_N=%d, TILE_K=%d BColMajor=%d ", testCase.TILE_M, testCase.TILE_N, testCase.TILE_K, testCase.BColMajor);
           // if (1 || !correctness) {
           //     printf("  %f TFlops\n", tflops);
           // }
            printf("\n bandwidth = %f Gb/s \n", bandwidth);
            // Upload the result from device memory.
            result = vkBeginCommandBuffer(commandBuffers[2], &commandBufferBeginInfo);
            CHECK_RESULT(result);
            {
                MatrixDesc &m = matrices[MAT_D];
                VkBufferCopy copy = { 0, 0, m.bufferSize };
                vkCmdCopyBuffer(commandBuffers[2], m.deviceBuffer, m.hostBuffer, 1, &copy);
            }
            result = vkEndCommandBuffer(commandBuffers[2]);
            CHECK_RESULT(result);

            submitInfo.pCommandBuffers = &commandBuffers[2];
            result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
            CHECK_RESULT(result);
            result = vkQueueWaitIdle(queue);
            CHECK_RESULT(result);

            if (correctness)
            {


                const MatrixDesc &mat_a = matrices[MAT_A];
                const MatrixDesc &mat_b = matrices[MAT_B];
                const MatrixDesc &mat_c = matrices[MAT_C];
                const MatrixDesc &mat_d = matrices[MAT_D];
                bool pass = true;



                // hack verify copy
                for (uint32_t i = 0; i < testCase.M; ++i)
                {
                    for (uint32_t j = 0; j < testCase.N; ++j)
                    {
                        if (mat_a.getDataFloat(i, j, false) != mat_d.getDataFloat(i, j, false)) {
                            pass = false;
                            printf("error %d %d %f != %f\n", i, j, mat_a.getDataFloat(i, j, false), mat_d.getDataFloat(i, j, false));
                        }
                    }
                }
                if (0) {
                    if (mat_a.isFloatType()) {
                        for (uint32_t i = 0; i < testCase.M; ++i)
                        {
                            for (uint32_t j = 0; j < testCase.N; ++j)
                            {
                                float ref = 0;
                                for (uint32_t k = 0; k < testCase.K; ++k)
                                {
                                    ref += mat_a.getDataFloat(i, k, false) * mat_b.getDataFloat(k, j, testCase.BColMajor);
                                }

                                ref = alpha * ref + beta * mat_c.getDataFloat(i, j, false);

                                float Dij = mat_d.getDataFloat(i, j, false);
                                if (ref != Dij) {
                                    pass = false;
                                    printf("error %d %d %f != %f\n", i, j, ref, Dij);
                                }
                            }
                        }
                    }
                    else {
                        for (uint32_t i = 0; i < testCase.M; ++i)
                        {
                            for (uint32_t j = 0; j < testCase.N; ++j)
                            {
                                uint32_t ref = 0;
                                for (uint32_t k = 0; k < testCase.K; ++k)
                                {
                                    ref += mat_a.getDataInt(i, k, false) * mat_b.getDataInt(k, j, testCase.BColMajor);
                                }

                                ref = ((int)alpha) * ref + ((int)beta) * mat_c.getDataInt(i, j, false);

                                uint32_t Dij = mat_d.getDataInt(i, j, false);
                                if (ref != Dij) {
                                    pass = false;
                                    printf("error %d %d %d != %d\n", i, j, ref, Dij);
                                }
                            }
                        }
                    }
                }
                printf("\n%s\n", pass ? "pass" : "fail");
            }

            // Free the memory/buffers/pipeline for this iteration.
            for (int i = 0; i < NUM_MATS; ++i) {
                destroyMatrixDesc(device, matrices[i]);
            }
            vkDestroyPipeline(device, pipeline, NULL);

            if (maxPerfThisIter < tflops) {
                maxPerfThisIter = tflops;
            }

        } // bcolmajor
        } // TILE_N_size
        } // TILE_M_size

        vkDestroyShaderModule(device, shaderModule, NULL);
    } // numCooperativeMatrixProperties
    } // TT_COUNT

    printf("\ndone\n");

    return 0;
}
