#ifndef LSFG_EXTERNAL_ONLY
#include <volk.h>
#endif
#include <vulkan/vulkan_core.h>

#if __has_include("../public/lsfg_3_1.hpp")
#include "../public/lsfg_3_1.hpp"
#else
#include "lsfg_3_1.hpp"
#endif
#ifndef LSFG_EXTERNAL_ONLY
#include "v3_1/context.hpp"
#include "core/commandpool.hpp"
#include "core/descriptorpool.hpp"
#include "core/instance.hpp"
#include "pool/shaderpool.hpp"
#include "common/exception.hpp"
#include "common/utils.hpp"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef LSFG_EXTERNAL_ONLY
using namespace LSFG;
using namespace LSFG_3_1;

namespace {
    std::optional<Core::Instance> instance;
    std::optional<Vulkan> device;
    std::unordered_map<int32_t, Context> contexts;
}

void LSFG_3_1::initialize(uint64_t deviceUUID,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader) {
    if (instance.has_value() || device.has_value())
        return;

    instance.emplace();
    device.emplace(Vulkan {
        .device{*instance, deviceUUID},
        .generationCount = generationCount,
        .flowScale = flowScale,
        .isHdr = isHdr
    });
    contexts = std::unordered_map<int32_t, Context>();

    device->commandPool = Core::CommandPool(device->device);
    device->descriptorPool = Core::DescriptorPool(device->device);

    device->resources = Pool::ResourcePool(device->isHdr, device->flowScale);
    device->shaders = Pool::ShaderPool(loader);

    std::srand(static_cast<uint32_t>(std::time(nullptr)));
}

int32_t LSFG_3_1::createContext(
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    const int32_t id = std::rand();
    contexts.emplace(id, Context(*device, in0, in1, outN, extent, format));
    return id;
}

void LSFG_3_1::presentContext(int32_t id, int inSem, const std::vector<int>& outSem) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    auto it = contexts.find(id);
    if (it == contexts.end())
        throw LSFG::vulkan_error(VK_ERROR_UNKNOWN, "Context not found");

    it->second.present(*device, inSem, outSem);
}

void LSFG_3_1::deleteContext(int32_t id) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    auto it = contexts.find(id);
    if (it == contexts.end())
        throw LSFG::vulkan_error(VK_ERROR_DEVICE_LOST, "No such context");

    // vkDeviceWaitIdle omitted for external backend non-blocking recording
    contexts.erase(it);
}

void LSFG_3_1::finalize() {
    if (!instance.has_value() || !device.has_value())
        return;

    // vkDeviceWaitIdle omitted for external backend non-blocking recording
    contexts.clear();
    device.reset();
    instance.reset();
}

#ifdef __ANDROID__

#include <android/hardware_buffer.h>

int32_t LSFG_3_1::createContextFromAHB(
        AHardwareBuffer* in0, AHardwareBuffer* in1,
        const std::vector<AHardwareBuffer*>& outN,
        VkExtent2D extent, VkFormat format) {
    if (!instance.has_value() || !device.has_value())
        throw LSFG::vulkan_error(VK_ERROR_INITIALIZATION_FAILED, "LSFG not initialized");

    const int32_t id = std::rand();
    contexts.emplace(id, Context(*device, in0, in1, outN, extent, format));
    return id;
}

#endif // __ANDROID__

#ifdef __ANDROID__
void LSFG_3_1::waitIdle() {
    if (!device.has_value()) return;
    // vkDeviceWaitIdle omitted for external backend non-blocking recording
}
#endif
#endif // !LSFG_EXTERNAL_ONLY

// -----------------------------------------------------------------------------
// External Device Context API Implementation (REAL-LSFG-R0)
// -----------------------------------------------------------------------------

namespace LSFG_3_1 {

struct ExternalPipelineInfo {
    uint32_t resId = 0;
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct ExternalSlotState {
    VkDescriptorSet mipmapsSet = VK_NULL_HANDLE;
    VkDescriptorSet generateSet = VK_NULL_HANDLE;
    VkDescriptorSet generateSetR4A = VK_NULL_HANDLE;
    VkDescriptorSet generateSetR4B = VK_NULL_HANDLE;
    VkBuffer constantBuffer = VK_NULL_HANDLE;
    VkDeviceMemory constantBufferMemory = VK_NULL_HANDLE;
};

// -----------------------------------------------------------------------------
// Authoritative LS-FG 3.1 Shader Descriptor ABI Table (res_255 .. res_279)
// -----------------------------------------------------------------------------

struct ShaderBindingDef {
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkShaderStageFlags stageFlags;
};

struct ShaderAbiDef {
    uint32_t resId;
    const char* name;
    uint32_t bindingCount;
    const ShaderBindingDef* bindings;
};

// 255: mipmaps (10 bindings: 1 UBO, 1 Sampler, 1 Sampled Image, 7 Storage Images)
static const ShaderBindingDef kBindings_255[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 256: generate (9 bindings: 1 UBO, 2 Samplers, 5 Sampled Images, 1 Storage Image)
static const ShaderBindingDef kBindings_256[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 257: gamma[0] / delta[0] (15 bindings: 1 UBO, 2 Samplers, 9 Sampled Images, 3 Storage Images)
static const ShaderBindingDef kBindings_257[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 258: delta[5] (15 bindings: 1 UBO, 2 Samplers, 10 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_258[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 259: gamma[1] (8 bindings: 1 Sampler, 3 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_259[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 260: gamma[2] (9 bindings: 1 Sampler, 4 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_260[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 261: gamma[3] (9 bindings: 1 Sampler, 4 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_261[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 262: gamma[4] (10 bindings: 1 UBO, 2 Samplers, 6 Sampled Images, 1 Storage Image)
static const ShaderBindingDef kBindings_262[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 263: delta[1] (8 bindings: 1 Sampler, 3 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_263[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 264: delta[2] (9 bindings: 1 Sampler, 4 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_264[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 265: delta[3] (9 bindings: 1 Sampler, 4 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_265[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 266: delta[4] (10 bindings: 1 UBO, 2 Samplers, 6 Sampled Images, 1 Storage Image)
static const ShaderBindingDef kBindings_266[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 267: alpha[0] (4 bindings: 1 Sampler, 1 Sampled Image, 2 Storage Images)
static const ShaderBindingDef kBindings_267[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 268: alpha[1] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_268[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 269: alpha[2] (7 bindings: 1 Sampler, 2 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_269[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 270: alpha[3] (9 bindings: 1 Sampler, 4 Sampled Images, 4 Storage Images)
static const ShaderBindingDef kBindings_270[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 271: delta[6] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_271[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 272: delta[7] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_272[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 273: delta[8] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_273[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 274: delta[9] (7 bindings: 1 UBO, 2 Samplers, 3 Sampled Images, 1 Storage Image)
static const ShaderBindingDef kBindings_274[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 275: beta[0] (15 bindings: 1 Sampler, 12 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_275[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 276: beta[1] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_276[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 277: beta[2] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_277[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 278: beta[3] (5 bindings: 1 Sampler, 2 Sampled Images, 2 Storage Images)
static const ShaderBindingDef kBindings_278[] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

// 279: beta[4] (10 bindings: 1 UBO, 1 Sampler, 2 Sampled Images, 6 Storage Images)
static const ShaderBindingDef kBindings_279[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
    { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
};

static const ShaderAbiDef kLsfg31ShaderAbi[] = {
    { 255, "res_255", 10, kBindings_255 },
    { 256, "res_256", 9,  kBindings_256 },
    { 257, "res_257", 15, kBindings_257 },
    { 258, "res_258", 15, kBindings_258 },
    { 259, "res_259", 8,  kBindings_259 },
    { 260, "res_260", 9,  kBindings_260 },
    { 261, "res_261", 9,  kBindings_261 },
    { 262, "res_262", 10, kBindings_262 },
    { 263, "res_263", 8,  kBindings_263 },
    { 264, "res_264", 9,  kBindings_264 },
    { 265, "res_265", 9,  kBindings_265 },
    { 266, "res_266", 10, kBindings_266 },
    { 267, "res_267", 4,  kBindings_267 },
    { 268, "res_268", 5,  kBindings_268 },
    { 269, "res_269", 7,  kBindings_269 },
    { 270, "res_270", 9,  kBindings_270 },
    { 271, "res_271", 5,  kBindings_271 },
    { 272, "res_272", 5,  kBindings_272 },
    { 273, "res_273", 5,  kBindings_273 },
    { 274, "res_274", 7,  kBindings_274 },
    { 275, "res_275", 15, kBindings_275 },
    { 276, "res_276", 5,  kBindings_276 },
    { 277, "res_277", 5,  kBindings_277 },
    { 278, "res_278", 5,  kBindings_278 },
    { 279, "res_279", 10, kBindings_279 },
};

static const ShaderAbiDef* getShaderAbi(uint32_t resId) {
    for (const auto& entry : kLsfg31ShaderAbi) {
        if (entry.resId == resId) return &entry;
    }
    return nullptr;
}


// =============================================================================
// REAL-LSFG 3.1 FUNCTIONAL EXTERNAL BACKEND IMPLEMENTATION
// 100-DISPATCH AUTHORITATIVE BASELINE
// =============================================================================

// -----------------------------------------------------------------------------
// Authoritative Extents & Dispatch Tables
// -----------------------------------------------------------------------------

struct Extent2D {
    uint32_t width;
    uint32_t height;
};

// Authoritative Gamma extents (levels 0..6):
// L0: 8x4, L1: 15x9, L2: 30x17, L3: 60x34, L4: 120x68, L5: 240x135, L6: 480x270
static const Extent2D kAuthoritativeGammaExtents[7] = {
    { 8, 4 },
    { 15, 9 },
    { 30, 17 },
    { 60, 34 },
    { 120, 68 },
    { 240, 135 },
    { 480, 270 }
};

// Authoritative Delta extents (levels 0..2):
// L0: 120x68, L1: 240x135, L2: 480x270
static const Extent2D kAuthoritativeDeltaExtents[3] = {
    { 120, 68 },
    { 240, 135 },
    { 480, 270 }
};

// -----------------------------------------------------------------------------
// ConstantBuffer Layout (48 bytes, std140 compliant)
// -----------------------------------------------------------------------------
struct ConstantBuffer {
    uint32_t inputOffset[2];
    uint32_t firstIter;
    uint32_t firstIterS;
    uint32_t advancedColorKind;
    uint32_t hdrSupport;
    float resolutionInvScale;
    float timestamp; // interpolationFactor
    float uiThreshold;
    uint32_t pad[3];
};

// Tracking structure for descriptor write completeness
struct DescriptorSetTracker {
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t expectedWriteCount = 0;
    uint32_t completedWriteCount = 0;
};

// Scratch images per slot (166 images total)
struct SlotScratchState {
    // Mipmaps out: 7 images (R8_UNORM)
    VkImage mipOutImgs[7]{};
    VkImageView mipOutViews[7]{};

    // Alpha scratch (7 levels):
    // tempImgs1 (2), tempImgs2 (2), tempImgs3 (4) = 8 per level -> 56 images (R8G8B8A8_UNORM)
    VkImage alphaTemp1[7][2]{};
    VkImageView alphaTempView1[7][2]{};
    VkImage alphaTemp2[7][2]{};
    VkImageView alphaTempView2[7][2]{};
    VkImage alphaTemp3[7][4]{};
    VkImageView alphaTempView3[7][4]{};

    // Beta scratch:
    // tempImgs1 (2), tempImgs2 (2) (R8G8B8A8_UNORM), outImgs (6) (R8_UNORM) = 10 images
    VkImage betaTemp1[2]{};
    VkImageView betaTempView1[2]{};
    VkImage betaTemp2[2]{};
    VkImageView betaTempView2[2]{};
    VkImage betaOut[6]{};
    VkImageView betaOutViews[6]{};

    // Gamma scratch (7 levels):
    // tempImgs1 (4), tempImgs2 (4) (R8G8B8A8_UNORM), outImg (1) (R16G16B16A16_SFLOAT) = 9 per level -> 63 images
    VkImage gammaTemp1[7][4]{};
    VkImageView gammaTempView1[7][4]{};
    VkImage gammaTemp2[7][4]{};
    VkImageView gammaTempView2[7][4]{};
    VkImage gammaOut[7]{};
    VkImageView gammaOutViews[7]{};

    // Delta scratch (3 levels):
    // tempImgs1 (4), tempImgs2 (4) (R8G8B8A8_UNORM), outImg1 (1), outImg2 (1) (R16G16B16A16_SFLOAT) = 10 per level -> 30 images
    VkImage deltaTemp1[3][4]{};
    VkImageView deltaTempView1[3][4]{};
    VkImage deltaTemp2[3][4]{};
    VkImageView deltaTempView2[3][4]{};
    VkImage deltaOut1[3]{};
    VkImageView deltaOutViews1[3]{};
    VkImage deltaOut2[3]{};
    VkImageView deltaOutViews2[3]{};

    // UBO backing buffer
    VkBuffer uboBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uboMemory = VK_NULL_HANDLE;
    void* uboMapped = nullptr;

    // 142 descriptor sets per slot
    VkDescriptorSet mipmapsSet = VK_NULL_HANDLE;               // 1 set
    VkDescriptorSet alphaSets[7][6]{};                         // 7 * 6 = 42 sets
    VkDescriptorSet betaSets[7]{};                             // 7 sets (3 for pass0, 4 for passes 1..4)
    VkDescriptorSet gammaSets[7][7]{};                        // 7 * 7 = 49 sets
    VkDescriptorSet deltaSets[3][14]{};                        // 3 * 14 = 42 sets
    VkDescriptorSet generateSet = VK_NULL_HANDLE;              // 1 set
    VkDescriptorSet generateSetR4A = VK_NULL_HANDLE;           // optional R4-A
    VkDescriptorSet generateSetR4B = VK_NULL_HANDLE;           // optional R4-B

    std::vector<DescriptorSetTracker> trackers;
};

// Global Alpha temporal state: 84 global images (3 banks * 4 images * 7 levels)
struct GlobalAlphaState {
    VkImage alphaGlobalOutImgs[7][3][4]{};
    VkImageView alphaGlobalOutViews[7][3][4]{};
};

struct LsfgExternalContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkExtent2D extent{0, 0};
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool isHdr = false;
    float flowScale = 1.0f;
    uint32_t generationCount = 1;

    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;

    // Device dispatch table
    PFN_vkCreateShaderModule createShaderModule = nullptr;
    PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout createPipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout destroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines createComputePipelines = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool createDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool destroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets freeDescriptorSets = nullptr;
    PFN_vkCreateSampler createSampler = nullptr;
    PFN_vkDestroySampler destroySampler = nullptr;
    PFN_vkCreateImage createImage = nullptr;
    PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements getImageMemoryRequirements = nullptr;
    PFN_vkCreateImageView createImageView = nullptr;
    PFN_vkDestroyImageView destroyImageView = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory allocateMemory = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;
    PFN_vkBindImageMemory bindImageMemory = nullptr;
    PFN_vkBindBufferMemory bindBufferMemory = nullptr;
    PFN_vkMapMemory mapMemory = nullptr;
    PFN_vkUnmapMemory unmapMemory = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkCmdClearColorImage cmdClearColorImage = nullptr;

    // Instance query functions
    PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties getPhysicalDeviceFormatProperties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties getPhysicalDeviceImageFormatProperties = nullptr;

    // Samplers: 3 standard samplers
    VkSampler sampler0 = VK_NULL_HANDLE; // Linear Clamp-to-edge
    VkSampler sampler1 = VK_NULL_HANDLE; // Clamp-to-border (White border)
    VkSampler sampler2 = VK_NULL_HANDLE; // Clamp-to-edge compare-always

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    std::unordered_map<uint32_t, ExternalPipelineInfo> pipelines;
    std::vector<LsfgEndpointSlot> slotEndpoints;
    VkImageView viewP = VK_NULL_HANDLE;

    // 84 global temporal Alpha images
    GlobalAlphaState globalAlpha;

    // Scratch states for 2 slots (166 images per slot -> 332 total scratch images)
    SlotScratchState slots[2];

    // Dummy fallback images (1x1 R16F and 1x1 R8)
    VkImage dummyImage = VK_NULL_HANDLE;
    VkImageView dummyImageView = VK_NULL_HANDLE;
    VkDeviceMemory dummyMemory = VK_NULL_HANDLE;

    VkImage dummyImageR8 = VK_NULL_HANDLE;
    VkImageView dummyImageViewR8 = VK_NULL_HANDLE;
    VkDeviceMemory dummyMemoryR8 = VK_NULL_HANDLE;

    // Allocated backing memory blocks
    std::vector<VkDeviceMemory> allocatedMemories;

    // Dynamic telemetry accounting
    uint64_t sumMemReqSize = 0;
    uint64_t sumBackingAllocationSize = 0;
    uint32_t backingAllocationCount = 0;
    uint32_t dedicatedAllocationCount = 0;

    // State machine & Chronology
    bool resourcesReady = false;
    bool initialLayoutsTransitioned = false;
    bool temporalBootstrapComplete = false;
    bool functionalReady = false;
    uint64_t algorithmFrameCount = 0;
    bool hasCommittedFrame = false;
    uint64_t lastCommittedIndex = 0;
};

// -----------------------------------------------------------------------------
// Suballocation Helper
// -----------------------------------------------------------------------------

struct SuballocationBlock {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize offset = 0;
    uint32_t memoryTypeIndex = 0;
};

static uint32_t findMemoryTypeIndex(
    const VkPhysicalDeviceMemoryProperties& memProps,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if (typeFilter & (1u << i)) {
            return i;
        }
    }
    return 0;
}

static VkResult allocateAndBindImage(
    LsfgExternalContext* ctx,
    const VkPhysicalDeviceMemoryProperties& memProps,
    std::vector<SuballocationBlock>& blocks,
    VkImage image)
{
    VkMemoryRequirements req{};
    ctx->getImageMemoryRequirements(ctx->device, image, &req);
    ctx->sumMemReqSize += req.size;

    uint32_t typeIndex = findMemoryTypeIndex(memProps, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Look for existing block with space and matching type
    for (auto& block : blocks) {
        if (block.memoryTypeIndex == typeIndex) {
            VkDeviceSize alignedOffset = (block.offset + req.alignment - 1) & ~(req.alignment - 1);
            if (alignedOffset + req.size <= block.size) {
                VkResult res = ctx->bindImageMemory(ctx->device, image, block.memory, alignedOffset);
                if (res == VK_SUCCESS) {
                    block.offset = alignedOffset + req.size;
                    return VK_SUCCESS;
                }
            }
        }
    }

    // Allocate new block (chunk size 64MB or req.size if larger)
    VkDeviceSize blockSize = (64ULL * 1024ULL * 1024ULL);
    if (req.size > blockSize) {
        blockSize = req.size;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = blockSize;
    allocInfo.memoryTypeIndex = typeIndex;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult res = ctx->allocateMemory(ctx->device, &allocInfo, nullptr, &memory);
    if (res != VK_SUCCESS) {
        return res;
    }

    ctx->allocatedMemories.push_back(memory);
    ctx->sumBackingAllocationSize += blockSize;
    ctx->backingAllocationCount++;

    res = ctx->bindImageMemory(ctx->device, image, memory, 0);
    if (res != VK_SUCCESS) {
        return res;
    }

    SuballocationBlock newBlock{};
    newBlock.memory = memory;
    newBlock.size = blockSize;
    newBlock.offset = req.size;
    newBlock.memoryTypeIndex = typeIndex;
    blocks.push_back(newBlock);

    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Image and View Helper
// -----------------------------------------------------------------------------

static VkResult create2DImage(
    LsfgExternalContext* ctx,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImage* outImage)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = { width, height, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return ctx->createImage(ctx->device, &info, nullptr, outImage);
}

static VkResult create2DImageView(
    LsfgExternalContext* ctx,
    VkImage image,
    VkFormat format,
    VkImageView* outView)
{
    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;
    return ctx->createImageView(ctx->device, &info, nullptr, outView);
}

// -----------------------------------------------------------------------------
// Descriptor Write Helper with Completeness Tracking
// -----------------------------------------------------------------------------

struct DescriptorWriteBatch {
    LsfgExternalContext* ctx = nullptr;
    VkDescriptorSet set = VK_NULL_HANDLE;
    DescriptorSetTracker* tracker = nullptr;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;

    void addBuffer(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
        bufferInfos.push_back({ buffer, offset, range });
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufferInfos.back();
        writes.push_back(write);
        if (tracker) tracker->completedWriteCount++;
    }

    void addSampler(uint32_t binding, VkSampler sampler) {
        imageInfos.push_back({ sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
        if (tracker) tracker->completedWriteCount++;
    }

    void addSampledImage(uint32_t binding, VkImageView view) {
        if (view == VK_NULL_HANDLE) view = ctx->dummyImageView;
        imageInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
        if (tracker) tracker->completedWriteCount++;
    }

    void addStorageImage(uint32_t binding, VkImageView view) {
        if (view == VK_NULL_HANDLE) view = ctx->dummyImageView;
        imageInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
        if (tracker) tracker->completedWriteCount++;
    }

    void apply() {
        if (!writes.empty()) {
            ctx->updateDescriptorSets(ctx->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }
};

// -----------------------------------------------------------------------------
// Context Creation: lsfg_create_context_external
// -----------------------------------------------------------------------------

LsfgExternalContextHandle lsfg_create_context_external(
    const LsfgExternalContextDesc* desc,
    const LsfgExternalEndpoints* endpoints,
    VkExtent2D extent,
    VkFormat format)
{
    if (desc == nullptr || endpoints == nullptr || desc->device == VK_NULL_HANDLE) {
        return nullptr;
    }

    auto *ctx = new LsfgExternalContext();
    ctx->instance = desc->instance;
    ctx->physicalDevice = desc->physicalDevice;
    ctx->device = desc->device;
    ctx->extent = extent;
    ctx->format = format;
    ctx->isHdr = desc->isHdr;
    ctx->flowScale = desc->flowScale > 0.0f ? desc->flowScale : 1.0f;
    ctx->generationCount = desc->generationCount > 0 ? desc->generationCount : 1;
    ctx->gipa = desc->getInstanceProcAddr;
    ctx->gdpa = desc->getDeviceProcAddr;

    if (ctx->gdpa == nullptr && ctx->gipa != nullptr) {
        ctx->gdpa = (PFN_vkGetDeviceProcAddr)ctx->gipa(ctx->instance, "vkGetDeviceProcAddr");
    }
    if (ctx->gdpa == nullptr) {
        delete ctx;
        return nullptr;
    }

    // Load device functions
    #define LOAD_DEV_FN(field, name) ctx->field = (PFN_vk##name)ctx->gdpa(ctx->device, "vk" #name)
    LOAD_DEV_FN(createShaderModule, CreateShaderModule);
    LOAD_DEV_FN(destroyShaderModule, DestroyShaderModule);
    LOAD_DEV_FN(createDescriptorSetLayout, CreateDescriptorSetLayout);
    LOAD_DEV_FN(destroyDescriptorSetLayout, DestroyDescriptorSetLayout);
    LOAD_DEV_FN(createPipelineLayout, CreatePipelineLayout);
    LOAD_DEV_FN(destroyPipelineLayout, DestroyPipelineLayout);
    LOAD_DEV_FN(createComputePipelines, CreateComputePipelines);
    LOAD_DEV_FN(destroyPipeline, DestroyPipeline);
    LOAD_DEV_FN(createDescriptorPool, CreateDescriptorPool);
    LOAD_DEV_FN(destroyDescriptorPool, DestroyDescriptorPool);
    LOAD_DEV_FN(allocateDescriptorSets, AllocateDescriptorSets);
    LOAD_DEV_FN(updateDescriptorSets, UpdateDescriptorSets);
    LOAD_DEV_FN(freeDescriptorSets, FreeDescriptorSets);
    LOAD_DEV_FN(createSampler, CreateSampler);
    LOAD_DEV_FN(destroySampler, DestroySampler);
    LOAD_DEV_FN(createImage, CreateImage);
    LOAD_DEV_FN(destroyImage, DestroyImage);
    LOAD_DEV_FN(getImageMemoryRequirements, GetImageMemoryRequirements);
    LOAD_DEV_FN(createImageView, CreateImageView);
    LOAD_DEV_FN(destroyImageView, DestroyImageView);
    LOAD_DEV_FN(createBuffer, CreateBuffer);
    LOAD_DEV_FN(destroyBuffer, DestroyBuffer);
    LOAD_DEV_FN(getBufferMemoryRequirements, GetBufferMemoryRequirements);
    LOAD_DEV_FN(allocateMemory, AllocateMemory);
    LOAD_DEV_FN(freeMemory, FreeMemory);
    LOAD_DEV_FN(bindImageMemory, BindImageMemory);
    LOAD_DEV_FN(bindBufferMemory, BindBufferMemory);
    LOAD_DEV_FN(mapMemory, MapMemory);
    LOAD_DEV_FN(unmapMemory, UnmapMemory);
    LOAD_DEV_FN(cmdBindPipeline, CmdBindPipeline);
    LOAD_DEV_FN(cmdBindDescriptorSets, CmdBindDescriptorSets);
    LOAD_DEV_FN(cmdDispatch, CmdDispatch);
    LOAD_DEV_FN(cmdPipelineBarrier, CmdPipelineBarrier);
    LOAD_DEV_FN(cmdUpdateBuffer, CmdUpdateBuffer);
    LOAD_DEV_FN(cmdClearColorImage, CmdClearColorImage);
    #undef LOAD_DEV_FN

    // Load instance query functions
    if (ctx->gipa != nullptr && ctx->instance != VK_NULL_HANDLE) {
        ctx->getPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)ctx->gipa(ctx->instance, "vkGetPhysicalDeviceMemoryProperties");
        ctx->getPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)ctx->gipa(ctx->instance, "vkGetPhysicalDeviceFormatProperties");
        ctx->getPhysicalDeviceImageFormatProperties = (PFN_vkGetPhysicalDeviceImageFormatProperties)ctx->gipa(ctx->instance, "vkGetPhysicalDeviceImageFormatProperties");
    }

    VkPhysicalDeviceMemoryProperties memProps{};
    if (ctx->getPhysicalDeviceMemoryProperties != nullptr && ctx->physicalDevice != VK_NULL_HANDLE) {
        ctx->getPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);
    }

    // 1. Create standard samplers: sampler0 (Linear Clamp), sampler1 (Border White), sampler2 (Compare Always)
    {
        VkSamplerCreateInfo sampInfo{};
        sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampInfo.magFilter = VK_FILTER_LINEAR;
        sampInfo.minFilter = VK_FILTER_LINEAR;
        sampInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.compareOp = VK_COMPARE_OP_NEVER;
        ctx->createSampler(ctx->device, &sampInfo, nullptr, &ctx->sampler0);

        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        ctx->createSampler(ctx->device, &sampInfo, nullptr, &ctx->sampler1);

        sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampInfo.compareEnable = VK_FALSE;
        sampInfo.compareOp = VK_COMPARE_OP_NEVER;
        ctx->createSampler(ctx->device, &sampInfo, nullptr, &ctx->sampler2);
    }

    // 2. Create Fallback Dummy Images (1x1 R16F and 1x1 R8)
    {
        create2DImage(ctx, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, &ctx->dummyImage);
        create2DImage(ctx, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, &ctx->dummyImageR8);

        std::vector<SuballocationBlock> dummyBlocks;
        allocateAndBindImage(ctx, memProps, dummyBlocks, ctx->dummyImage);
        allocateAndBindImage(ctx, memProps, dummyBlocks, ctx->dummyImageR8);

        create2DImageView(ctx, ctx->dummyImage, VK_FORMAT_R16G16B16A16_SFLOAT, &ctx->dummyImageView);
        create2DImageView(ctx, ctx->dummyImageR8, VK_FORMAT_R8G8B8A8_UNORM, &ctx->dummyImageViewR8);
    }

    // 3. Compile/Instantiate All 25 Shaders, Descriptor Set Layouts, Pipeline Layouts, Pipelines
    for (const auto& entry : kLsfg31ShaderAbi) {
        ExternalPipelineInfo pinfo{};
        pinfo.resId = entry.resId;

        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(entry.bindingCount);
        for (uint32_t b = 0; b < entry.bindingCount; ++b) {
            bindings[b].binding = entry.bindings[b].binding;
            bindings[b].descriptorType = entry.bindings[b].descriptorType;
            bindings[b].descriptorCount = entry.bindings[b].descriptorCount;
            bindings[b].stageFlags = entry.bindings[b].stageFlags;
            bindings[b].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo dslInfo{};
        dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        dslInfo.pBindings = bindings.data();
        if (ctx->createDescriptorSetLayout(ctx->device, &dslInfo, nullptr, &pinfo.descSetLayout) != VK_SUCCESS) {
            continue;
        }

        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &pinfo.descSetLayout;
        if (ctx->createPipelineLayout(ctx->device, &plInfo, nullptr, &pinfo.pipelineLayout) != VK_SUCCESS) {
            continue;
        }

        if (desc->shaderLoader) {
            auto spirv = desc->shaderLoader(entry.name);
            if (!spirv.empty() && (spirv.size() % 4 == 0)) {
                VkShaderModuleCreateInfo smInfo{};
                smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smInfo.codeSize = spirv.size();
                smInfo.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
                if (ctx->createShaderModule(ctx->device, &smInfo, nullptr, &pinfo.module) == VK_SUCCESS) {
                    VkComputePipelineCreateInfo cpInfo{};
                    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                    cpInfo.stage.module = pinfo.module;
                    cpInfo.stage.pName = "main";
                    cpInfo.layout = pinfo.pipelineLayout;
                    ctx->createComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pinfo.pipeline);
                }
            }
        }

        ctx->pipelines[entry.resId] = pinfo;
    }

    // 4. Create all 416 Algorithm Graph Images
    // (a) 84 global temporal Alpha images (7 levels * 3 banks * 4 images)
    std::vector<VkImage> allGraphImages;
    allGraphImages.reserve(416);

    // Alpha level dimensions derived from Mipmaps downscaled extents
    Extent2D mipExtents[7];
    Extent2D alphaHalfExtents[7];
    Extent2D alphaQuarterExtents[7];
    for (int lvl = 0; lvl < 7; ++lvl) {
        mipExtents[lvl].width = std::max(1u, (extent.width >> lvl));
        mipExtents[lvl].height = std::max(1u, (extent.height >> lvl));
        alphaHalfExtents[lvl].width = (mipExtents[lvl].width + 1) >> 1;
        alphaHalfExtents[lvl].height = (mipExtents[lvl].height + 1) >> 1;
        alphaQuarterExtents[lvl].width = (alphaHalfExtents[lvl].width + 1) >> 1;
        alphaQuarterExtents[lvl].height = (alphaHalfExtents[lvl].height + 1) >> 1;

        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 4; ++i) {
                create2DImage(ctx, alphaQuarterExtents[lvl].width, alphaQuarterExtents[lvl].height,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i]);
                allGraphImages.push_back(ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i]);
            }
        }
    }

    // (b) 166 scratch images per slot (332 total for 2 slots)
    for (int s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];

        // Mipmaps out (7)
        for (int i = 0; i < 7; ++i) {
            create2DImage(ctx, mipExtents[i].width, mipExtents[i].height,
                         VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.mipOutImgs[i]);
            allGraphImages.push_back(slot.mipOutImgs[i]);
        }

        // Alpha scratch (7 levels * 8 = 56)
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 2; ++i) {
                create2DImage(ctx, alphaHalfExtents[lvl].width, alphaHalfExtents[lvl].height,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.alphaTemp1[lvl][i]);
                allGraphImages.push_back(slot.alphaTemp1[lvl][i]);

                create2DImage(ctx, alphaHalfExtents[lvl].width, alphaHalfExtents[lvl].height,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.alphaTemp2[lvl][i]);
                allGraphImages.push_back(slot.alphaTemp2[lvl][i]);
            }
            for (int i = 0; i < 4; ++i) {
                create2DImage(ctx, alphaQuarterExtents[lvl].width, alphaQuarterExtents[lvl].height,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.alphaTemp3[lvl][i]);
                allGraphImages.push_back(slot.alphaTemp3[lvl][i]);
            }
        }

        // Beta scratch (10): temp1 (2), temp2 (2) RGBA8, out (6) R8
        Extent2D betaExtent = alphaQuarterExtents[0]; // 480x270 for 1080p
        for (int i = 0; i < 2; ++i) {
            create2DImage(ctx, betaExtent.width, betaExtent.height,
                         VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.betaTemp1[i]);
            allGraphImages.push_back(slot.betaTemp1[i]);

            create2DImage(ctx, betaExtent.width, betaExtent.height,
                         VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.betaTemp2[i]);
            allGraphImages.push_back(slot.betaTemp2[i]);
        }
        for (int i = 0; i < 6; ++i) {
            uint32_t bw = std::max(1u, (betaExtent.width >> i));
            uint32_t bh = std::max(1u, (betaExtent.height >> i));
            create2DImage(ctx, bw, bh,
                         VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.betaOut[i]);
            allGraphImages.push_back(slot.betaOut[i]);
        }

        // Gamma scratch (7 levels * 9 = 63): temp1 (4), temp2 (4) RGBA8, out (1) RGBA16F
        for (int lvl = 0; lvl < 7; ++lvl) {
            uint32_t gw = kAuthoritativeGammaExtents[lvl].width;
            uint32_t gh = kAuthoritativeGammaExtents[lvl].height;
            for (int i = 0; i < 4; ++i) {
                create2DImage(ctx, gw, gh,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.gammaTemp1[lvl][i]);
                allGraphImages.push_back(slot.gammaTemp1[lvl][i]);

                create2DImage(ctx, gw, gh,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.gammaTemp2[lvl][i]);
                allGraphImages.push_back(slot.gammaTemp2[lvl][i]);
            }
            create2DImage(ctx, gw, gh,
                         VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.gammaOut[lvl]);
            allGraphImages.push_back(slot.gammaOut[lvl]);
        }

        // Delta scratch (3 levels * 10 = 30): temp1 (4), temp2 (4) RGBA8, out1 (1), out2 (1) RGBA16F
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            uint32_t dw = kAuthoritativeDeltaExtents[dlvl].width;
            uint32_t dh = kAuthoritativeDeltaExtents[dlvl].height;
            for (int i = 0; i < 4; ++i) {
                create2DImage(ctx, dw, dh,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.deltaTemp1[dlvl][i]);
                allGraphImages.push_back(slot.deltaTemp1[dlvl][i]);

                create2DImage(ctx, dw, dh,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                             &slot.deltaTemp2[dlvl][i]);
                allGraphImages.push_back(slot.deltaTemp2[dlvl][i]);
            }
            create2DImage(ctx, dw, dh,
                         VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.deltaOut1[dlvl]);
            allGraphImages.push_back(slot.deltaOut1[dlvl]);

            create2DImage(ctx, dw, dh,
                         VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                         &slot.deltaOut2[dlvl]);
            allGraphImages.push_back(slot.deltaOut2[dlvl]);
        }
    }

    // 5. Backing Memory Suballocation for all 416 Graph Images
    std::vector<SuballocationBlock> graphBlocks;
    for (VkImage img : allGraphImages) {
        allocateAndBindImage(ctx, memProps, graphBlocks, img);
    }

    // 6. Create Image Views for all 416 Graph Images
    for (int lvl = 0; lvl < 7; ++lvl) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 4; ++i) {
                create2DImageView(ctx, ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i],
                                 VK_FORMAT_R8G8B8A8_UNORM, &ctx->globalAlpha.alphaGlobalOutViews[lvl][b][i]);
            }
        }
    }

    for (int s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];
        for (int i = 0; i < 7; ++i) {
            create2DImageView(ctx, slot.mipOutImgs[i], VK_FORMAT_R8_UNORM, &slot.mipOutViews[i]);
        }
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 2; ++i) {
                create2DImageView(ctx, slot.alphaTemp1[lvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.alphaTempView1[lvl][i]);
                create2DImageView(ctx, slot.alphaTemp2[lvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.alphaTempView2[lvl][i]);
            }
            for (int i = 0; i < 4; ++i) {
                create2DImageView(ctx, slot.alphaTemp3[lvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.alphaTempView3[lvl][i]);
            }
        }
        for (int i = 0; i < 2; ++i) {
            create2DImageView(ctx, slot.betaTemp1[i], VK_FORMAT_R8G8B8A8_UNORM, &slot.betaTempView1[i]);
            create2DImageView(ctx, slot.betaTemp2[i], VK_FORMAT_R8G8B8A8_UNORM, &slot.betaTempView2[i]);
        }
        for (int i = 0; i < 6; ++i) {
            create2DImageView(ctx, slot.betaOut[i], VK_FORMAT_R8_UNORM, &slot.betaOutViews[i]);
        }
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 4; ++i) {
                create2DImageView(ctx, slot.gammaTemp1[lvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.gammaTempView1[lvl][i]);
                create2DImageView(ctx, slot.gammaTemp2[lvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.gammaTempView2[lvl][i]);
            }
            create2DImageView(ctx, slot.gammaOut[lvl], VK_FORMAT_R16G16B16A16_SFLOAT, &slot.gammaOutViews[lvl]);
        }
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            for (int i = 0; i < 4; ++i) {
                create2DImageView(ctx, slot.deltaTemp1[dlvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.deltaTempView1[dlvl][i]);
                create2DImageView(ctx, slot.deltaTemp2[dlvl][i], VK_FORMAT_R8G8B8A8_UNORM, &slot.deltaTempView2[dlvl][i]);
            }
            create2DImageView(ctx, slot.deltaOut1[dlvl], VK_FORMAT_R16G16B16A16_SFLOAT, &slot.deltaOutViews1[dlvl]);
            create2DImageView(ctx, slot.deltaOut2[dlvl], VK_FORMAT_R16G16B16A16_SFLOAT, &slot.deltaOutViews2[dlvl]);
        }
    }

    // 7. Create UBO Buffers (Host Visible & Coherent)
    for (int s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = 65536;
        bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ctx->createBuffer(ctx->device, &bufInfo, nullptr, &slot.uboBuffer);

        VkMemoryRequirements req{};
        ctx->getBufferMemoryRequirements(ctx->device, slot.uboBuffer, &req);
        uint32_t hostType = findMemoryTypeIndex(memProps, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = req.size;
        allocInfo.memoryTypeIndex = hostType;
        ctx->allocateMemory(ctx->device, &allocInfo, nullptr, &slot.uboMemory);
        ctx->allocatedMemories.push_back(slot.uboMemory);
        ctx->bindBufferMemory(ctx->device, slot.uboBuffer, slot.uboMemory, 0);

        ctx->mapMemory(ctx->device, slot.uboMemory, 0, 65536, 0, &slot.uboMapped);

        // Populate ConstantBuffer records
        auto writeUbo = [&](uint32_t offset, float timestamp, bool firstIter, bool firstIterS) {
            ConstantBuffer cb{};
            cb.inputOffset[0] = 0;
            cb.inputOffset[1] = 0;
            cb.firstIter = firstIter ? 1U : 0U;
            cb.firstIterS = firstIterS ? 1U : 0U;
            cb.advancedColorKind = ctx->isHdr ? 2U : 0U;
            cb.hdrSupport = ctx->isHdr ? 1U : 0U;
            cb.resolutionInvScale = ctx->flowScale;
            cb.timestamp = timestamp; // interpolationFactor
            cb.uiThreshold = 0.5f;
            cb.pad[0] = 0;
            cb.pad[1] = 0;
            cb.pad[2] = 0;
            if (slot.uboMapped) {
                memcpy(reinterpret_cast<uint8_t*>(slot.uboMapped) + offset, &cb, sizeof(ConstantBuffer));
            }
        };

        writeUbo(0, 0.0f, false, false);          // Mipmaps: offset 0
        writeUbo(256, 0.5f, false, false);        // Beta: offset 256
        for (int lvl = 0; lvl < 7; ++lvl) {
            writeUbo(512 + lvl * 256, 0.5f, (lvl == 0), false); // Gamma lvl: offset 512 + lvl*256
        }
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            writeUbo(2304 + dlvl * 256, 0.5f, false, (dlvl == 0)); // Delta dlvl: offset 2304 + dlvl*256
        }
        writeUbo(3072, 0.5f, false, false);       // Generate: offset 3072
    }

    // 8. Create Descriptor Pool (sized for 284 sets, 1452 sampled images, etc.)
    {
        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 256;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSizes[1].descriptorCount = 512;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[2].descriptorCount = 1024;
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSizes[3].descriptorCount = 2048;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 512;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        ctx->createDescriptorPool(ctx->device, &poolInfo, nullptr, &ctx->descriptorPool);
    }

    // 9. Allocate & Update Exactly 142 Descriptor Sets per Slot (284 sets total)
    for (uint32_t s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];
        VkImageView viewC = (endpoints->slots && s < endpoints->slotCount) ? endpoints->slots[s].viewC : ctx->dummyImageView;
        VkImageView viewG = (endpoints->slots && s < endpoints->slotCount) ? endpoints->slots[s].viewG : ctx->dummyImageView;
        VkImageView viewP = endpoints->viewP ? endpoints->viewP : ctx->dummyImageView;

        auto allocSet = [&](uint32_t resId, uint32_t expectedWrites) -> std::pair<VkDescriptorSet, DescriptorSetTracker*> {
            VkDescriptorSet set = VK_NULL_HANDLE;
            auto it = ctx->pipelines.find(resId);
            if (it != ctx->pipelines.end() && it->second.descSetLayout != VK_NULL_HANDLE) {
                VkDescriptorSetAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                allocInfo.descriptorPool = ctx->descriptorPool;
                allocInfo.descriptorSetCount = 1;
                allocInfo.pSetLayouts = &it->second.descSetLayout;
                ctx->allocateDescriptorSets(ctx->device, &allocInfo, &set);
            }
            slot.trackers.push_back({ set, expectedWrites, 0 });
            return { set, &slot.trackers.back() };
        };

        // (a) Mipmaps (res 255, 10 bindings)
        {
            auto [set, tracker] = allocSet(255, 10);
            slot.mipmapsSet = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addBuffer(0, slot.uboBuffer, 0, sizeof(ConstantBuffer));
            batch.addSampler(1, ctx->sampler0);
            batch.addSampledImage(2, viewC);
            for (int i = 0; i < 7; ++i) {
                batch.addStorageImage(3 + i, slot.mipOutViews[i]);
            }
            batch.apply();
        }

        // (b) Alpha (7 levels * 6 sets = 42 sets)
        for (int lvl = 0; lvl < 7; ++lvl) {
            // alpha[0] (res 267, 4 bindings)
            {
                auto [set, tracker] = allocSet(267, 4);
                slot.alphaSets[lvl][0] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.mipOutViews[lvl]);
                batch.addStorageImage(2, slot.alphaTempView1[lvl][0]);
                batch.addStorageImage(3, slot.alphaTempView1[lvl][1]);
                batch.apply();
            }
            // alpha[1] (res 268, 5 bindings)
            {
                auto [set, tracker] = allocSet(268, 5);
                slot.alphaSets[lvl][1] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.alphaTempView1[lvl][0]);
                batch.addSampledImage(2, slot.alphaTempView1[lvl][1]);
                batch.addStorageImage(3, slot.alphaTempView2[lvl][0]);
                batch.addStorageImage(4, slot.alphaTempView2[lvl][1]);
                batch.apply();
            }
            // alpha[2] (res 269, 7 bindings)
            {
                auto [set, tracker] = allocSet(269, 7);
                slot.alphaSets[lvl][2] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.alphaTempView2[lvl][0]);
                batch.addSampledImage(2, slot.alphaTempView2[lvl][1]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(3 + i, slot.alphaTempView3[lvl][i]);
                batch.apply();
            }
            // alpha[3] across 3 banks (res 270, 9 bindings each)
            for (int b = 0; b < 3; ++b) {
                auto [set, tracker] = allocSet(270, 9);
                slot.alphaSets[lvl][3 + b] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, slot.alphaTempView3[lvl][i]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(5 + i, ctx->globalAlpha.alphaGlobalOutViews[lvl][b][i]);
                batch.apply();
            }
        }

        // (c) Beta (7 sets)
        // beta[0] across 3 banks (res 275, 15 bindings each)
        for (int b = 0; b < 3; ++b) {
            auto [set, tracker] = allocSet(275, 15);
            slot.betaSets[b] = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addSampler(0, ctx->sampler1);
            for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, ctx->globalAlpha.alphaGlobalOutViews[0][(b + 1) % 3][i]);
            for (int i = 0; i < 4; ++i) batch.addSampledImage(5 + i, ctx->globalAlpha.alphaGlobalOutViews[0][(b + 2) % 3][i]);
            for (int i = 0; i < 4; ++i) batch.addSampledImage(9 + i, ctx->globalAlpha.alphaGlobalOutViews[0][b % 3][i]);
            batch.addStorageImage(13, slot.betaTempView1[0]);
            batch.addStorageImage(14, slot.betaTempView1[1]);
            batch.apply();
        }
        // beta[1] (res 276, 5 bindings)
        {
            auto [set, tracker] = allocSet(276, 5);
            slot.betaSets[3] = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addSampler(0, ctx->sampler0);
            batch.addSampledImage(1, slot.betaTempView1[0]);
            batch.addSampledImage(2, slot.betaTempView1[1]);
            batch.addStorageImage(3, slot.betaTempView2[0]);
            batch.addStorageImage(4, slot.betaTempView2[1]);
            batch.apply();
        }
        // beta[2] (res 277, 5 bindings)
        {
            auto [set, tracker] = allocSet(277, 5);
            slot.betaSets[4] = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addSampler(0, ctx->sampler0);
            batch.addSampledImage(1, slot.betaTempView2[0]);
            batch.addSampledImage(2, slot.betaTempView2[1]);
            batch.addStorageImage(3, slot.betaTempView1[0]);
            batch.addStorageImage(4, slot.betaTempView1[1]);
            batch.apply();
        }
        // beta[3] (res 278, 5 bindings)
        {
            auto [set, tracker] = allocSet(278, 5);
            slot.betaSets[5] = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addSampler(0, ctx->sampler0);
            batch.addSampledImage(1, slot.betaTempView1[0]);
            batch.addSampledImage(2, slot.betaTempView1[1]);
            batch.addStorageImage(3, slot.betaTempView2[0]);
            batch.addStorageImage(4, slot.betaTempView2[1]);
            batch.apply();
        }
        // beta[4] (res 279, 10 bindings)
        {
            auto [set, tracker] = allocSet(279, 10);
            slot.betaSets[6] = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addBuffer(0, slot.uboBuffer, 256, sizeof(ConstantBuffer));
            batch.addSampler(1, ctx->sampler0);
            batch.addSampledImage(2, slot.betaTempView2[0]);
            batch.addSampledImage(3, slot.betaTempView2[1]);
            for (int i = 0; i < 6; ++i) batch.addStorageImage(4 + i, slot.betaOutViews[i]);
            batch.apply();
        }

        // (d) Gamma (7 levels * 7 sets = 49 sets)
        for (int lvl = 0; lvl < 7; ++lvl) {
            VkImageView prevGamma = (lvl == 0) ? ctx->dummyImageView : slot.gammaOutViews[lvl - 1];
            // gamma[0] across 3 banks (res 257, 15 bindings each)
            for (int b = 0; b < 3; ++b) {
                auto [set, tracker] = allocSet(257, 15);
                slot.gammaSets[lvl][b] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 512 + lvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler1);
                batch.addSampler(2, ctx->sampler2);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(3 + i, ctx->globalAlpha.alphaGlobalOutViews[6 - lvl][(b + 2) % 3][i]);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(7 + i, ctx->globalAlpha.alphaGlobalOutViews[6 - lvl][b % 3][i]);
                batch.addSampledImage(11, prevGamma);
                batch.addStorageImage(12, slot.gammaTempView1[lvl][0]);
                batch.addStorageImage(13, slot.gammaTempView1[lvl][1]);
                batch.addStorageImage(14, slot.gammaTempView1[lvl][2]);
                batch.apply();
            }
            // gamma[1] (res 259, 8 bindings)
            {
                auto [set, tracker] = allocSet(259, 8);
                slot.gammaSets[lvl][3] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.gammaTempView1[lvl][0]);
                batch.addSampledImage(2, slot.gammaTempView1[lvl][1]);
                batch.addSampledImage(3, slot.gammaTempView1[lvl][2]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(4 + i, slot.gammaTempView2[lvl][i]);
                batch.apply();
            }
            // gamma[2] (res 260, 9 bindings)
            {
                auto [set, tracker] = allocSet(260, 9);
                slot.gammaSets[lvl][4] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, slot.gammaTempView2[lvl][i]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(5 + i, slot.gammaTempView1[lvl][i]);
                batch.apply();
            }
            // gamma[3] (res 261, 9 bindings)
            {
                auto [set, tracker] = allocSet(261, 9);
                slot.gammaSets[lvl][5] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, slot.gammaTempView1[lvl][i]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(5 + i, slot.gammaTempView2[lvl][i]);
                batch.apply();
            }
            // gamma[4] (res 262, 10 bindings)
            {
                auto [set, tracker] = allocSet(262, 10);
                slot.gammaSets[lvl][6] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 512 + lvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler0);
                batch.addSampler(2, ctx->sampler2);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(3 + i, slot.gammaTempView2[lvl][i]);
                batch.addSampledImage(7, prevGamma);
                batch.addSampledImage(8, slot.betaOutViews[std::min(6 - lvl, 5)]);
                batch.addStorageImage(9, slot.gammaOutViews[lvl]);
                batch.apply();
            }
        }

        // (e) Delta (3 levels * 14 sets = 42 sets)
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            VkImageView prevGamma = slot.gammaOutViews[3 + dlvl];
            VkImageView prevDelta1 = (dlvl == 0) ? ctx->dummyImageView : slot.deltaOutViews1[dlvl - 1];
            VkImageView prevDelta2 = (dlvl == 0) ? ctx->dummyImageView : slot.deltaOutViews2[dlvl - 1];

            // delta[0] across 3 banks (res 257, 15 bindings each)
            for (int b = 0; b < 3; ++b) {
                auto [set, tracker] = allocSet(257, 15);
                slot.deltaSets[dlvl][b] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 2304 + dlvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler1);
                batch.addSampler(2, ctx->sampler2);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(3 + i, ctx->globalAlpha.alphaGlobalOutViews[2 - dlvl][(b + 2) % 3][i]);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(7 + i, ctx->globalAlpha.alphaGlobalOutViews[2 - dlvl][b % 3][i]);
                batch.addSampledImage(11, prevDelta1);
                batch.addStorageImage(12, slot.deltaTempView1[dlvl][0]);
                batch.addStorageImage(13, slot.deltaTempView1[dlvl][1]);
                batch.addStorageImage(14, slot.deltaTempView1[dlvl][2]);
                batch.apply();
            }
            // delta[1] (res 263, 8 bindings)
            {
                auto [set, tracker] = allocSet(263, 8);
                slot.deltaSets[dlvl][3] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.deltaTempView1[dlvl][0]);
                batch.addSampledImage(2, slot.deltaTempView1[dlvl][1]);
                batch.addSampledImage(3, slot.deltaTempView1[dlvl][2]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(4 + i, slot.deltaTempView2[dlvl][i]);
                batch.apply();
            }
            // delta[2] (res 264, 9 bindings)
            {
                auto [set, tracker] = allocSet(264, 9);
                slot.deltaSets[dlvl][4] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, slot.deltaTempView2[dlvl][i]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(5 + i, slot.deltaTempView1[dlvl][i]);
                batch.apply();
            }
            // delta[3] (res 265, 9 bindings)
            {
                auto [set, tracker] = allocSet(265, 9);
                slot.deltaSets[dlvl][5] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(1 + i, slot.deltaTempView1[dlvl][i]);
                for (int i = 0; i < 4; ++i) batch.addStorageImage(5 + i, slot.deltaTempView2[dlvl][i]);
                batch.apply();
            }
            // delta[4] (res 266, 10 bindings)
            {
                auto [set, tracker] = allocSet(266, 10);
                slot.deltaSets[dlvl][6] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 2304 + dlvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler0);
                batch.addSampler(2, ctx->sampler2);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(3 + i, slot.deltaTempView2[dlvl][i]);
                batch.addSampledImage(7, prevDelta1);
                batch.addSampledImage(8, slot.betaOutViews[2 - dlvl]);
                batch.addStorageImage(9, slot.deltaOutViews1[dlvl]);
                batch.apply();
            }
            // delta[5] across 3 banks (res 258, 15 bindings each)
            for (int b = 0; b < 3; ++b) {
                auto [set, tracker] = allocSet(258, 15);
                slot.deltaSets[dlvl][7 + b] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 2304 + dlvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler1);
                batch.addSampler(2, ctx->sampler2);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(3 + i, ctx->globalAlpha.alphaGlobalOutViews[2 - dlvl][(b + 2) % 3][i]);
                for (int i = 0; i < 4; ++i) batch.addSampledImage(7 + i, ctx->globalAlpha.alphaGlobalOutViews[2 - dlvl][b % 3][i]);
                batch.addSampledImage(11, prevGamma);
                batch.addSampledImage(12, prevDelta1);
                batch.addStorageImage(13, slot.deltaTempView2[dlvl][0]);
                batch.addStorageImage(14, slot.deltaTempView2[dlvl][1]);
                batch.apply();
            }
            // delta[6] (res 271, 5 bindings)
            {
                auto [set, tracker] = allocSet(271, 5);
                slot.deltaSets[dlvl][10] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.deltaTempView2[dlvl][0]);
                batch.addSampledImage(2, slot.deltaTempView2[dlvl][1]);
                batch.addStorageImage(3, slot.deltaTempView1[dlvl][0]);
                batch.addStorageImage(4, slot.deltaTempView1[dlvl][1]);
                batch.apply();
            }
            // delta[7] (res 272, 5 bindings)
            {
                auto [set, tracker] = allocSet(272, 5);
                slot.deltaSets[dlvl][11] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.deltaTempView1[dlvl][0]);
                batch.addSampledImage(2, slot.deltaTempView1[dlvl][1]);
                batch.addStorageImage(3, slot.deltaTempView2[dlvl][0]);
                batch.addStorageImage(4, slot.deltaTempView2[dlvl][1]);
                batch.apply();
            }
            // delta[8] (res 273, 5 bindings)
            {
                auto [set, tracker] = allocSet(273, 5);
                slot.deltaSets[dlvl][12] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addSampler(0, ctx->sampler0);
                batch.addSampledImage(1, slot.deltaTempView2[dlvl][0]);
                batch.addSampledImage(2, slot.deltaTempView2[dlvl][1]);
                batch.addStorageImage(3, slot.deltaTempView1[dlvl][0]);
                batch.addStorageImage(4, slot.deltaTempView1[dlvl][1]);
                batch.apply();
            }
            // delta[9] (res 274, 7 bindings)
            {
                auto [set, tracker] = allocSet(274, 7);
                slot.deltaSets[dlvl][13] = set;
                DescriptorWriteBatch batch{ ctx, set, tracker };
                batch.addBuffer(0, slot.uboBuffer, 2304 + dlvl * 256, sizeof(ConstantBuffer));
                batch.addSampler(1, ctx->sampler0);
                batch.addSampler(2, ctx->sampler2);
                batch.addSampledImage(3, slot.deltaTempView1[dlvl][0]);
                batch.addSampledImage(4, slot.deltaTempView1[dlvl][1]);
                batch.addSampledImage(5, prevDelta2);
                batch.addStorageImage(6, slot.deltaOutViews2[dlvl]);
                batch.apply();
            }
        }

        // (f) Generate pass (res 256, 9 bindings)
        {
            auto [set, tracker] = allocSet(256, 9);
            slot.generateSet = set;
            DescriptorWriteBatch batch{ ctx, set, tracker };
            batch.addBuffer(0, slot.uboBuffer, 3072, sizeof(ConstantBuffer));
            batch.addSampler(1, ctx->sampler0);
            batch.addSampler(2, ctx->sampler2);
            batch.addSampledImage(3, viewP);
            batch.addSampledImage(4, viewC);
            batch.addSampledImage(5, slot.gammaOutViews[6]);
            batch.addSampledImage(6, slot.deltaOutViews1[2]);
            batch.addSampledImage(7, slot.deltaOutViews2[2]);
            batch.addStorageImage(8, viewG);
            batch.apply();
        }

        // Optional R4-A and R4-B Generate sets
        slot.generateSetR4A = slot.generateSet;
        slot.generateSetR4B = slot.generateSet;

        // Verify write completeness for this slot
        for (const auto& tr : slot.trackers) {
            if (tr.completedWriteCount != tr.expectedWriteCount) {
                // If any write incomplete, resources are not ready
                delete ctx;
                return nullptr;
            }
        }
    }

    ctx->resourcesReady = true;
    ctx->functionalReady = true;
    return ctx;
}

// -----------------------------------------------------------------------------
// Destroy External Context: lsfg_destroy_context_external
// -----------------------------------------------------------------------------

void lsfg_destroy_context_external(LsfgExternalContextHandle ctx)
{
    if (ctx == nullptr) return;

    // Destroys all created images and views
    for (int lvl = 0; lvl < 7; ++lvl) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 4; ++i) {
                if (ctx->globalAlpha.alphaGlobalOutViews[lvl][b][i])
                    ctx->destroyImageView(ctx->device, ctx->globalAlpha.alphaGlobalOutViews[lvl][b][i], nullptr);
                if (ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i])
                    ctx->destroyImage(ctx->device, ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i], nullptr);
            }
        }
    }

    for (int s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];
        for (int i = 0; i < 7; ++i) {
            if (slot.mipOutViews[i]) ctx->destroyImageView(ctx->device, slot.mipOutViews[i], nullptr);
            if (slot.mipOutImgs[i]) ctx->destroyImage(ctx->device, slot.mipOutImgs[i], nullptr);
        }
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 2; ++i) {
                if (slot.alphaTempView1[lvl][i]) ctx->destroyImageView(ctx->device, slot.alphaTempView1[lvl][i], nullptr);
                if (slot.alphaTemp1[lvl][i]) ctx->destroyImage(ctx->device, slot.alphaTemp1[lvl][i], nullptr);
                if (slot.alphaTempView2[lvl][i]) ctx->destroyImageView(ctx->device, slot.alphaTempView2[lvl][i], nullptr);
                if (slot.alphaTemp2[lvl][i]) ctx->destroyImage(ctx->device, slot.alphaTemp2[lvl][i], nullptr);
            }
            for (int i = 0; i < 4; ++i) {
                if (slot.alphaTempView3[lvl][i]) ctx->destroyImageView(ctx->device, slot.alphaTempView3[lvl][i], nullptr);
                if (slot.alphaTemp3[lvl][i]) ctx->destroyImage(ctx->device, slot.alphaTemp3[lvl][i], nullptr);
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (slot.betaTempView1[i]) ctx->destroyImageView(ctx->device, slot.betaTempView1[i], nullptr);
            if (slot.betaTemp1[i]) ctx->destroyImage(ctx->device, slot.betaTemp1[i], nullptr);
            if (slot.betaTempView2[i]) ctx->destroyImageView(ctx->device, slot.betaTempView2[i], nullptr);
            if (slot.betaTemp2[i]) ctx->destroyImage(ctx->device, slot.betaTemp2[i], nullptr);
        }
        for (int i = 0; i < 6; ++i) {
            if (slot.betaOutViews[i]) ctx->destroyImageView(ctx->device, slot.betaOutViews[i], nullptr);
            if (slot.betaOut[i]) ctx->destroyImage(ctx->device, slot.betaOut[i], nullptr);
        }
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 4; ++i) {
                if (slot.gammaTempView1[lvl][i]) ctx->destroyImageView(ctx->device, slot.gammaTempView1[lvl][i], nullptr);
                if (slot.gammaTemp1[lvl][i]) ctx->destroyImage(ctx->device, slot.gammaTemp1[lvl][i], nullptr);
                if (slot.gammaTempView2[lvl][i]) ctx->destroyImageView(ctx->device, slot.gammaTempView2[lvl][i], nullptr);
                if (slot.gammaTemp2[lvl][i]) ctx->destroyImage(ctx->device, slot.gammaTemp2[lvl][i], nullptr);
            }
            if (slot.gammaOutViews[lvl]) ctx->destroyImageView(ctx->device, slot.gammaOutViews[lvl], nullptr);
            if (slot.gammaOut[lvl]) ctx->destroyImage(ctx->device, slot.gammaOut[lvl], nullptr);
        }
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            for (int i = 0; i < 4; ++i) {
                if (slot.deltaTempView1[dlvl][i]) ctx->destroyImageView(ctx->device, slot.deltaTempView1[dlvl][i], nullptr);
                if (slot.deltaTemp1[dlvl][i]) ctx->destroyImage(ctx->device, slot.deltaTemp1[dlvl][i], nullptr);
                if (slot.deltaTempView2[dlvl][i]) ctx->destroyImageView(ctx->device, slot.deltaTempView2[dlvl][i], nullptr);
                if (slot.deltaTemp2[dlvl][i]) ctx->destroyImage(ctx->device, slot.deltaTemp2[dlvl][i], nullptr);
            }
            if (slot.deltaOutViews1[dlvl]) ctx->destroyImageView(ctx->device, slot.deltaOutViews1[dlvl], nullptr);
            if (slot.deltaOut1[dlvl]) ctx->destroyImage(ctx->device, slot.deltaOut1[dlvl], nullptr);
            if (slot.deltaOutViews2[dlvl]) ctx->destroyImageView(ctx->device, slot.deltaOutViews2[dlvl], nullptr);
            if (slot.deltaOut2[dlvl]) ctx->destroyImage(ctx->device, slot.deltaOut2[dlvl], nullptr);
        }

        if (slot.uboMapped) {
            ctx->unmapMemory(ctx->device, slot.uboMemory);
            slot.uboMapped = nullptr;
        }
        if (slot.uboBuffer) ctx->destroyBuffer(ctx->device, slot.uboBuffer, nullptr);
    }

    if (ctx->dummyImageView) ctx->destroyImageView(ctx->device, ctx->dummyImageView, nullptr);
    if (ctx->dummyImage) ctx->destroyImage(ctx->device, ctx->dummyImage, nullptr);
    if (ctx->dummyImageViewR8) ctx->destroyImageView(ctx->device, ctx->dummyImageViewR8, nullptr);
    if (ctx->dummyImageR8) ctx->destroyImage(ctx->device, ctx->dummyImageR8, nullptr);

    for (VkDeviceMemory mem : ctx->allocatedMemories) {
        if (mem != VK_NULL_HANDLE) ctx->freeMemory(ctx->device, mem, nullptr);
    }

    if (ctx->descriptorPool) ctx->destroyDescriptorPool(ctx->device, ctx->descriptorPool, nullptr);
    if (ctx->sampler0) ctx->destroySampler(ctx->device, ctx->sampler0, nullptr);
    if (ctx->sampler1) ctx->destroySampler(ctx->device, ctx->sampler1, nullptr);
    if (ctx->sampler2) ctx->destroySampler(ctx->device, ctx->sampler2, nullptr);

    for (auto& pair : ctx->pipelines) {
        if (pair.second.pipeline != VK_NULL_HANDLE) ctx->destroyPipeline(ctx->device, pair.second.pipeline, nullptr);
        if (pair.second.pipelineLayout != VK_NULL_HANDLE) ctx->destroyPipelineLayout(ctx->device, pair.second.pipelineLayout, nullptr);
        if (pair.second.descSetLayout != VK_NULL_HANDLE) ctx->destroyDescriptorSetLayout(ctx->device, pair.second.descSetLayout, nullptr);
        if (pair.second.module != VK_NULL_HANDLE) ctx->destroyShaderModule(ctx->device, pair.second.module, nullptr);
    }

    delete ctx;
}

// -----------------------------------------------------------------------------
// Transition All 416 Graph Images + 2 Dummies from UNDEFINED to GENERAL
// -----------------------------------------------------------------------------

VkResult lsfg_record_initialize(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (ctx->cmdPipelineBarrier == nullptr || ctx->cmdClearColorImage == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (ctx->initialLayoutsTransitioned) return VK_SUCCESS;

    std::vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(418);

    auto addTransition = [&](VkImage img) {
        if (img == VK_NULL_HANDLE) return;
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = img;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barriers.push_back(barrier);
    };

    // Transition 84 global temporal Alpha images
    for (int lvl = 0; lvl < 7; ++lvl) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 4; ++i) {
                addTransition(ctx->globalAlpha.alphaGlobalOutImgs[lvl][b][i]);
            }
        }
    }

    // Transition 166 scratch images for both slots (332 images total)
    for (int s = 0; s < 2; ++s) {
        auto& slot = ctx->slots[s];
        for (int i = 0; i < 7; ++i) addTransition(slot.mipOutImgs[i]);
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 2; ++i) {
                addTransition(slot.alphaTemp1[lvl][i]);
                addTransition(slot.alphaTemp2[lvl][i]);
            }
            for (int i = 0; i < 4; ++i) addTransition(slot.alphaTemp3[lvl][i]);
        }
        for (int i = 0; i < 2; ++i) {
            addTransition(slot.betaTemp1[i]);
            addTransition(slot.betaTemp2[i]);
        }
        for (int i = 0; i < 6; ++i) addTransition(slot.betaOut[i]);
        for (int lvl = 0; lvl < 7; ++lvl) {
            for (int i = 0; i < 4; ++i) {
                addTransition(slot.gammaTemp1[lvl][i]);
                addTransition(slot.gammaTemp2[lvl][i]);
            }
            addTransition(slot.gammaOut[lvl]);
        }
        for (int dlvl = 0; dlvl < 3; ++dlvl) {
            for (int i = 0; i < 4; ++i) {
                addTransition(slot.deltaTemp1[dlvl][i]);
                addTransition(slot.deltaTemp2[dlvl][i]);
            }
            addTransition(slot.deltaOut1[dlvl]);
            addTransition(slot.deltaOut2[dlvl]);
        }
    }

    if (ctx->cmdPipelineBarrier && !barriers.empty()) {
        ctx->cmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            static_cast<uint32_t>(barriers.size()),
            barriers.data());
    }

    // Explicitly initialize fallback dummy images with defined neutral zero content
    if (ctx->dummyImage != VK_NULL_HANDLE || ctx->dummyImageR8 != VK_NULL_HANDLE) {
        VkImage dummyImages[2] = { ctx->dummyImage, ctx->dummyImageR8 };
        uint32_t dummyCount = 0;
        VkImage validDummies[2];
        for (int i = 0; i < 2; ++i) {
            if (dummyImages[i] != VK_NULL_HANDLE) validDummies[dummyCount++] = dummyImages[i];
        }

        if (dummyCount > 0 && ctx->cmdPipelineBarrier) {
            VkImageMemoryBarrier preClearBarriers[2]{};
            for (uint32_t i = 0; i < dummyCount; ++i) {
                preClearBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                preClearBarriers[i].srcAccessMask = 0;
                preClearBarriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                preClearBarriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                preClearBarriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                preClearBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preClearBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preClearBarriers[i].image = validDummies[i];
                preClearBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                preClearBarriers[i].subresourceRange.baseMipLevel = 0;
                preClearBarriers[i].subresourceRange.levelCount = 1;
                preClearBarriers[i].subresourceRange.baseArrayLayer = 0;
                preClearBarriers[i].subresourceRange.layerCount = 1;
            }
            ctx->cmdPipelineBarrier(
                cmdBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                dummyCount, preClearBarriers);

            if (ctx->cmdClearColorImage) {
                VkClearColorValue zeroColor{}; // 0.0f
                VkImageSubresourceRange range{};
                range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                range.baseMipLevel = 0;
                range.levelCount = 1;
                range.baseArrayLayer = 0;
                range.layerCount = 1;
                for (uint32_t i = 0; i < dummyCount; ++i) {
                    ctx->cmdClearColorImage(cmdBuffer, validDummies[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zeroColor, 1, &range);
                }
            }

            VkImageMemoryBarrier postClearBarriers[2]{};
            for (uint32_t i = 0; i < dummyCount; ++i) {
                postClearBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                postClearBarriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                postClearBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                postClearBarriers[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                postClearBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                postClearBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                postClearBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                postClearBarriers[i].image = validDummies[i];
                postClearBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                postClearBarriers[i].subresourceRange.baseMipLevel = 0;
                postClearBarriers[i].subresourceRange.levelCount = 1;
                postClearBarriers[i].subresourceRange.baseArrayLayer = 0;
                postClearBarriers[i].subresourceRange.layerCount = 1;
            }
            ctx->cmdPipelineBarrier(
                cmdBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                dummyCount, postClearBarriers);
        }
    }

    ctx->initialLayoutsTransitioned = true;
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Seed Mode: 43 Dispatches (1 Mipmaps + 42 Alpha with Pass-3 Replay)
// -----------------------------------------------------------------------------

VkResult lsfg_record_seed(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t algorithmFrameCount)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->initialLayoutsTransitioned) {
        VkResult initRes = lsfg_record_initialize(ctx, cmdBuffer);
        if (initRes != VK_SUCCESS || !ctx->initialLayoutsTransitioned) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    auto& slot = ctx->slots[slotIndex % 2];

    auto emitBarrier = [&]() {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ctx->cmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // 1. Mipmaps on N0 (1 dispatch)
    auto itMip = ctx->pipelines.find(255);
    if (itMip != ctx->pipelines.end() && itMip->second.pipeline != VK_NULL_HANDLE) {
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itMip->second.pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itMip->second.pipelineLayout, 0, 1, &slot.mipmapsSet, 0, nullptr);
        uint32_t gx = (ctx->extent.width + 63) / 64;
        uint32_t gy = (ctx->extent.height + 63) / 64;
        ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    }
    emitBarrier();

    // 2. Alpha passes across 7 levels (levels 6 down to 0)
    // Pass 0, 1, 2, and Pass 3 replayed for bank 0, 1, 2 (6 dispatches per level * 7 = 42 dispatches)
    uint32_t alphaResIds[4] = { 267, 268, 269, 270 };
    for (int i = 0; i < 7; ++i) {
        int lvl = 6 - i;
        uint32_t mw = std::max(1u, (ctx->extent.width >> lvl));
        uint32_t mh = std::max(1u, (ctx->extent.height >> lvl));
        uint32_t hw = (mw + 1) >> 1;
        uint32_t hh = (mh + 1) >> 1;
        uint32_t qw = (hw + 1) >> 1;
        uint32_t qh = (hh + 1) >> 1;

        // Pass 0: alpha[0]
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[267].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[267].pipelineLayout, 0, 1, &slot.alphaSets[lvl][0], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (hw + 7) >> 3, (hh + 7) >> 3, 1);
        emitBarrier();

        // Pass 1: alpha[1]
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[268].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[268].pipelineLayout, 0, 1, &slot.alphaSets[lvl][1], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (hw + 7) >> 3, (hh + 7) >> 3, 1);
        emitBarrier();

        // Pass 2: alpha[2]
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[269].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[269].pipelineLayout, 0, 1, &slot.alphaSets[lvl][2], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (qw + 7) >> 3, (qh + 7) >> 3, 1);
        emitBarrier();

        // Pass 3: alpha[3] replayed for all 3 banks: bank 0, bank 1, bank 2 (3 dispatches)
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[270].pipeline);
        for (int bank = 0; bank < 3; ++bank) {
            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[270].pipelineLayout, 0, 1, &slot.alphaSets[lvl][3 + bank], 0, nullptr);
            ctx->cmdDispatch(cmdBuffer, (qw + 7) >> 3, (qh + 7) >> 3, 1);
        }
        emitBarrier();
    }

    // 1 + 42 = 43 dispatches total
    ctx->temporalBootstrapComplete = true;
    ctx->algorithmFrameCount = algorithmFrameCount;
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// History-Only Mode: 29 Dispatches (1 Mipmaps + 28 Alpha)
// -----------------------------------------------------------------------------

VkResult lsfg_record_history_only(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t algorithmFrameCount)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    auto& slot = ctx->slots[slotIndex % 2];

    auto emitBarrier = [&]() {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ctx->cmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // 1. Mipmaps (1 dispatch)
    ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipeline);
    ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipelineLayout, 0, 1, &slot.mipmapsSet, 0, nullptr);
    uint32_t gx = (ctx->extent.width + 63) / 64;
    uint32_t gy = (ctx->extent.height + 63) / 64;
    ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    emitBarrier();

    // 2. Alpha passes across 7 levels (4 dispatches per level * 7 = 28 dispatches)
    uint32_t targetBank = algorithmFrameCount % 3;
    for (int i = 0; i < 7; ++i) {
        int lvl = 6 - i;
        uint32_t mw = std::max(1u, (ctx->extent.width >> lvl));
        uint32_t mh = std::max(1u, (ctx->extent.height >> lvl));
        uint32_t hw = (mw + 1) >> 1;
        uint32_t hh = (mh + 1) >> 1;
        uint32_t qw = (hw + 1) >> 1;
        uint32_t qh = (hh + 1) >> 1;

        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[267].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[267].pipelineLayout, 0, 1, &slot.alphaSets[lvl][0], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (hw + 7) >> 3, (hh + 7) >> 3, 1);
        emitBarrier();

        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[268].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[268].pipelineLayout, 0, 1, &slot.alphaSets[lvl][1], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (hw + 7) >> 3, (hh + 7) >> 3, 1);
        emitBarrier();

        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[269].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[269].pipelineLayout, 0, 1, &slot.alphaSets[lvl][2], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (qw + 7) >> 3, (qh + 7) >> 3, 1);
        emitBarrier();

        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[270].pipeline);
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[270].pipelineLayout, 0, 1, &slot.alphaSets[lvl][3 + targetBank], 0, nullptr);
        ctx->cmdDispatch(cmdBuffer, (qw + 7) >> 3, (qh + 7) >> 3, 1);
        emitBarrier();
    }

    ctx->algorithmFrameCount = algorithmFrameCount;
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// True X3 Split Execution API Implementation
// -----------------------------------------------------------------------------

void lsfg_update_timestamp(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    float interpolationFactor)
{
    if (ctx == nullptr || ctx->cmdUpdateBuffer == nullptr || cmdBuffer == VK_NULL_HANDLE) return;
    auto& slot = ctx->slots[slotIndex % 2];
    float t = interpolationFactor;

    // Pre-barrier: Ensure any previous compute reads or transfers to uboBuffer are complete before writing again
    VkBufferMemoryBarrier preBmb{};
    preBmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    preBmb.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    preBmb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    preBmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preBmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    preBmb.buffer = slot.uboBuffer;
    preBmb.offset = 0;
    preBmb.size = VK_WHOLE_SIZE;

    ctx->cmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 1, &preBmb, 0, nullptr);

    // Gamma passes (7 levels: offset = 512 + lvl * 256 + 28)
    for (int lvl = 0; lvl < 7; ++lvl) {
        ctx->cmdUpdateBuffer(cmdBuffer, slot.uboBuffer, 512 + lvl * 256 + 28, sizeof(float), &t);
    }
    // Delta passes (3 levels: offset = 2304 + dlvl * 256 + 28)
    for (int dlvl = 0; dlvl < 3; ++dlvl) {
        ctx->cmdUpdateBuffer(cmdBuffer, slot.uboBuffer, 2304 + dlvl * 256 + 28, sizeof(float), &t);
    }
    // Generate pass (offset = 3072 + 28)
    ctx->cmdUpdateBuffer(cmdBuffer, slot.uboBuffer, 3072 + 28, sizeof(float), &t);

    // Barrier: TRANSFER_WRITE -> COMPUTE_SHADER_READ / UNIFORM_READ
    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = slot.uboBuffer;
    bmb.offset = 0;
    bmb.size = VK_WHOLE_SIZE;

    ctx->cmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 0, nullptr, 1, &bmb, 0, nullptr);
}

VkResult lsfg_record_shared_stages(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->temporalBootstrapComplete) {
        return VK_ERROR_NOT_PERMITTED_EXT;
    }

    auto& slot = ctx->slots[slotIndex % 2];

    if (ctx->hasCommittedFrame) {
        VkMemoryBarrier crossFrameBarrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        ctx->cmdPipelineBarrier(cmdBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &crossFrameBarrier, 0, nullptr, 0, nullptr);
    }

    auto emitComputeBarrier = [&]() {
        VkMemoryBarrier cb{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        ctx->cmdPipelineBarrier(cmdBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &cb, 0, nullptr, 0, nullptr);
    };

    uint32_t bank = proposedFrameIndex % 3;

    // Stage 1: Mipmaps pass (Res 255, 1 dispatch)
    ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipeline);
    ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipelineLayout, 0, 1, &slot.mipmapsSet, 0, nullptr);
    uint32_t gx = (ctx->extent.width + 63) / 64;
    uint32_t gy = (ctx->extent.height + 63) / 64;
    ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    emitComputeBarrier();

    // Stage 2: Alpha passes (7 levels x 4 passes = 28 dispatches)
    uint32_t alphaResIds[4] = { 267, 268, 269, 270 };
    for (int lvl = 0; lvl < 7; ++lvl) {
        uint32_t mw = std::max(1u, (ctx->extent.width >> lvl));
        uint32_t mh = std::max(1u, (ctx->extent.height >> lvl));
        uint32_t hw = (mw + 1) >> 1;
        uint32_t hh = (mh + 1) >> 1;
        uint32_t qw = (hw + 1) >> 1;
        uint32_t qh = (hh + 1) >> 1;
        for (int p = 0; p < 4; ++p) {
            uint32_t resId = alphaResIds[p];
            auto itA = ctx->pipelines.find(resId);
            if (itA != ctx->pipelines.end() && itA->second.pipeline != VK_NULL_HANDLE) {
                ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itA->second.pipeline);
                VkDescriptorSet aSet = (p < 3) ? slot.alphaSets[lvl][p] : slot.alphaSets[lvl][3 + bank];
                ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itA->second.pipelineLayout, 0, 1, &aSet, 0, nullptr);
                uint32_t dw = (p < 2) ? hw : qw;
                uint32_t dh = (p < 2) ? hh : qh;
                ctx->cmdDispatch(cmdBuffer, (dw + 7) >> 3, (dh + 7) >> 3, 1);
            }
            emitComputeBarrier();
        }
    }

    // Stage 3: Beta passes (5 passes = 5 dispatches)
    uint32_t betaResIds[5] = { 275, 276, 277, 278, 279 };
    for (int p = 0; p < 5; ++p) {
        uint32_t resId = betaResIds[p];
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
        VkDescriptorSet bSet = (p == 0) ? slot.betaSets[bank] : slot.betaSets[2 + p];
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &bSet, 0, nullptr);

        uint32_t bw = std::max(1u, (ctx->extent.width + 3) / 4);
        uint32_t bh = std::max(1u, (ctx->extent.height + 3) / 4);
        uint32_t bs = (p == 4) ? 32 : 8;
        ctx->cmdDispatch(cmdBuffer, (bw + bs - 1) / bs, (bh + bs - 1) / bs, 1);
        emitComputeBarrier();
    }

    return VK_SUCCESS;
}

VkResult lsfg_record_branch_stages(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    auto& slot = ctx->slots[slotIndex % 2];

    auto emitComputeBarrier = [&]() {
        VkMemoryBarrier cb{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        ctx->cmdPipelineBarrier(cmdBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &cb, 0, nullptr, 0, nullptr);
    };

    uint32_t bank = proposedFrameIndex % 3;

    // Update UBO timestamp for this branch
    lsfg_update_timestamp(ctx, cmdBuffer, slotIndex, interpolationFactor);

    // Stage 4: Gamma passes (7 levels x 5 passes = 35 dispatches)
    uint32_t gammaResIds[5] = { 257, 259, 260, 261, 262 };
    for (int lvl = 0; lvl < 7; ++lvl) {
        const auto& ext = kAuthoritativeGammaExtents[lvl];
        uint32_t threadsX = (ext.width + 7) >> 3;
        uint32_t threadsY = (ext.height + 7) >> 3;

        for (int p = 0; p < 5; ++p) {
            uint32_t resId = gammaResIds[p];
            ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
            VkDescriptorSet gSet = (p == 0) ? slot.gammaSets[lvl][bank] : slot.gammaSets[lvl][2 + p];
            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &gSet, 0, nullptr);
            ctx->cmdDispatch(cmdBuffer, threadsX, threadsY, 1);
            emitComputeBarrier();
        }
    }

    // Stage 5: Delta passes (3 levels x 10 passes = 30 dispatches)
    uint32_t deltaResIds[10] = { 257, 263, 264, 265, 266, 258, 271, 272, 273, 274 };
    for (int dlvl = 0; dlvl < 3; ++dlvl) {
        const auto& ext = kAuthoritativeDeltaExtents[dlvl];
        uint32_t threadsX = (ext.width + 7) >> 3;
        uint32_t threadsY = (ext.height + 7) >> 3;

        for (int p = 0; p < 10; ++p) {
            uint32_t resId = deltaResIds[p];
            ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
            VkDescriptorSet dSet = VK_NULL_HANDLE;
            if (p == 0) dSet = slot.deltaSets[dlvl][bank];
            else if (p < 5) dSet = slot.deltaSets[dlvl][2 + p];
            else if (p == 5) dSet = slot.deltaSets[dlvl][7 + bank];
            else dSet = slot.deltaSets[dlvl][4 + p];

            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &dSet, 0, nullptr);
            ctx->cmdDispatch(cmdBuffer, threadsX, threadsY, 1);
            emitComputeBarrier();
        }
    }

    // Stage 6: Generate pass (Res 256, 1 dispatch)
    ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[256].pipeline);
    VkDescriptorSet genSet = slot.generateSet;
    ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[256].pipelineLayout, 0, 1, &genSet, 0, nullptr);
    uint32_t gxGen = (ctx->extent.width + 15) / 16;
    uint32_t gyGen = (ctx->extent.height + 15) / 16;
    ctx->cmdDispatch(cmdBuffer, gxGen > 0 ? gxGen : 1, gyGen > 0 ? gyGen : 1, 1);

    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Unified Internal Generation Recording (100 Dispatches Baseline)
// -----------------------------------------------------------------------------

// Interposer-only diagnostic hook: the X2 monolithic recorder hides the
// shared-to-M1 boundary. No public LSFG ABI or productive work is changed.
struct LsfgCostProfilerHook {
    VkQueryPool queryPool = VK_NULL_HANDLE;
    PFN_vkCmdWriteTimestamp writeTimestamp = nullptr;
    uint32_t sharedEndQuery = 0;
    uint32_t m1BeginQuery = 0;
};

static VkResult record_generation_internal(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    bool profilingEnabled,
    VkQueryPool queryPool,
    uint32_t queryBase,
    PFN_vkCmdWriteTimestamp cmdWriteTimestamp,
    int pMode,
    bool deltaL2Bypass = false,
    bool gammaL6Bypass = false,
    const LsfgCostProfilerHook* costHook = nullptr)
{
    if (ctx == nullptr || cmdBuffer == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (!ctx->temporalBootstrapComplete) {
        // Contract L: Full generation guarded by valid temporal history state
        return VK_ERROR_NOT_PERMITTED_EXT;
    }

    auto& slot = ctx->slots[slotIndex % 2];

    auto emitTimestamp = [&](uint32_t queryOffset) {
        if (profilingEnabled && queryPool != VK_NULL_HANDLE && cmdWriteTimestamp != nullptr) {
            cmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              queryPool, queryBase + queryOffset);
        }
    };

    if (ctx->hasCommittedFrame) {
        VkMemoryBarrier crossFrameBarrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        ctx->cmdPipelineBarrier(cmdBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &crossFrameBarrier, 0, nullptr, 0, nullptr);
    }

    auto emitComputeBarrier = [&]() {
        VkMemoryBarrier cb{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        };
        ctx->cmdPipelineBarrier(cmdBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &cb, 0, nullptr, 0, nullptr);
    };

    uint32_t bank = proposedFrameIndex % 3;

    // Q0: Beginning of compute sequence
    emitTimestamp(0);

    // Stage 1: Mipmaps pass (Res 255, 1 dispatch)
    ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipeline);
    ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[255].pipelineLayout, 0, 1, &slot.mipmapsSet, 0, nullptr);
    uint32_t gx = (ctx->extent.width + 63) / 64;
    uint32_t gy = (ctx->extent.height + 63) / 64;
    ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    emitComputeBarrier();
    emitTimestamp(1);

    // Stage 2: Alpha passes (7 levels x 4 passes = 28 dispatches)
    uint32_t alphaResIds[4] = { 267, 268, 269, 270 };
    for (int lvl = 0; lvl < 7; ++lvl) {
        uint32_t mw = std::max(1u, (ctx->extent.width >> lvl));
        uint32_t mh = std::max(1u, (ctx->extent.height >> lvl));
        uint32_t hw = (mw + 1) >> 1;
        uint32_t hh = (mh + 1) >> 1;
        uint32_t qw = (hw + 1) >> 1;
        uint32_t qh = (hh + 1) >> 1;
        for (int p = 0; p < 4; ++p) {
            uint32_t resId = alphaResIds[p];
            auto itA = ctx->pipelines.find(resId);
            if (itA != ctx->pipelines.end() && itA->second.pipeline != VK_NULL_HANDLE) {
                ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itA->second.pipeline);
                VkDescriptorSet aSet = (p < 3) ? slot.alphaSets[lvl][p] : slot.alphaSets[lvl][3 + bank];
                ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itA->second.pipelineLayout, 0, 1, &aSet, 0, nullptr);
                uint32_t dw = (p < 2) ? hw : qw;
                uint32_t dh = (p < 2) ? hh : qh;
                ctx->cmdDispatch(cmdBuffer, (dw + 7) >> 3, (dh + 7) >> 3, 1);
            }
            emitComputeBarrier();
            if (pMode == 1) {
                emitTimestamp(2 + lvl * 4 + p);
            }
        }
    }
    if (pMode == 0) {
        emitTimestamp(2);
    }

    // Stage 3: Beta passes (5 passes = 5 dispatches)
    uint32_t betaResIds[5] = { 275, 276, 277, 278, 279 };
    for (int p = 0; p < 5; ++p) {
        uint32_t resId = betaResIds[p];
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
        VkDescriptorSet bSet = (p == 0) ? slot.betaSets[bank] : slot.betaSets[2 + p];
        ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &bSet, 0, nullptr);

        uint32_t bw = std::max(1u, (ctx->extent.width + 3) / 4);
        uint32_t bh = std::max(1u, (ctx->extent.height + 3) / 4);
        uint32_t bs = (p == 4) ? 32 : 8;
        ctx->cmdDispatch(cmdBuffer, (bw + bs - 1) / bs, (bh + bs - 1) / bs, 1);
        emitComputeBarrier();
        if (pMode == 1) emitTimestamp(30 + p);
    }
    if (pMode == 0) emitTimestamp(3);

    // Beta's final compute barrier precedes the UBO update and all M1 work.
    // This exact boundary is invisible to the interposer in monolithic X2.
    if (costHook != nullptr && costHook->queryPool != VK_NULL_HANDLE &&
        costHook->writeTimestamp != nullptr) {
        costHook->writeTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 costHook->queryPool, costHook->sharedEndQuery);
        costHook->writeTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 costHook->queryPool, costHook->m1BeginQuery);
    }

    // Apply requested interpolation factor to UBO before consumers (Gamma, Delta, Generate)
    lsfg_update_timestamp(ctx, cmdBuffer, slotIndex, interpolationFactor);

    // Stage 4: Gamma passes (7 levels x 5 passes = 35 dispatches)
    uint32_t gammaResIds[5] = { 257, 259, 260, 261, 262 };
    for (int lvl = 0; lvl < 7; ++lvl) {
        if (gammaL6Bypass && lvl == 6) {
            if (pMode == 1) {
                for (int p = 0; p < 5; ++p) {
                    emitTimestamp(35 + 6 * 5 + p); // Q65..Q69 bypass markers
                }
            }
            break;
        }

        const auto& ext = kAuthoritativeGammaExtents[lvl];
        uint32_t threadsX = (ext.width + 7) >> 3;
        uint32_t threadsY = (ext.height + 7) >> 3;

        for (int p = 0; p < 5; ++p) {
            uint32_t resId = gammaResIds[p];
            ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
            VkDescriptorSet gSet = (p == 0) ? slot.gammaSets[lvl][bank] : slot.gammaSets[lvl][2 + p];
            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &gSet, 0, nullptr);
            ctx->cmdDispatch(cmdBuffer, threadsX, threadsY, 1);
            emitComputeBarrier();
            if (pMode == 1) emitTimestamp(35 + lvl * 5 + p);
        }
    }
    if (pMode == 0) emitTimestamp(4);

    // Stage 5: Delta passes (3 levels x 10 passes = 30 dispatches)
    uint32_t deltaResIds[10] = { 257, 263, 264, 265, 266, 258, 271, 272, 273, 274 };
    for (int dlvl = 0; dlvl < 3; ++dlvl) {
        if (deltaL2Bypass && dlvl == 2) {
            if (pMode == 1) {
                for (int p = 0; p < 10; ++p) {
                    emitTimestamp(70 + 2 * 10 + p); // Q90..Q99 bypass markers
                }
            }
            break;
        }

        const auto& ext = kAuthoritativeDeltaExtents[dlvl];
        uint32_t threadsX = (ext.width + 7) >> 3;
        uint32_t threadsY = (ext.height + 7) >> 3;

        for (int p = 0; p < 10; ++p) {
            uint32_t resId = deltaResIds[p];
            ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipeline);
            VkDescriptorSet dSet = VK_NULL_HANDLE;
            if (p == 0) dSet = slot.deltaSets[dlvl][bank];
            else if (p < 5) dSet = slot.deltaSets[dlvl][2 + p];
            else if (p == 5) dSet = slot.deltaSets[dlvl][7 + bank];
            else dSet = slot.deltaSets[dlvl][4 + p];

            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[resId].pipelineLayout, 0, 1, &dSet, 0, nullptr);
            ctx->cmdDispatch(cmdBuffer, threadsX, threadsY, 1);
            emitComputeBarrier();
            if (pMode == 1) emitTimestamp(70 + dlvl * 10 + p);
        }
    }
    if (pMode == 0) emitTimestamp(5);

    // Stage 6: Generate pass (Res 256, 1 dispatch)
    ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[256].pipeline);
    VkDescriptorSet genSet = slot.generateSet;
    if (gammaL6Bypass && deltaL2Bypass && slot.generateSetR4B != VK_NULL_HANDLE) {
        genSet = slot.generateSetR4B;
    } else if (deltaL2Bypass && slot.generateSetR4A != VK_NULL_HANDLE) {
        genSet = slot.generateSetR4A;
    }
    ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[256].pipelineLayout, 0, 1, &genSet, 0, nullptr);
    uint32_t gxGen = (ctx->extent.width + 15) / 16;
    uint32_t gyGen = (ctx->extent.height + 15) / 16;
    ctx->cmdDispatch(cmdBuffer, gxGen > 0 ? gxGen : 1, gyGen > 0 ? gyGen : 1, 1);

    if (pMode == 1) {
        emitTimestamp(100); // Q100: R3/R4 max compute query index
    } else {
        emitTimestamp(6);   // Q6:   R2 max compute query index
    }

    ctx->algorithmFrameCount = proposedFrameIndex;
    return VK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Public Generation Entry Points
// -----------------------------------------------------------------------------

VkResult lsfg_record_generation(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor)
{
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex, interpolationFactor,
                                      false, VK_NULL_HANDLE, 0, nullptr, 0);
}

VkResult lsfg_record_generation_profiled(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    const LsfgStageProfilingInfo* profiling)
{
    bool active = (profiling != nullptr && profiling->enabled &&
                   profiling->queryPool != VK_NULL_HANDLE &&
                   profiling->cmdWriteTimestamp != nullptr);
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex, interpolationFactor,
                                      active,
                                      active ? profiling->queryPool : VK_NULL_HANDLE,
                                      active ? profiling->queryBase : 0,
                                      active ? profiling->cmdWriteTimestamp : nullptr,
                                      0 /* pMode=0: R2, Q0..Q6 */);
}

VkResult lsfg_record_generation_profiled_r3(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    const LsfgDispatchProfilingInfo* profiling)
{
    bool active = (profiling != nullptr && profiling->enabled &&
                   profiling->queryPool != VK_NULL_HANDLE &&
                   profiling->cmdWriteTimestamp != nullptr);
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex, interpolationFactor,
                                      active,
                                      active ? profiling->queryPool : VK_NULL_HANDLE,
                                      active ? profiling->queryBase : 0,
                                      active ? profiling->cmdWriteTimestamp : nullptr,
                                      1 /* pMode=1: R3, Q0..Q100 */,
                                      false, false);
}

// Diagnostic-only entry point available to the interposer because it includes
// this source in the same translation unit. The public LSFG interface is intact.
static VkResult lsfg_record_generation_cost_profiled(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    const LsfgCostProfilerHook* hook)
{
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex,
                                      interpolationFactor, false, VK_NULL_HANDLE,
                                      0, nullptr, 0, false, false, hook);
}

VkResult lsfg_record_generation_profiled_r4a(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    const LsfgR4AOptions* options)
{
    bool bypassDeltaL2 = (options != nullptr && options->enabled && options->deltaL2Bypass);
    const LsfgDispatchProfilingInfo* profiling = options ? options->profiling : nullptr;
    bool active = (profiling != nullptr && profiling->enabled &&
                   profiling->queryPool != VK_NULL_HANDLE &&
                   profiling->cmdWriteTimestamp != nullptr);
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex, interpolationFactor,
                                      active,
                                      active ? profiling->queryPool : VK_NULL_HANDLE,
                                      active ? profiling->queryBase : 0,
                                      active ? profiling->cmdWriteTimestamp : nullptr,
                                      1, bypassDeltaL2, false);
}

VkResult lsfg_record_generation_profiled_r4b(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor,
    const LsfgR4BOptions* options)
{
    bool bypassDeltaL2 = (options != nullptr && options->enabled && options->deltaL2Bypass);
    bool bypassGammaL6 = (options != nullptr && options->enabled && options->gammaL6Bypass);
    const LsfgDispatchProfilingInfo* profiling = options ? options->profiling : nullptr;
    bool active = (profiling != nullptr && profiling->enabled &&
                   profiling->queryPool != VK_NULL_HANDLE &&
                   profiling->cmdWriteTimestamp != nullptr);
    return record_generation_internal(ctx, cmdBuffer, slotIndex, proposedFrameIndex, interpolationFactor,
                                      active,
                                      active ? profiling->queryPool : VK_NULL_HANDLE,
                                      active ? profiling->queryBase : 0,
                                      active ? profiling->cmdWriteTimestamp : nullptr,
                                      1, bypassDeltaL2, bypassGammaL6);
}

void lsfg_commit_generation(
    LsfgExternalContextHandle ctx,
    uint64_t committedFrameIndex)
{
    if (ctx == nullptr) return;
    ctx->hasCommittedFrame = true;
    ctx->lastCommittedIndex = committedFrameIndex;
}

} // namespace LSFG_3_1
