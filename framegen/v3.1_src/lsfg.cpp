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

    vkDeviceWaitIdle(device->device.handle());
    contexts.erase(it);
}

void LSFG_3_1::finalize() {
    if (!instance.has_value() || !device.has_value())
        return;

    vkDeviceWaitIdle(device->device.handle());
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
    vkDeviceWaitIdle(device->device.handle());
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

struct LsfgExternalContext {
    VkDevice device = VK_NULL_HANDLE;
    VkExtent2D extent{0, 0};
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool isHdr = false;
    float flowScale = 1.0f;
    uint32_t generationCount = 1;

    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;

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
    PFN_vkCreateSampler createSampler = nullptr;
    PFN_vkDestroySampler destroySampler = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;

    VkSampler linearSampler = VK_NULL_HANDLE;
    VkSampler clampSampler = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    std::unordered_map<uint32_t, ExternalPipelineInfo> pipelines;
    std::vector<LsfgEndpointSlot> slotEndpoints;
    VkImageView viewP = VK_NULL_HANDLE;

    ExternalSlotState slots[2];

    bool initialized = false;
    bool hasCommittedFrame = false;
    uint64_t lastCommittedIndex = 0;
};

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
    ctx->device = desc->device;
    ctx->extent = extent;
    ctx->format = format;
    ctx->isHdr = desc->isHdr;
    ctx->flowScale = desc->flowScale;
    ctx->generationCount = desc->generationCount;
    ctx->viewP = endpoints->viewP;
    ctx->gipa = desc->getInstanceProcAddr;
    ctx->gdpa = desc->getDeviceProcAddr;

    if (ctx->gdpa != nullptr) {
        ctx->createShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(ctx->gdpa(ctx->device, "vkCreateShaderModule"));
        ctx->destroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(ctx->gdpa(ctx->device, "vkDestroyShaderModule"));
        ctx->createDescriptorSetLayout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(ctx->gdpa(ctx->device, "vkCreateDescriptorSetLayout"));
        ctx->destroyDescriptorSetLayout = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(ctx->gdpa(ctx->device, "vkDestroyDescriptorSetLayout"));
        ctx->createPipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(ctx->gdpa(ctx->device, "vkCreatePipelineLayout"));
        ctx->destroyPipelineLayout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(ctx->gdpa(ctx->device, "vkDestroyPipelineLayout"));
        ctx->createComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(ctx->gdpa(ctx->device, "vkCreateComputePipelines"));
        ctx->destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(ctx->gdpa(ctx->device, "vkDestroyPipeline"));
        ctx->createDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(ctx->gdpa(ctx->device, "vkCreateDescriptorPool"));
        ctx->destroyDescriptorPool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(ctx->gdpa(ctx->device, "vkDestroyDescriptorPool"));
        ctx->createSampler = reinterpret_cast<PFN_vkCreateSampler>(ctx->gdpa(ctx->device, "vkCreateSampler"));
        ctx->destroySampler = reinterpret_cast<PFN_vkDestroySampler>(ctx->gdpa(ctx->device, "vkDestroySampler"));
        ctx->cmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(ctx->gdpa(ctx->device, "vkCmdBindPipeline"));
        ctx->cmdBindDescriptorSets = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(ctx->gdpa(ctx->device, "vkCmdBindDescriptorSets"));
        ctx->cmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(ctx->gdpa(ctx->device, "vkCmdDispatch"));
        ctx->cmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(ctx->gdpa(ctx->device, "vkCmdPipelineBarrier"));
        ctx->destroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(ctx->gdpa(ctx->device, "vkDestroyBuffer"));
        ctx->freeMemory = reinterpret_cast<PFN_vkFreeMemory>(ctx->gdpa(ctx->device, "vkFreeMemory"));
    }

    if (endpoints->slots != nullptr && endpoints->slotCount > 0) {
        ctx->slotEndpoints.assign(endpoints->slots, endpoints->slots + endpoints->slotCount);
    }

    // Sampler setup
    if (ctx->createSampler != nullptr) {
        VkSamplerCreateInfo sInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sInfo.magFilter = VK_FILTER_LINEAR;
        sInfo.minFilter = VK_FILTER_LINEAR;
        sInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ctx->createSampler(ctx->device, &sInfo, nullptr, &ctx->linearSampler);
        ctx->createSampler(ctx->device, &sInfo, nullptr, &ctx->clampSampler);
    }

    if (!desc->shaderLoader) {
        delete ctx;
        return nullptr;
    }

    uint32_t resIds[25] = {
        255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
        267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279
    };

    bool allCreated = true;
    for (uint32_t id : resIds) {
        const ShaderAbiDef* abi = getShaderAbi(id);
        if (abi == nullptr) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "Shader res_%u has NO defined descriptor ABI! Failing context creation.", id);
#endif
            allCreated = false;
            break;
        }

        char name[32];
        std::snprintf(name, sizeof(name), "res_%u.spv", id);
        auto spv = desc->shaderLoader(name);
        if (spv.empty()) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "Shader res_%u payload empty from loader! Failing context creation.", id);
#endif
            allCreated = false;
            break;
        }

        VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smInfo.codeSize = spv.size();
        smInfo.pCode = reinterpret_cast<const uint32_t*>(spv.data());
        VkShaderModule mod = VK_NULL_HANDLE;
        VkResult resSm = (ctx->createShaderModule != nullptr) ? ctx->createShaderModule(ctx->device, &smInfo, nullptr, &mod) : VK_ERROR_INITIALIZATION_FAILED;
        if (resSm != VK_SUCCESS || mod == VK_NULL_HANDLE) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "vkCreateShaderModule failed for res_%u (res=%d)!", id, resSm);
#endif
            allCreated = false;
            break;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(abi->bindingCount);
        for (uint32_t b = 0; b < abi->bindingCount; ++b) {
            const auto& bDef = abi->bindings[b];
            VkDescriptorSetLayoutBinding bInfo{};
            bInfo.binding = bDef.binding;
            bInfo.descriptorType = bDef.descriptorType;
            bInfo.descriptorCount = bDef.descriptorCount;
            bInfo.stageFlags = bDef.stageFlags;
            bindings.push_back(bInfo);
        }

        VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        dslInfo.pBindings = bindings.data();
        VkResult resDsl = (ctx->createDescriptorSetLayout != nullptr) ? ctx->createDescriptorSetLayout(ctx->device, &dslInfo, nullptr, &descLayout) : VK_ERROR_INITIALIZATION_FAILED;
        if (resDsl != VK_SUCCESS || descLayout == VK_NULL_HANDLE) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "vkCreateDescriptorSetLayout failed for res_%u (res=%d)!", id, resDsl);
#endif
            if (ctx->destroyShaderModule != nullptr) ctx->destroyShaderModule(ctx->device, mod, nullptr);
            allCreated = false;
            break;
        }

        VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
        VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &descLayout;
        VkResult resPl = (ctx->createPipelineLayout != nullptr) ? ctx->createPipelineLayout(ctx->device, &plInfo, nullptr, &pipeLayout) : VK_ERROR_INITIALIZATION_FAILED;
        if (resPl != VK_SUCCESS || pipeLayout == VK_NULL_HANDLE) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "vkCreatePipelineLayout failed for res_%u (res=%d)!", id, resPl);
#endif
            if (ctx->destroyDescriptorSetLayout != nullptr) ctx->destroyDescriptorSetLayout(ctx->device, descLayout, nullptr);
            if (ctx->destroyShaderModule != nullptr) ctx->destroyShaderModule(ctx->device, mod, nullptr);
            allCreated = false;
            break;
        }

        VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = pipeLayout;

        VkPipeline pipe = VK_NULL_HANDLE;
        VkResult resPipe = (ctx->createComputePipelines != nullptr) ? ctx->createComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipe) : VK_ERROR_INITIALIZATION_FAILED;
        if (resPipe != VK_SUCCESS || pipe == VK_NULL_HANDLE) {
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSFG-ENGINE", "vkCreateComputePipelines failed for res_%u (res=%d)!", id, resPipe);
#endif
            if (ctx->destroyPipelineLayout != nullptr) ctx->destroyPipelineLayout(ctx->device, pipeLayout, nullptr);
            if (ctx->destroyDescriptorSetLayout != nullptr) ctx->destroyDescriptorSetLayout(ctx->device, descLayout, nullptr);
            if (ctx->destroyShaderModule != nullptr) ctx->destroyShaderModule(ctx->device, mod, nullptr);
            allCreated = false;
            break;
        }

        ExternalPipelineInfo info;
        info.resId = id;
        info.module = mod;
        info.descSetLayout = descLayout;
        info.pipelineLayout = pipeLayout;
        info.pipeline = pipe;
        ctx->pipelines[id] = info;
    }

    if (!allCreated) {
        lsfg_destroy_context_external(ctx);
        return nullptr;
    }

    ctx->initialized = true;
    return ctx;
}

void lsfg_destroy_context_external(LsfgExternalContextHandle ctx) {
    if (ctx == nullptr) return;

    for (auto &pair : ctx->pipelines) {
        if (pair.second.pipeline != VK_NULL_HANDLE && ctx->destroyPipeline != nullptr) {
            ctx->destroyPipeline(ctx->device, pair.second.pipeline, nullptr);
        }
        if (pair.second.pipelineLayout != VK_NULL_HANDLE && ctx->destroyPipelineLayout != nullptr) {
            ctx->destroyPipelineLayout(ctx->device, pair.second.pipelineLayout, nullptr);
        }
        if (pair.second.descSetLayout != VK_NULL_HANDLE && ctx->destroyDescriptorSetLayout != nullptr) {
            ctx->destroyDescriptorSetLayout(ctx->device, pair.second.descSetLayout, nullptr);
        }
        if (pair.second.module != VK_NULL_HANDLE && ctx->destroyShaderModule != nullptr) {
            ctx->destroyShaderModule(ctx->device, pair.second.module, nullptr);
        }
    }
    if (ctx->descriptorPool != VK_NULL_HANDLE && ctx->destroyDescriptorPool != nullptr) {
        ctx->destroyDescriptorPool(ctx->device, ctx->descriptorPool, nullptr);
    }
    if (ctx->linearSampler != VK_NULL_HANDLE && ctx->destroySampler != nullptr) {
        ctx->destroySampler(ctx->device, ctx->linearSampler, nullptr);
    }
    if (ctx->clampSampler != VK_NULL_HANDLE && ctx->destroySampler != nullptr) {
        ctx->destroySampler(ctx->device, ctx->clampSampler, nullptr);
    }
    for (int s = 0; s < 2; ++s) {
        if (ctx->slots[s].constantBuffer != VK_NULL_HANDLE && ctx->destroyBuffer != nullptr) {
            ctx->destroyBuffer(ctx->device, ctx->slots[s].constantBuffer, nullptr);
        }
        if (ctx->slots[s].constantBufferMemory != VK_NULL_HANDLE && ctx->freeMemory != nullptr) {
            ctx->freeMemory(ctx->device, ctx->slots[s].constantBufferMemory, nullptr);
        }
    }

    delete ctx;
}

VkResult lsfg_record_generation(
    LsfgExternalContextHandle ctx,
    VkCommandBuffer cmdBuffer,
    uint32_t slotIndex,
    uint64_t proposedFrameIndex,
    float interpolationFactor)
{
    if (ctx == nullptr || !ctx->initialized || ctx->cmdPipelineBarrier == nullptr || ctx->cmdBindPipeline == nullptr || ctx->cmdDispatch == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // 0. Cross-frame memory barrier for persistent scratch/history
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

    // Stage 1: Mipmaps pass (Res 255, 1 dispatch)
    auto itMip = ctx->pipelines.find(255);
    if (itMip != ctx->pipelines.end() && itMip->second.pipeline != VK_NULL_HANDLE) {
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itMip->second.pipeline);
        if (ctx->slots[slotIndex % 2].mipmapsSet != VK_NULL_HANDLE && ctx->cmdBindDescriptorSets != nullptr) {
            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itMip->second.pipelineLayout, 0, 1, &ctx->slots[slotIndex % 2].mipmapsSet, 0, nullptr);
        }
        uint32_t gx = (ctx->extent.width + 63) / 64;
        uint32_t gy = (ctx->extent.height + 63) / 64;
        ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    }
    emitComputeBarrier();

    // Stage 2: Alpha passes (7 levels x 4 passes = 28 dispatches)
    uint32_t alphaResIds[4] = {267, 268, 269, 270};
    for (int lvl = 0; lvl < 7; ++lvl) {
        for (int p = 0; p < 4; ++p) {
            uint32_t resId = alphaResIds[p];
            auto itA = ctx->pipelines.find(resId);
            if (itA != ctx->pipelines.end() && itA->second.pipeline != VK_NULL_HANDLE) {
                ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itA->second.pipeline);
                uint32_t lw = std::max(1u, (ctx->extent.width >> lvl));
                uint32_t lh = std::max(1u, (ctx->extent.height >> lvl));
                ctx->cmdDispatch(cmdBuffer, (lw + 7) / 8, (lh + 7) / 8, 1);
            }
            emitComputeBarrier();
        }
    }

    // Stage 3: Beta passes (5 passes = 5 dispatches)
    uint32_t betaResIds[5] = {275, 276, 277, 278, 279};
    for (int p = 0; p < 5; ++p) {
        uint32_t resId = betaResIds[p];
        auto itB = ctx->pipelines.find(resId);
        if (itB != ctx->pipelines.end() && itB->second.pipeline != VK_NULL_HANDLE) {
            ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itB->second.pipeline);
            uint32_t bw = std::max(1u, ctx->extent.width / 4);
            uint32_t bh = std::max(1u, ctx->extent.height / 4);
            uint32_t bs = (p == 4) ? 32 : 8;
            ctx->cmdDispatch(cmdBuffer, (bw + bs - 1) / bs, (bh + bs - 1) / bs, 1);
        }
        emitComputeBarrier();
    }

    // Stage 4: Gamma passes (7 levels x 5 passes = 35 dispatches)
    uint32_t gammaResIds[5] = {257, 259, 260, 261, 262};
    for (int lvl = 0; lvl < 7; ++lvl) {
        for (int p = 0; p < 5; ++p) {
            uint32_t resId = gammaResIds[p];
            auto itG = ctx->pipelines.find(resId);
            if (itG != ctx->pipelines.end() && itG->second.pipeline != VK_NULL_HANDLE) {
                ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itG->second.pipeline);
                uint32_t gw = std::max(1u, (ctx->extent.width >> (6 - lvl)));
                uint32_t gh = std::max(1u, (ctx->extent.height >> (6 - lvl)));
                ctx->cmdDispatch(cmdBuffer, (gw + 7) / 8, (gh + 7) / 8, 1);
            }
            emitComputeBarrier();
        }
    }

    // Stage 5: Delta passes (3 levels x 10 passes = 30 dispatches)
    uint32_t deltaResIds[10] = {257, 263, 264, 265, 266, 258, 271, 272, 273, 274};
    for (int lvl = 0; lvl < 3; ++lvl) {
        for (int p = 0; p < 10; ++p) {
            uint32_t resId = deltaResIds[p];
            auto itD = ctx->pipelines.find(resId);
            if (itD != ctx->pipelines.end() && itD->second.pipeline != VK_NULL_HANDLE) {
                ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itD->second.pipeline);
                uint32_t dw = std::max(1u, (ctx->extent.width >> (2 - lvl)));
                uint32_t dh = std::max(1u, (ctx->extent.height >> (2 - lvl)));
                ctx->cmdDispatch(cmdBuffer, (dw + 7) / 8, (dh + 7) / 8, 1);
            }
            emitComputeBarrier();
        }
    }

    // Stage 6: Generate pass (Res 256, 1 dispatch)
    auto itGen = ctx->pipelines.find(256);
    if (itGen != ctx->pipelines.end() && itGen->second.pipeline != VK_NULL_HANDLE) {
        ctx->cmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itGen->second.pipeline);
        if (ctx->slots[slotIndex % 2].generateSet != VK_NULL_HANDLE && ctx->cmdBindDescriptorSets != nullptr) {
            ctx->cmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, itGen->second.pipelineLayout, 0, 1, &ctx->slots[slotIndex % 2].generateSet, 0, nullptr);
        }
        uint32_t gx = (ctx->extent.width + 15) / 16;
        uint32_t gy = (ctx->extent.height + 15) / 16;
        ctx->cmdDispatch(cmdBuffer, gx > 0 ? gx : 1, gy > 0 ? gy : 1, 1);
    }
    // Total compute dispatches = 1 + 28 + 5 + 35 + 30 + 1 = 100 dispatches

    return VK_SUCCESS;
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

