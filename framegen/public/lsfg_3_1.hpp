#pragma once

#include <vulkan/vulkan_core.h>

#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __ANDROID__
struct AHardwareBuffer;
#endif

namespace LSFG_3_1 {

    ///
    /// Initialize the LSFG library.
    ///
    /// @param deviceUUID The UUID of the Vulkan device to use.
    /// @param isHdr Whether the images are in HDR format.
    /// @param flowScale Internal flow scale factor.
    /// @param generationCount Number of frames to generate.
    /// @param loader Function to load shader source code by name.
    ///
    /// @throws LSFG::vulkan_error if Vulkan objects fail to initialize.
    ///
    __attribute__((visibility("default")))
    void initialize(uint64_t deviceUUID,
        bool isHdr, float flowScale, uint64_t generationCount,
        const std::function<std::vector<uint8_t>(const std::string&)>& loader);

    ///
    /// Create a new LSFG context on a swapchain.
    ///
    /// @param in0 File descriptor for the first input image.
    /// @param in1 File descriptor for the second input image.
    /// @param outN File descriptor for each output image. This defines the LSFG level.
    /// @param extent The size of the images
    /// @param format The format of the images.
    /// @return A unique identifier for the created context.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be created.
    ///
    __attribute__((visibility("default")))
    int32_t createContext(
        int in0, int in1, const std::vector<int>& outN,
        VkExtent2D extent, VkFormat format);

#ifdef __ANDROID__
    ///
    /// Android-specific variant: share input/output images via AHardwareBuffer
    /// instead of opaque file descriptors. Required because Adreno/Mali drivers
    /// refuse vkGetMemoryFdKHR(OPAQUE_FD) on AHB-imported memory, breaking the
    /// FD-based path. The caller retains ownership of all AHBs and must keep
    /// them alive for the lifetime of the context.
    ///
    /// @param in0 First input image's AHardwareBuffer.
    /// @param in1 Second input image's AHardwareBuffer.
    /// @param outN Output image AHardwareBuffers, one per generated frame.
    /// @param extent Image dimensions.
    /// @param format Vulkan format of all images (must match the AHB format).
    /// @return Unique context identifier.
    ///
    __attribute__((visibility("default")))
    int32_t createContextFromAHB(
        AHardwareBuffer* in0, AHardwareBuffer* in1,
        const std::vector<AHardwareBuffer*>& outN,
        VkExtent2D extent, VkFormat format);
#endif

    ///
    /// Present a context.
    ///
    /// @param id Unique identifier of the context to present.
    /// @param inSem Semaphore to wait on before starting the generation.
    /// @param outSem Semaphores to signal once each output image is ready.
    ///
    /// @throws LSFG::vulkan_error if the context cannot be presented.
    ///
    __attribute__((visibility("default")))
    void presentContext(int32_t id, int inSem, const std::vector<int>& outSem);

    ///
    /// Delete an LSFG context.
    ///
    /// @param id Unique identifier of the context to delete.
    ///
    __attribute__((visibility("default")))
    void deleteContext(int32_t id);

    ///
    /// Deinitialize the LSFG library.
    ///
    __attribute__((visibility("default")))
    void finalize();

#ifdef __ANDROID__
    /// Block until framegen's internal Vulkan device is idle. Used by the
    /// Android wrapper to sync between its own device (which writes input
    /// AHBs) and framegen's device (which reads them) — without an explicit
    /// shared semaphore this is the only safe way to avoid a write-after-read
    /// race on the shared AHardwareBuffer storage.
    __attribute__((visibility("default")))
    void waitIdle();
#endif

    ///
    /// External device context descriptor for callers that manage their own
    /// Vulkan instance, physical device, device, and present queue (e.g. Amethyst).
    ///
    struct LsfgExternalContextDesc {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
        PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
        bool isHdr = false;
        float flowScale = 1.0f;
        uint32_t generationCount = 1;
        std::function<std::vector<uint8_t>(const std::string&)> shaderLoader;
    };

    struct LsfgEndpointSlot {
        VkImage imageC = VK_NULL_HANDLE;
        VkImageView viewC = VK_NULL_HANDLE;
        VkImage imageG = VK_NULL_HANDLE;
        VkImageView viewG = VK_NULL_HANDLE;
    };

    struct LsfgExternalEndpoints {
        VkImage imageP = VK_NULL_HANDLE;
        VkImageView viewP = VK_NULL_HANDLE;
        uint32_t slotCount = 0;
        const LsfgEndpointSlot* slots = nullptr;
    };

    typedef struct LsfgExternalContext* LsfgExternalContextHandle;

    __attribute__((visibility("default")))
    LsfgExternalContextHandle lsfg_create_context_external(
        const LsfgExternalContextDesc* desc,
        const LsfgExternalEndpoints* endpoints,
        VkExtent2D extent,
        VkFormat format);

    __attribute__((visibility("default")))
    void lsfg_destroy_context_external(LsfgExternalContextHandle ctx);

    // R2 public profiling struct — exact binary ABI, never extended.
    // lsfg_record_generation_profiled emits Q0..Q6 only.
    // DO NOT add fields here; adding fields would break binary ABI with
    // pre-compiled R2 callers that pass a pointer to the 4-field layout.
    struct LsfgStageProfilingInfo {
        bool enabled = false;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint32_t queryBase = 0;
        PFN_vkCmdWriteTimestamp cmdWriteTimestamp = nullptr;
        // sizeof on arm64: 4 (bool+pad3) + 8 (VkQueryPool ptr) + 4 (queryBase) + 4 (pad) + 8 (fn ptr) = 24 bytes
        // offsetof: enabled=0, queryPool=8, queryBase=16, cmdWriteTimestamp=24 (total=32 bytes)
    };

    // R3 dispatch-level profiling struct — separate type for the R3 API.
    // Identical field layout to LsfgStageProfilingInfo by design so that
    // existing pool/cmd infrastructure is reused without ABI coupling.
    // lsfg_record_generation_profiled_r3 emits Q0..Q100 only.
    struct LsfgDispatchProfilingInfo {
        bool enabled = false;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint32_t queryBase = 0;
        PFN_vkCmdWriteTimestamp cmdWriteTimestamp = nullptr;
    };

    __attribute__((visibility("default")))
    VkResult lsfg_record_generation(
        LsfgExternalContextHandle ctx,
        VkCommandBuffer cmdBuffer,
        uint32_t slotIndex,
        uint64_t proposedFrameIndex,
        float interpolationFactor);

    // R2 API: emits Q0..Q6 stage boundaries only.
    // Maximum compute query index emitted: 6.
    // Binary ABI frozen at 4-field LsfgStageProfilingInfo.
    __attribute__((visibility("default")))
    VkResult lsfg_record_generation_profiled(
        LsfgExternalContextHandle ctx,
        VkCommandBuffer cmdBuffer,
        uint32_t slotIndex,
        uint64_t proposedFrameIndex,
        float interpolationFactor,
        const LsfgStageProfilingInfo* profiling);

    // R3 API: emits Q0..Q100 full per-dispatch boundaries only.
    // Maximum compute query index emitted: 100.
    // Amethyst then emits Q101 (G->D) and Q102 (D->M) independently.
    // Query pool must have queryCount >= 103 (Amethyst uses exactly 103).
    __attribute__((visibility("default")))
    VkResult lsfg_record_generation_profiled_r3(
        LsfgExternalContextHandle ctx,
        VkCommandBuffer cmdBuffer,
        uint32_t slotIndex,
        uint64_t proposedFrameIndex,
        float interpolationFactor,
        const LsfgDispatchProfilingInfo* profiling);

    __attribute__((visibility("default")))
    void lsfg_commit_generation(
        LsfgExternalContextHandle ctx,
        uint64_t committedFrameIndex);

}
