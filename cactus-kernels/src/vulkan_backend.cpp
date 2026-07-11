#include "vulkan_backend.h"

#if !defined(__ANDROID__)

bool cactus_vulkan_available() { return false; }
const char* cactus_vulkan_device_info() { return "Vulkan: unavailable (non-Android build)"; }
bool cactus_vulkan_op_enabled(const char*) { return false; }
void cactus_vulkan_session_begin() {}
void cactus_vulkan_session_flush() {}
void cactus_vulkan_session_sync() {}
void cactus_vulkan_session_end() {}
void cactus_vulkan_invalidate_host_wraps() {}
void cactus_vulkan_trim_prefill_cache() {}
void* cactus_vulkan_alloc_shared(size_t) { return nullptr; }
void* cactus_vulkan_alloc_pooled(size_t) { return nullptr; }
void cactus_vulkan_free_shared(void*) {}
bool cactus_vulkan_encode_binary_f16(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_scalar_f16(int, void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_unary_f16(int, void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_swiglu_f16(void*, const void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_rms_norm_f16(void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_vulkan_encode_cq_gemv(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_binary_f16(int, __fp16*, const __fp16*, const __fp16*, size_t) { return false; }
bool cactus_vulkan_scalar_f16(int, __fp16*, const __fp16*, size_t, float) { return false; }
bool cactus_vulkan_unary_f16(int, __fp16*, const __fp16*, size_t) { return false; }
bool cactus_vulkan_swiglu_f16(__fp16*, const __fp16*, const __fp16*, size_t, float) { return false; }
bool cactus_vulkan_rms_norm_f16(__fp16*, const __fp16*, const __fp16*, size_t, size_t, float) { return false; }
bool cactus_vulkan_cq_gemv(__fp16*, const __fp16*, const CactusQuantMatrix*, int, double*) { return false; }
bool cactus_vulkan_argmax_f16(const __fp16*, uint32_t, uint32_t*, float*) { return false; }

#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "vk_binary.h"
#include "vk_scalar.h"
#include "vk_unary.h"
#include "vk_swiglu.h"
#include "vk_rms_norm.h"
#include "vk_cq4_transform.h"
#include "vk_cq4_gemv.h"
#include "vk_argmax.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

#define VK_FNS_GLOBAL(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties)
#define VK_FNS_INSTANCE(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFeatures2) X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define VK_FNS_DEVICE(X) \
    X(vkGetDeviceQueue) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines) \
    X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkQueueSubmit) \
    X(vkCreateFence) X(vkResetFences) X(vkWaitForFences) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) \
    X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)

enum { KB, KSC, KU, KSW, KR, KT, KG, KA, KCOUNT };

struct KDef {
    const char* spv;
    size_t len;
    uint32_t nbuf;
    uint32_t push;
    uint32_t write_mask;
};

const KDef kdefs[KCOUNT] = {
    {kSpv_binary,        sizeof(kSpv_binary) - 1,        3, 8,  0x4},
    {kSpv_scalar,        sizeof(kSpv_scalar) - 1,        2, 12, 0x2},
    {kSpv_unary,         sizeof(kSpv_unary) - 1,         2, 8,  0x2},
    {kSpv_swiglu,        sizeof(kSpv_swiglu) - 1,        3, 8,  0x4},
    {kSpv_rms_norm,      sizeof(kSpv_rms_norm) - 1,      3, 8,  0x4},
    {kSpv_cq4_transform, sizeof(kSpv_cq4_transform) - 1, 6, 4,  0x20},
    {kSpv_cq4_gemv,      sizeof(kSpv_cq4_gemv) - 1,      5, 20, 0x10},
    {kSpv_argmax,        sizeof(kSpv_argmax) - 1,        2, 4,  0x2},
};

struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool desc = VK_NULL_HANDLE;
    bool recording = false;
    bool in_flight = false;
    bool dirty = true;
};

struct VK {
    void* lib = nullptr;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
#define D(f) PFN_##f f = nullptr;
    VK_FNS_GLOBAL(D)
    VK_FNS_INSTANCE(D)
    VK_FNS_DEVICE(D)
#undef D

    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkPhysicalDeviceMemoryProperties memprops = {};
    VkDeviceSize sb_align = 256;
    Slot slots[3];
    int cur = 0;
    VkQueryPool query = VK_NULL_HANDLE;
    float ts_period = 0.0f;
    struct Pipe { VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkPipeline p = VK_NULL_HANDLE; } pipes[KCOUNT];
    std::string info = "Vulkan: not initialized";
    bool ok = false;

    VK() { init(); }

    void init() {
        lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) { info = "Vulkan: libvulkan.so not found"; return; }
        gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
        if (!gipa) { info = "Vulkan: vkGetInstanceProcAddr missing"; return; }
#define R(f) f = reinterpret_cast<PFN_##f>(gipa(nullptr, #f)); if (!f) { info = "Vulkan: missing " #f; return; }
        VK_FNS_GLOBAL(R)
#undef R

        const char* layers[1] = {"VK_LAYER_KHRONOS_validation"};
        uint32_t nlayers = 0;
        if (getenv("CACTUS_VK_VALIDATE")) {
            uint32_t avail = 0;
            vkEnumerateInstanceLayerProperties(&avail, nullptr);
            std::vector<VkLayerProperties> props(avail);
            vkEnumerateInstanceLayerProperties(&avail, props.data());
            for (const auto& lp : props)
                if (std::strcmp(lp.layerName, layers[0]) == 0) nlayers = 1;
        }

        VkApplicationInfo app = {};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "cactus";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        ici.enabledLayerCount = nlayers;
        ici.ppEnabledLayerNames = layers;
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS || !inst) {
            info = "Vulkan: instance creation failed";
            return;
        }
#define R(f) f = reinterpret_cast<PFN_##f>(gipa(inst, #f)); if (!f) { info = "Vulkan: missing " #f; return; }
        VK_FNS_INSTANCE(R)
#undef R

        uint32_t ndev = 0;
        vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
        if (ndev == 0) { info = "Vulkan: no physical devices"; return; }
        std::vector<VkPhysicalDevice> devs(ndev);
        vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

        VkPhysicalDeviceSubgroupProperties subp = {};
        subp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subp;
        VkPhysicalDevice16BitStorageFeatures s16 = {};
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties2 p2 = props2;
            vkGetPhysicalDeviceProperties2(d, &p2);
            if (p2.properties.apiVersion < VK_API_VERSION_1_1) continue;
            uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qs(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
            uint32_t fam = UINT32_MAX;
            for (uint32_t i = 0; i < nq; ++i)
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = i; break; }
            if (fam == UINT32_MAX) continue;
            s16 = {};
            s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            VkPhysicalDeviceFeatures2 f2 = {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &s16;
            vkGetPhysicalDeviceFeatures2(d, &f2);
            if (!s16.storageBuffer16BitAccess) continue;
            phys = d;
            qfam = fam;
            props2 = p2;
            break;
        }
        if (!phys) { info = "Vulkan: no compute device with 16-bit storage"; return; }

        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
        sb_align = props2.properties.limits.minStorageBufferOffsetAlignment;
        if (sb_align < 16) sb_align = 16;
        if (props2.properties.limits.maxPushConstantsSize < 128) {
            info = "Vulkan: push constant limit below 128";
            return;
        }
        if (props2.properties.limits.timestampComputeAndGraphics)
            ts_period = props2.properties.limits.timestampPeriod;

        bool f16ext = props2.properties.apiVersion >= VK_API_VERSION_1_2;
        if (!f16ext) {
            uint32_t next = 0;
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
            std::vector<VkExtensionProperties> exts(next);
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, exts.data());
            for (const auto& e : exts)
                if (std::strcmp(e.extensionName, "VK_KHR_shader_float16_int8") == 0) f16ext = true;
        }
        bool shader_f16 = false;
        if (f16ext) {
            VkPhysicalDeviceShaderFloat16Int8Features hf = {};
            hf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            VkPhysicalDeviceFeatures2 f2 = {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &hf;
            vkGetPhysicalDeviceFeatures2(phys, &f2);
            shader_f16 = hf.shaderFloat16;
        }

        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s | Vulkan %u.%u | driver %u | 16bitStorage:yes shaderFloat16:%s subgroup:%u sharedMem:%u",
                      props2.properties.deviceName,
                      VK_API_VERSION_MAJOR(props2.properties.apiVersion),
                      VK_API_VERSION_MINOR(props2.properties.apiVersion),
                      props2.properties.driverVersion,
                      shader_f16 ? "yes" : "no",
                      subp.subgroupSize,
                      props2.properties.limits.maxComputeSharedMemorySize);
        info = buf;

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDevice16BitStorageFeatures en16 = {};
        en16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        en16.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 enf2 = {};
        enf2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enf2.pNext = &en16;
        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &enf2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) {
            info += " | device creation failed";
            return;
        }
#define R(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(dev, #f)); if (!f) { info += " | missing " #f; return; }
        VK_FNS_DEVICE(R)
#undef R
        vkGetDeviceQueue(dev, qfam, 0, &q);

        for (Slot& s : slots) {
            VkCommandPoolCreateInfo cpci = {};
            cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpci.queueFamilyIndex = qfam;
            if (vkCreateCommandPool(dev, &cpci, nullptr, &s.pool) != VK_SUCCESS) { info += " | command pool failed"; return; }
            VkCommandBufferAllocateInfo cbai = {};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = s.pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(dev, &cbai, &s.cmd) != VK_SUCCESS) { info += " | command buffer failed"; return; }
            VkFenceCreateInfo fci = {};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(dev, &fci, nullptr, &s.fence) != VK_SUCCESS) { info += " | fence failed"; return; }
            VkDescriptorPoolSize psz = {};
            psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            psz.descriptorCount = 8192;
            VkDescriptorPoolCreateInfo dpci = {};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = 2048;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &psz;
            if (vkCreateDescriptorPool(dev, &dpci, nullptr, &s.desc) != VK_SUCCESS) { info += " | descriptor pool failed"; return; }
        }

        if (ts_period > 0.0f) {
            VkQueryPoolCreateInfo qpci = {};
            qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = 2;
            if (vkCreateQueryPool(dev, &qpci, nullptr, &query) != VK_SUCCESS) { query = VK_NULL_HANDLE; ts_period = 0.0f; }
        }

        for (int k = 0; k < KCOUNT; ++k) {
            const KDef& kd = kdefs[k];
            if (kd.len % 4 != 0) { info += " | spv size misaligned"; return; }
            std::vector<VkDescriptorSetLayoutBinding> binds(kd.nbuf);
            for (uint32_t i = 0; i < kd.nbuf; ++i) {
                binds[i] = {};
                binds[i].binding = i;
                binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                binds[i].descriptorCount = 1;
                binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dslci = {};
            dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = kd.nbuf;
            dslci.pBindings = binds.data();
            if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &pipes[k].dsl) != VK_SUCCESS) { info += " | dsl failed"; return; }
            VkPushConstantRange range = {};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = kd.push;
            VkPipelineLayoutCreateInfo plci = {};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &pipes[k].dsl;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &range;
            if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipes[k].pl) != VK_SUCCESS) { info += " | pipeline layout failed"; return; }
            std::vector<uint32_t> code(kd.len / 4);
            std::memcpy(code.data(), kd.spv, kd.len);
            VkShaderModuleCreateInfo smci = {};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = kd.len;
            smci.pCode = code.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            if (vkCreateShaderModule(dev, &smci, nullptr, &mod) != VK_SUCCESS) { info += " | shader module failed"; return; }
            VkComputePipelineCreateInfo cpi = {};
            cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = mod;
            cpi.stage.pName = "main";
            cpi.layout = pipes[k].pl;
            VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipes[k].p);
            vkDestroyShaderModule(dev, mod, nullptr);
            if (pr != VK_SUCCESS) { info += " | pipeline creation failed"; return; }
        }
        ok = true;
        info += " | build:ok";
    }
};

VK& vk() { static VK v; return v; }
std::recursive_mutex g_mu;

constexpr size_t BUCKET = 16384;
constexpr size_t SLAB_BYTES = 32u << 20;
constexpr size_t POOL_MAX = 4u << 20;

struct MemBlock {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    char* map = nullptr;
    size_t cap = 0;
    bool slab = false;
    bool coherent = true;
    size_t used = 0;
    uint32_t live = 0;
};

bool g_noncoherent = false;
std::map<uintptr_t, MemBlock*> g_blocks;
std::unordered_map<size_t, std::vector<MemBlock*>> g_free_blocks;
std::vector<MemBlock*> g_pending;
std::vector<MemBlock*> g_slabs;
MemBlock* g_cur_slab = nullptr;

uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < vk().memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (vk().memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool create_block(size_t bytes, MemBlock& b, bool cached) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes ? bytes : 4;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk().vkCreateBuffer(vk().dev, &bci, nullptr, &b.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req = {};
    vk().vkGetBufferMemoryRequirements(vk().dev, b.buf, &req);
    uint32_t type = UINT32_MAX;
    if (cached) {
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (type == UINT32_MAX)
            type = mem_type(req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        static const bool force_cached = [] {
            const char* v = getenv("CACTUS_VK_CACHED");
            return v && std::strcmp(v, "force") == 0;
        }();
        if (type == UINT32_MAX && force_cached) {
            type = mem_type(req.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (type == UINT32_MAX)
                type = mem_type(req.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        }
    }
    if (type == UINT32_MAX)
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX)
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) { vk().vkDestroyBuffer(vk().dev, b.buf, nullptr); b.buf = VK_NULL_HANDLE; return false; }
    VkMemoryPropertyFlags mprops = vk().memprops.memoryTypes[type].propertyFlags;
    b.coherent = (mprops & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (!b.coherent) g_noncoherent = true;
    if (cached) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            char m[48];
            std::snprintf(m, sizeof(m), " | actMem:%s%s%s",
                          (mprops & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "dl+" : "",
                          (mprops & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "cached" : "wc",
                          b.coherent ? "+coh" : "");
            vk().info += m;
        }
    }
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    void* map = nullptr;
    if (vk().vkAllocateMemory(vk().dev, &mai, nullptr, &b.mem) != VK_SUCCESS
        || vk().vkBindBufferMemory(vk().dev, b.buf, b.mem, 0) != VK_SUCCESS
        || vk().vkMapMemory(vk().dev, b.mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
        if (b.mem) vk().vkFreeMemory(vk().dev, b.mem, nullptr);
        vk().vkDestroyBuffer(vk().dev, b.buf, nullptr);
        b = MemBlock();
        return false;
    }
    b.map = static_cast<char*>(map);
    b.cap = bytes;
    return true;
}

void* alloc_shared_i(size_t bytes) {
    size_t bk = (bytes ? bytes : 1);
    bk = (bk + BUCKET - 1) & ~(BUCKET - 1);
    auto& fl = g_free_blocks[bk];
    MemBlock* b;
    if (!fl.empty()) {
        b = fl.back();
        fl.pop_back();
    } else {
        b = new MemBlock();
        if (!create_block(bk, *b, true)) { delete b; return nullptr; }
        g_blocks[(uintptr_t)b->map] = b;
    }
    b->live = 1;
    return b->map;
}

void* alloc_pooled_i(size_t bytes) {
    if (bytes > POOL_MAX) return alloc_shared_i(bytes);
    size_t need = (bytes + 255) & ~(size_t)255;
    if (!g_cur_slab || g_cur_slab->used + need > g_cur_slab->cap) {
        for (MemBlock* s : g_slabs)
            if (s != g_cur_slab && s->live == 0 && s->cap >= need) { s->used = 0; g_cur_slab = s; break; }
        if (!g_cur_slab || g_cur_slab->used + need > g_cur_slab->cap) {
            MemBlock* s = new MemBlock();
            if (!create_block(SLAB_BYTES, *s, true)) { delete s; return alloc_shared_i(bytes); }
            s->slab = true;
            g_blocks[(uintptr_t)s->map] = s;
            g_slabs.push_back(s);
            g_cur_slab = s;
        }
    }
    void* p = g_cur_slab->map + g_cur_slab->used;
    g_cur_slab->used += need;
    g_cur_slab->live++;
    return p;
}

MemBlock* block_of(const void* p, size_t* off) {
    uintptr_t a = (uintptr_t)p;
    auto it = g_blocks.upper_bound(a);
    if (it == g_blocks.begin()) return nullptr;
    --it;
    MemBlock* b = it->second;
    if (a >= it->first + b->cap) return nullptr;
    *off = a - it->first;
    return b;
}

void free_shared_i(void* p) {
    if (!p) return;
    size_t off = 0;
    MemBlock* b = block_of(p, &off);
    if (!b) return;
    if (b->slab) {
        if (b->live) b->live--;
        return;
    }
    b->live = 0;
    g_pending.push_back(b);
}

void flush_noncoherent(bool invalidate) {
    if (!g_noncoherent) return;
    std::vector<VkMappedMemoryRange> rs;
    for (auto& kv : g_blocks) {
        if (kv.second->coherent) continue;
        VkMappedMemoryRange r = {};
        r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r.memory = kv.second->mem;
        r.size = VK_WHOLE_SIZE;
        rs.push_back(r);
    }
    if (rs.empty()) return;
    if (invalidate) vk().vkInvalidateMappedMemoryRanges(vk().dev, (uint32_t)rs.size(), rs.data());
    else vk().vkFlushMappedMemoryRanges(vk().dev, (uint32_t)rs.size(), rs.data());
}

bool submit_slot(Slot& s) {
    flush_noncoherent(false);
    if (vk().vkEndCommandBuffer(s.cmd) != VK_SUCCESS) { vk().ok = false; return false; }
    if (vk().vkResetFences(vk().dev, 1, &s.fence) != VK_SUCCESS) { vk().ok = false; return false; }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    if (vk().vkQueueSubmit(vk().q, 1, &si, s.fence) != VK_SUCCESS) { vk().ok = false; return false; }
    s.recording = false;
    s.in_flight = true;
    s.dirty = true;
    return true;
}

bool wait_slot(Slot& s) {
    if (!s.in_flight) return true;
    VkResult r = vk().vkWaitForFences(vk().dev, 1, &s.fence, VK_TRUE, 5000000000ull);
    s.in_flight = false;
    if (r != VK_SUCCESS) { vk().ok = false; return false; }
    return true;
}

bool ensure_cmd() {
    Slot& s = vk().slots[vk().cur];
    if (s.recording) return true;
    if (!wait_slot(s)) return false;
    if (s.dirty) {
        if (vk().vkResetCommandPool(vk().dev, s.pool, 0) != VK_SUCCESS) return false;
        if (vk().vkResetDescriptorPool(vk().dev, s.desc, 0) != VK_SUCCESS) return false;
        s.dirty = false;
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk().vkBeginCommandBuffer(s.cmd, &bi) != VK_SUCCESS) return false;
    s.recording = true;
    return true;
}

void flush_i() {
    Slot& s = vk().slots[vk().cur];
    if (!s.recording) return;
    if (!submit_slot(s)) return;
    vk().cur = (vk().cur + 1) % 3;
}

void recycle_pending() {
    for (MemBlock* b : g_pending) g_free_blocks[b->cap].push_back(b);
    g_pending.clear();
    for (MemBlock* s : g_slabs)
        if (s->live == 0) s->used = 0;
}

void sync_i() {
    flush_i();
    for (Slot& s : vk().slots) wait_slot(s);
    flush_noncoherent(true);
    recycle_pending();
}

VkCommandBuffer cur_cmd() { return vk().slots[vk().cur].cmd; }

VkDescriptorSet make_set(int k, const VkDescriptorBufferInfo* infos, uint32_t count) {
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = vk().slots[vk().cur].desc;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &vk().pipes[k].dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vk().vkAllocateDescriptorSets(vk().dev, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkWriteDescriptorSet writes[16] = {};
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vk().vkUpdateDescriptorSets(vk().dev, count, writes, 0, nullptr);
    return set;
}

void mem_barrier() {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vk().vkCmdPipelineBarrier(cur_cmd(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

struct AccessRange {
    VkBuffer buf;
    VkDeviceSize beg;
    VkDeviceSize end;
};
std::vector<AccessRange> g_writes, g_reads;

bool ranges_hit(const std::vector<AccessRange>& v, VkBuffer buf, VkDeviceSize beg, VkDeviceSize end) {
    for (const AccessRange& r : v)
        if (r.buf == buf && beg < r.end && r.beg < end) return true;
    return false;
}

void dispatch(int k, VkDescriptorSet set, const void* push, uint32_t gx, uint32_t gy,
              const VkDescriptorBufferInfo* infos, uint32_t count) {
    static const bool barrier_all = [] {
        const char* v = getenv("CACTUS_VK_BARRIER");
        return v && std::strcmp(v, "all") == 0;
    }();
    const uint32_t wm = kdefs[k].write_mask;
    bool hazard = barrier_all || g_writes.size() + g_reads.size() > 1024;
    for (uint32_t i = 0; i < count && !hazard; ++i) {
        VkDeviceSize beg = infos[i].offset, end = beg + infos[i].range;
        hazard = ranges_hit(g_writes, infos[i].buffer, beg, end)
              || (((wm >> i) & 1u) && ranges_hit(g_reads, infos[i].buffer, beg, end));
    }
    if (hazard) {
        mem_barrier();
        g_writes.clear();
        g_reads.clear();
    }
    for (uint32_t i = 0; i < count; ++i) {
        VkDeviceSize beg = infos[i].offset, end = beg + infos[i].range;
        (((wm >> i) & 1u) ? g_writes : g_reads).push_back({infos[i].buffer, beg, end});
    }
    vk().vkCmdBindPipeline(cur_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, vk().pipes[k].p);
    vk().vkCmdBindDescriptorSets(cur_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, vk().pipes[k].pl, 0, 1, &set, 0, nullptr);
    vk().vkCmdPushConstants(cur_cmd(), vk().pipes[k].pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, kdefs[k].push, push);
    vk().vkCmdDispatch(cur_cmd(), gx, gy, 1);
}

bool dinfo(const void* p, size_t bytes, VkDescriptorBufferInfo* out) {
    size_t off = 0;
    MemBlock* b = block_of(p, &off);
    if (!b) return false;
    if (off % vk().sb_align != 0) return false;
    out->buffer = b->buf;
    out->offset = off;
    out->range = bytes ? bytes : 4;
    return true;
}

uint32_t ew_groups(size_t n) {
    size_t g = (n + 63) / 64;
    return (uint32_t)(g > 65535 ? 65535 : g);
}

bool enc_ew(int k, void* y, size_t out_bytes,
            const void* i0, size_t b0, const void* i1, size_t b1,
            const void* push, uint32_t groups) {
    if (!vk().ok) return false;
    VkDescriptorBufferInfo infos[3];
    uint32_t idx = 0;
    if (!dinfo(i0, b0, &infos[idx++])) return false;
    if (i1 && !dinfo(i1, b1, &infos[idx++])) return false;
    if (!dinfo(y, out_bytes, &infos[idx++])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(k, infos, idx);
    if (!set) return false;
    dispatch(k, set, push, groups, 1, infos, idx);
    return true;
}

struct ResW {
    MemBlock b;
    VkDeviceSize off[8] = {};
    VkDeviceSize sz[8] = {};
    bool ok = false;
};
std::unordered_map<uint64_t, ResW> g_resident;

uint64_t resident_key(const CactusQuantMatrix* W) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(W->bits); mix(W->K); mix(W->N); mix(W->group_size); mix(W->num_groups); mix(W->flags);
    const uint8_t* p = W->packed_indices;
    size_t pkb = (size_t)W->N * W->num_groups * cactus_quant_packed_group_bytes(W->bits, W->group_size);
    if (!p || pkb == 0) return h;
    size_t take = pkb < 64 ? pkb : 64;
    for (size_t i = 0; i < take; ++i) mix(p[i]);
    for (size_t i = pkb - take; i < pkb; ++i) mix(p[i]);
    return h;
}

enum { W_PACKED, W_NORMS, W_CB, W_RECIP, W_LS, W_RS, W_PERM, W_CODE };

ResW& resident(const CactusQuantMatrix* W) {
    uint64_t key = resident_key(W);
    auto it = g_resident.find(key);
    if (it != g_resident.end()) return it->second;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const bool il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    const size_t rows = il ? ((size_t)(N + 3) & ~(size_t)3) : N;
    ResW r;
    const void* srcs[8] = {W->packed_indices, W->norms, W->codebook, W->input_scale_recip,
                           W->left_signs, W->right_signs, W->permutation, nullptr};
    r.sz[W_PACKED] = rows * ng * pgb;
    r.sz[W_NORMS] = rows * ng * sizeof(__fp16);
    r.sz[W_CB] = 16 * sizeof(__fp16);
    r.sz[W_RECIP] = (size_t)K * sizeof(__fp16);
    r.sz[W_LS] = gs;
    r.sz[W_RS] = gs;
    r.sz[W_PERM] = (size_t)gs * sizeof(uint32_t);
    r.sz[W_CODE] = (size_t)K * sizeof(__fp16);
    VkDeviceSize total = 0;
    for (int i = 0; i < 8; ++i) {
        total = (total + vk().sb_align - 1) & ~(vk().sb_align - 1);
        r.off[i] = total;
        total += r.sz[i];
    }
    if (create_block(total, r.b, false)) {
        for (int i = 0; i < 8; ++i)
            if (srcs[i]) std::memcpy(r.b.map + r.off[i], srcs[i], r.sz[i]);
        r.ok = true;
    }
    return g_resident.emplace(key, r).first->second;
}

bool enc_cq_gemv(void* y, const void* x, const CactusQuantMatrix* W) {
    if (!vk().ok || !W || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    if (!W->input_scale_recip || !W->left_signs || !W->right_signs || !W->permutation
        || !W->codebook || !W->norms || !W->packed_indices) return false;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    if (gs < 16 || gs > 512 || (gs & (gs - 1)) != 0 || (size_t)ng * gs != K) return false;
    if ((W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) && (N & 3)) return false;
    ResW& r = resident(W);
    if (!r.ok) return false;
    VkDescriptorBufferInfo xb, yb;
    if (!dinfo(x, (size_t)K * sizeof(__fp16), &xb)) return false;
    if (!dinfo(y, (size_t)N * sizeof(__fp16), &yb)) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf;
        d.offset = r.off[i];
        d.range = r.sz[i];
        return d;
    };
    if (!ensure_cmd()) return false;
    VkDescriptorBufferInfo tinfos[6] = {xb, ri(W_RECIP), ri(W_LS), ri(W_RS), ri(W_PERM), ri(W_CODE)};
    VkDescriptorBufferInfo ginfos[5] = {ri(W_CODE), ri(W_PACKED), ri(W_CB), ri(W_NORMS), yb};
    VkDescriptorSet tset = make_set(KT, tinfos, 6);
    VkDescriptorSet gset = make_set(KG, ginfos, 5);
    if (!tset || !gset) return false;
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const uint32_t il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) ? 1 : 0;
    struct { uint32_t gs; } tpush = {gs};
    struct { uint32_t gs, ng, pgb, N, il; } gpush = {gs, ng, pgb, N, il};
    const uint32_t rows4 = (N + 3) / 4;
    dispatch(KT, tset, &tpush, ng, 1, tinfos, 6);
    dispatch(KG, gset, &gpush, rows4 > 65535 ? 65535 : rows4, 1, ginfos, 5);
    return true;
}

struct HostIn {
    void* p = nullptr;
    HostIn(const void* src, size_t bytes) {
        p = alloc_shared_i(bytes);
        if (p && src) std::memcpy(p, src, bytes);
    }
    ~HostIn() { if (p) free_shared_i(p); }
};

}

bool cactus_vulkan_available() { return vk().ok; }

const char* cactus_vulkan_device_info() {
    if (vk().ok) {
        std::lock_guard<std::recursive_mutex> lk(g_mu);
        void* p = alloc_shared_i(1);
        if (p) free_shared_i(p);
    }
    return vk().info.c_str();
}

bool cactus_vulkan_op_enabled(const char* name) {
    static const char* filter = getenv("CACTUS_VK_OPS");
    if (!filter || !*filter) return true;
    return std::strstr(filter, name) != nullptr;
}

void cactus_vulkan_session_begin() {}

void cactus_vulkan_session_flush() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    flush_i();
}

void cactus_vulkan_session_sync() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    sync_i();
}

void cactus_vulkan_session_end() {
    cactus_vulkan_session_sync();
}

void cactus_vulkan_invalidate_host_wraps() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    sync_i();
    for (auto& kv : g_resident) {
        if (kv.second.b.buf) vk().vkDestroyBuffer(vk().dev, kv.second.b.buf, nullptr);
        if (kv.second.b.mem) vk().vkFreeMemory(vk().dev, kv.second.b.mem, nullptr);
    }
    g_resident.clear();
}

void cactus_vulkan_trim_prefill_cache() {}

void* cactus_vulkan_alloc_shared(size_t bytes) {
    if (!vk().ok) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return alloc_shared_i(bytes);
}

void* cactus_vulkan_alloc_pooled(size_t bytes) {
    if (!vk().ok) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return alloc_pooled_i(bytes);
}

void cactus_vulkan_free_shared(void* p) {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    free_shared_i(p);
}

bool cactus_vulkan_encode_binary_f16(int op, void* y, const void* a, const void* b, size_t n) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_ew(KB, y, n * 2, a, n * 2, b, n * 2, &push, ew_groups(n));
}

bool cactus_vulkan_encode_scalar_f16(int op, void* y, const void* in, size_t n, float p) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; float p; } push = {(uint32_t)n, op, p};
    return enc_ew(KSC, y, n * 2, in, n * 2, nullptr, 0, &push, ew_groups(n));
}

bool cactus_vulkan_encode_unary_f16(int op, void* y, const void* in, size_t n) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_ew(KU, y, n * 2, in, n * 2, nullptr, 0, &push, ew_groups(n));
}

bool cactus_vulkan_encode_swiglu_f16(void* y, const void* gate, const void* up, size_t n, float scale) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; float scale; } push = {(uint32_t)n, scale};
    return enc_ew(KSW, y, n * 2, gate, n * 2, up, n * 2, &push, ew_groups(n));
}

bool cactus_vulkan_encode_rms_norm_f16(void* y, const void* in, const void* w,
                                       size_t rows, size_t dim, float eps) {
    if (rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t dim; float eps; } push = {(uint32_t)dim, eps};
    return enc_ew(KR, y, rows * dim * 2, in, rows * dim * 2, w, dim * 2, &push, (uint32_t)rows);
}

bool cactus_vulkan_encode_cq_gemv(void* y, const void* x, const CactusQuantMatrix* W) {
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return enc_cq_gemv(y, x, W);
}

bool cactus_vulkan_binary_f16(int op, __fp16* out, const __fp16* a, const __fp16* b, size_t n) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ia(a, n * 2), ib(b, n * 2), oy(nullptr, n * 2);
    if (!ia.p || !ib.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    bool r = enc_ew(KB, oy.p, n * 2, ia.p, n * 2, ib.p, n * 2, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_scalar_f16(int op, __fp16* out, const __fp16* in, size_t n, float p) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, n * 2), oy(nullptr, n * 2);
    if (!ii.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; float p; } push = {(uint32_t)n, op, p};
    bool r = enc_ew(KSC, oy.p, n * 2, ii.p, n * 2, nullptr, 0, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_unary_f16(int op, __fp16* out, const __fp16* in, size_t n) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, n * 2), oy(nullptr, n * 2);
    if (!ii.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    bool r = enc_ew(KU, oy.p, n * 2, ii.p, n * 2, nullptr, 0, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_swiglu_f16(__fp16* out, const __fp16* gate, const __fp16* up, size_t n, float scale) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ig(gate, n * 2), iu(up, n * 2), oy(nullptr, n * 2);
    if (!ig.p || !iu.p || !oy.p) return false;
    struct { uint32_t n; float scale; } push = {(uint32_t)n, scale};
    bool r = enc_ew(KSW, oy.p, n * 2, ig.p, n * 2, iu.p, n * 2, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_rms_norm_f16(__fp16* out, const __fp16* in, const __fp16* weight,
                                size_t rows, size_t dim, float eps) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, rows * dim * 2), iw(weight, dim * 2), oy(nullptr, rows * dim * 2);
    if (!ii.p || !iw.p || !oy.p) return false;
    struct { uint32_t dim; float eps; } push = {(uint32_t)dim, eps};
    bool r = enc_ew(KR, oy.p, rows * dim * 2, ii.p, rows * dim * 2, iw.p, dim * 2, &push, (uint32_t)rows);
    sync_i();
    if (r) std::memcpy(out, oy.p, rows * dim * 2);
    return r && vk().ok;
}

bool cactus_vulkan_cq_gemv(__fp16* y, const __fp16* x, const CactusQuantMatrix* W,
                           int iters, double* kernel_ms) {
    if (!vk().ok || !W) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const size_t xb = (size_t)W->K * 2, yb = (size_t)W->N * 2;
    HostIn ix(x, xb), oy(nullptr, yb);
    if (!ix.p || !oy.p) return false;
    const int reps = iters > 1 ? iters : 1;
    const bool ts = kernel_ms && vk().ts_period > 0.0f;
    if (!ensure_cmd()) return false;
    if (ts) {
        vk().vkCmdResetQueryPool(cur_cmd(), vk().query, 0, 2);
        vk().vkCmdWriteTimestamp(cur_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk().query, 0);
    }
    for (int i = 0; i < reps; ++i)
        if (!enc_cq_gemv(oy.p, ix.p, W)) return false;
    if (ts)
        vk().vkCmdWriteTimestamp(cur_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk().query, 1);
    sync_i();
    if (!vk().ok) return false;
    std::memcpy(y, oy.p, yb);
    if (ts) {
        uint64_t stamps[2] = {};
        if (vk().vkGetQueryPoolResults(vk().dev, vk().query, 0, 2, sizeof(stamps), stamps,
                                       sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
            *kernel_ms = (double)(stamps[1] - stamps[0]) * vk().ts_period / 1e6;
        else
            *kernel_ms = 0.0;
    } else if (kernel_ms) {
        *kernel_ms = 0.0;
    }
    return true;
}

bool cactus_vulkan_argmax_f16(const __fp16* logits, uint32_t n, uint32_t* idx, float* best) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const uint32_t G = 64;
    HostIn il(logits, (size_t)n * 2), ob(nullptr, G * 2 * sizeof(float));
    if (!il.p || !ob.p) return false;
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(il.p, (size_t)n * 2, &infos[0]) || !dinfo(ob.p, G * 2 * sizeof(float), &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KA, infos, 2);
    if (!set) return false;
    struct { uint32_t n; } push = {n};
    dispatch(KA, set, &push, G, 1, infos, 2);
    sync_i();
    if (!vk().ok) return false;
    const float* parts = (const float*)ob.p;
    float b = parts[0];
    uint32_t bi;
    std::memcpy(&bi, &parts[1], sizeof(bi));
    for (uint32_t i = 1; i < G; ++i) {
        uint32_t ci;
        std::memcpy(&ci, &parts[2 * i + 1], sizeof(ci));
        if (parts[2 * i] > b || (parts[2 * i] == b && ci < bi)) { b = parts[2 * i]; bi = ci; }
    }
    *best = b;
    *idx = bi;
    return true;
}

#endif
