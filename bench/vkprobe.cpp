// vkprobe - the Vulkan numbers a Strata decode window depends on, on Honeykrisp (Apple GPU under Asahi Linux).
//
//   bw         streaming read bandwidth (uvec4 grid-stride loads), GB/s, over workgroup counts
//   dispatch   GPU time per small dependent dispatch in one command buffer (with and without barriers)
//   rt_submit  host<->GPU handshake as one submit + fence wait per layer
//   rt_tl      the same with one batch of timeline-semaphore submits (wait-before-signal, host signals)
//   rt_spin    the same inside ONE command buffer: the GPU spins on a host-written flag (Metal's shared-event shape)
//
// Choose a driver per process with VK_DRIVER_FILES=<icd.json>.  Buffers are small (1 GiB at most).
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Ctx {
    VkInstance inst{};
    VkPhysicalDevice pd{};
    VkDevice dev{};
    VkQueue q{};
    uint32_t qf = 0;
    VkCommandPool pool{};
    VkQueryPool qp{};
    VkDescriptorPool dpool{};
    double ts_period_ns = 1;
    VkPhysicalDeviceMemoryProperties mp{};
};

struct Buf { VkBuffer b{}; VkDeviceMemory m{}; void* p = nullptr; VkDeviceSize size = 0; };

struct Pipe { VkPipeline p{}; VkPipelineLayout l{}; VkDescriptorSetLayout dsl{}; VkDescriptorSet ds{}; };

static std::string g_shader_dir = "build";

static Ctx init() {
    Ctx c;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "vkprobe";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    CK(vkCreateInstance(&ici, nullptr, &c.inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
    c.pd = VK_NULL_HANDLE;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { c.pd = p; break; }
    }
    if (!c.pd) { fprintf(stderr, "no GPU\n"); exit(1); }
    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 pr2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(c.pd, &pr2);
    c.ts_period_ns = pr2.properties.limits.timestampPeriod;
    printf("device: %s | driver: %s %s\n", pr2.properties.deviceName, drv.driverName, drv.driverInfo);
    vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);

    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(n);
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &n, qfs.data());
    for (uint32_t i = 0; i < n; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
    if (!qfs[c.qf].timestampValidBits) printf("warning: no timestamps on this queue\n");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
    f12.shaderInt8 = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE;
    f12.vulkanMemoryModelDeviceScope = VK_TRUE;
    f12.hostQueryReset = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f2.features.shaderInt16 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(c.pd, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.q);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = c.qf;
    CK(vkCreateCommandPool(c.dev, &cpi, nullptr, &c.pool));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    CK(vkCreateQueryPool(c.dev, &qpi, nullptr, &c.qp));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 64;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    CK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool));
    return c;
}

static Buf make_buf(Ctx& c, VkDeviceSize size) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    CK(vkCreateBuffer(c.dev, &bci, nullptr, &b.b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c.dev, b.b, &mr);
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (c.mp.memoryTypes[i].propertyFlags & want) == want) { type = i; break; }
    if (type == UINT32_MAX) { fprintf(stderr, "no host-visible coherent device-local memory type\n"); exit(1); }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = type;
    CK(vkAllocateMemory(c.dev, &mai, nullptr, &b.m));
    CK(vkBindBufferMemory(c.dev, b.b, b.m, 0));
    CK(vkMapMemory(c.dev, b.m, 0, VK_WHOLE_SIZE, 0, &b.p));
    return b;
}

static void free_buf(Ctx& c, Buf& b) {
    vkUnmapMemory(c.dev, b.m);
    vkDestroyBuffer(c.dev, b.b, nullptr);
    vkFreeMemory(c.dev, b.m, nullptr);
    b = {};
}

static std::vector<uint32_t> load_spv(const std::string& name) {
    std::ifstream f(g_shader_dir + "/" + name + ".spv", std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "missing %s/%s.spv\n", g_shader_dir.c_str(), name.c_str()); exit(1); }
    size_t n = (size_t) f.tellg();
    std::vector<uint32_t> v(n / 4);
    f.seekg(0);
    f.read((char*) v.data(), (std::streamsize) n);
    return v;
}

static Pipe make_pipe(Ctx& c, const char* name, std::vector<Buf*> bufs, uint32_t push_bytes, uint32_t wg = 0) {
    Pipe p;
    std::vector<VkDescriptorSetLayoutBinding> bs;
    for (uint32_t i = 0; i < bufs.size(); i++)
        bs.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = (uint32_t) bs.size();
    dli.pBindings = bs.data();
    CK(vkCreateDescriptorSetLayout(c.dev, &dli, nullptr, &p.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = push_bytes ? 1 : 0;
    pli.pPushConstantRanges = &pcr;
    CK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &p.l));
    auto spv = load_spv(name);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spv.size() * 4;
    smi.pCode = spv.data();
    VkShaderModule sm;
    CK(vkCreateShaderModule(c.dev, &smi, nullptr, &sm));
    VkSpecializationMapEntry me{0, 0, 4};
    VkSpecializationInfo si{1, &me, 4, &wg};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main",
                 wg ? &si : nullptr};
    cpi.layout = p.l;
    CK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.p));
    vkDestroyShaderModule(c.dev, sm, nullptr);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = c.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &p.dsl;
    CK(vkAllocateDescriptorSets(c.dev, &dai, &p.ds));
    std::vector<VkDescriptorBufferInfo> bis;
    for (auto* b : bufs) bis.push_back({b->b, 0, VK_WHOLE_SIZE});
    std::vector<VkWriteDescriptorSet> ws;
    for (uint32_t i = 0; i < bufs.size(); i++) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = p.ds;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bis[i];
        ws.push_back(w);
    }
    vkUpdateDescriptorSets(c.dev, (uint32_t) ws.size(), ws.data(), 0, nullptr);
    return p;
}

static void destroy_pipe(Ctx& c, Pipe& p) {
    vkDestroyPipeline(c.dev, p.p, nullptr);
    vkDestroyPipelineLayout(c.dev, p.l, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, p.dsl, nullptr);
}

static VkCommandBuffer new_cb(Ctx& c) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = c.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(c.dev, &ai, &cb));
    return cb;
}

static void begin(VkCommandBuffer cb) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
}

static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

static void submit_wait(Ctx& c, VkCommandBuffer cb) {
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    CK(vkCreateFence(c.dev, &fi, nullptr, &f));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(c.q, 1, &si, f));
    CK(vkWaitForFences(c.dev, 1, &f, VK_TRUE, UINT64_MAX));
    vkDestroyFence(c.dev, f, nullptr);
}

static double gpu_ms(Ctx& c) {
    uint64_t ts[2];
    CK(vkGetQueryPoolResults(c.dev, c.qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double) (ts[1] - ts[0]) * c.ts_period_ns / 1e6;
}

// ---------------------------------------------------------------------------------------------------------------

static void test_bw(Ctx& c, size_t mib) {
    Buf src = make_buf(c, (VkDeviceSize) mib << 20);
    memset(src.p, 0x5a, src.size);
    const uint32_t max_threads = 1u << 20;
    Buf dst = make_buf(c, max_threads * 4ull);
    uint32_t n_vec4 = (uint32_t) (src.size / 16);
    printf("\n[bw] %zu MiB read, best of 5 (GB/s, wall clock over 8 dispatches)\n", mib);
    for (uint32_t wg : {128u, 256u, 1024u}) {
        Pipe p = make_pipe(c, "bw_read", {&src, &dst}, 4, wg);
        printf("  wg %4u:", wg);
        for (uint32_t groups : {64u, 256u, 1024u, 4096u}) {
            if ((uint64_t) groups * wg > max_threads) { printf("        -"); continue; }
            double best = 1e30;
            const int D = 8;   // dispatches per submission; wall clock (Honeykrisp's compute timestamps are not usable)
            VkCommandBuffer cb = new_cb(c);
            begin(cb);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
            vkCmdPushConstants(cb, p.l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_vec4);
            for (int d = 0; d < D; d++) { vkCmdDispatch(cb, groups, 1, 1); barrier(cb); }
            CK(vkEndCommandBuffer(cb));
            for (int rep = 0; rep < 6; rep++) {
                double t0 = now_us();
                submit_wait(c, cb);
                double ms = (now_us() - t0) / 1000.0 / D;
                if (rep) best = std::min(best, ms);   // rep 0 warms caches/pipeline
            }
            vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
            printf("  %5u:%6.1f", groups, (double) src.size / (best * 1e6));
        }
        printf("\n");
        destroy_pipe(c, p);
    }
    free_buf(c, src);
    free_buf(c, dst);
}

// The Q2_0 GEMV's access pattern without its arithmetic (bw_q2.comp): GB/s of weight bytes (18 per 64 weights).
static void test_bw_q2(Ctx& c, size_t mib) {
    Buf src = make_buf(c, (VkDeviceSize) mib << 20);
    memset(src.p, 0x5a, src.size);
    const uint32_t max_threads = 1u << 20;
    Buf dst = make_buf(c, max_threads * 4ull);
    const uint32_t n_units = (uint32_t) (src.size / 18 * 4);
    printf("\n[bw_q2] %zu MiB of Q2_0 blocks read as the GEMV reads them (16-bit loads), GB/s\n", mib);
    for (uint32_t wg : {128u, 256u}) {
        Pipe p = make_pipe(c, "bw_q2", {&src, &dst}, 4, wg);
        printf("  wg %4u:", wg);
        for (uint32_t groups : {64u, 256u, 1024u, 4096u}) {
            if ((uint64_t) groups * wg > max_threads) { printf("        -"); continue; }
            double best = 1e30;
            const int D = 8;
            VkCommandBuffer cb = new_cb(c);
            begin(cb);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
            vkCmdPushConstants(cb, p.l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_units);
            for (int d = 0; d < D; d++) { vkCmdDispatch(cb, groups, 1, 1); barrier(cb); }
            CK(vkEndCommandBuffer(cb));
            for (int rep = 0; rep < 6; rep++) {
                double t0 = now_us();
                submit_wait(c, cb);
                double ms = (now_us() - t0) / 1000.0 / D;
                if (rep) best = std::min(best, ms);
            }
            vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
            printf("  %5u:%6.1f", groups, (double) src.size / (best * 1e6));
        }
        printf("\n");
        destroy_pipe(c, p);
    }
    free_buf(c, src);
    free_buf(c, dst);
}

static void test_dispatch(Ctx& c) {
    Buf b = make_buf(c, 4096);
    Pipe p = make_pipe(c, "tiny", {&b}, 0);
    const int N = 2000;
    printf("\n[dispatch] %d tiny dispatches in one command buffer (GPU us per dispatch; host wall us per dispatch)\n", N);
    for (int with_barrier = 1; with_barrier >= 0; with_barrier--) {
        double best_gpu = 1e30, best_wall = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            *(uint32_t*) b.p = 0;
            VkCommandBuffer cb = new_cb(c);
            begin(cb);
            vkCmdResetQueryPool(cb, c.qp, 0, 2);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, c.qp, 0);
            for (int i = 0; i < N; i++) {
                vkCmdDispatch(cb, 1, 1, 1);
                if (with_barrier) barrier(cb);
            }
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, c.qp, 1);
            CK(vkEndCommandBuffer(cb));
            double t0 = now_us();
            submit_wait(c, cb);
            double wall = now_us() - t0;
            vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
            if (rep) {
                best_gpu = std::min(best_gpu, gpu_ms(c) * 1000 / N);
                best_wall = std::min(best_wall, wall / N);
            }
        }
        printf("  %-16s gpu %6.2f us   wall %6.2f us   (counter %u, expect %d)\n", with_barrier ? "with barriers" : "no barriers",
               best_gpu, best_wall, *(uint32_t*) b.p, with_barrier ? N : -1);
    }
    destroy_pipe(c, p);
    free_buf(c, b);
}

static void test_rt_submit(Ctx& c, int K) {
    Buf b = make_buf(c, 4096);
    Pipe p = make_pipe(c, "tiny", {&b}, 0);
    VkCommandBuffer cb = new_cb(c);
    begin(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);
    CK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    CK(vkCreateFence(c.dev, &fi, nullptr, &f));
    std::vector<double> per;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now_us();
        for (int i = 0; i < K; i++) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            CK(vkQueueSubmit(c.q, 1, &si, f));
            CK(vkWaitForFences(c.dev, 1, &f, VK_TRUE, UINT64_MAX));
            CK(vkResetFences(c.dev, 1, &f));
        }
        if (rep) per.push_back((now_us() - t0) / K);
    }
    std::sort(per.begin(), per.end());
    printf("\n[rt_submit] submit + fence wait per layer, %d layers: median %.1f us per handshake (min %.1f)\n", K,
           per[per.size() / 2], per[0]);
    vkDestroyFence(c.dev, f, nullptr);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    destroy_pipe(c, p);
    free_buf(c, b);
}

static void test_rt_timeline(Ctx& c, int K, bool poll) {
    Buf b = make_buf(c, 4096);
    Pipe p = make_pipe(c, "tiny", {&b}, 0);
    VkCommandBuffer cb = new_cb(c);
    begin(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);
    CK(vkEndCommandBuffer(cb));
    std::vector<double> per;
    for (int rep = 0; rep < 5; rep++) {
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
        VkSemaphore host_sem, gpu_sem;
        CK(vkCreateSemaphore(c.dev, &sci, nullptr, &host_sem));
        CK(vkCreateSemaphore(c.dev, &sci, nullptr, &gpu_sem));
        // segment i (1..K): waits host >= i-1, runs, signals gpu = i.  All submitted up front (wait-before-signal).
        std::vector<VkSubmitInfo2> sub(K);
        std::vector<VkSemaphoreSubmitInfo> waits(K), sigs(K);
        VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbi.commandBuffer = cb;
        for (int i = 0; i < K; i++) {
            waits[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, host_sem, (uint64_t) i,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0};
            sigs[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, gpu_sem, (uint64_t) i + 1,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
            sub[i] = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            sub[i].waitSemaphoreInfoCount = i ? 1 : 0;
            sub[i].pWaitSemaphoreInfos = &waits[i];
            sub[i].commandBufferInfoCount = 1;
            sub[i].pCommandBufferInfos = &cbi;
            sub[i].signalSemaphoreInfoCount = 1;
            sub[i].pSignalSemaphoreInfos = &sigs[i];
        }
        double t0 = now_us();
        CK(vkQueueSubmit2(c.q, (uint32_t) K, sub.data(), VK_NULL_HANDLE));
        for (int i = 1; i <= K; i++) {
            if (poll) {
                uint64_t v = 0;
                do { vkGetSemaphoreCounterValue(c.dev, gpu_sem, &v); } while (v < (uint64_t) i);
            } else {
                uint64_t v = (uint64_t) i;
                VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                wi.semaphoreCount = 1;
                wi.pSemaphores = &gpu_sem;
                wi.pValues = &v;
                CK(vkWaitSemaphores(c.dev, &wi, UINT64_MAX));
            }
            VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            si.semaphore = host_sem;
            si.value = (uint64_t) i;
            CK(vkSignalSemaphore(c.dev, &si));
        }
        double dt = now_us() - t0;
        CK(vkQueueWaitIdle(c.q));
        if (rep) per.push_back(dt / K);
        vkDestroySemaphore(c.dev, host_sem, nullptr);
        vkDestroySemaphore(c.dev, gpu_sem, nullptr);
    }
    std::sort(per.begin(), per.end());
    printf("[rt_tl%s] timeline wait-before-signal, %d layers: median %.1f us per handshake (min %.1f)\n",
           poll ? "_poll" : "", K, per[per.size() / 2], per[0]);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    destroy_pipe(c, p);
    free_buf(c, b);
}

static void test_rt_spin(Ctx& c, int K) {
    Buf f = make_buf(c, 4096);
    Pipe p = make_pipe(c, "spin", {&f}, 8);
    const uint32_t max_iter = 2u * 1000u * 1000u;   // bounded: a missed answer ends the wait instead of hanging
    std::vector<double> per;
    uint32_t timeouts = 0;
    bool ok = true;
    for (int rep = 0; rep < 5 && ok; rep++) {
        uint32_t* fl = (uint32_t*) f.p;
        __atomic_store_n(&fl[0], 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&fl[1], 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&fl[2], 0u, __ATOMIC_SEQ_CST);
        VkCommandBuffer cb = new_cb(c);
        begin(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
        for (int i = 1; i <= K; i++) {
            uint32_t pc[2] = {(uint32_t) i, max_iter};
            vkCmdPushConstants(cb, p.l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
            vkCmdDispatch(cb, 1, 1, 1);
            barrier(cb);
        }
        CK(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        CK(vkCreateFence(c.dev, &fi, nullptr, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        double t0 = now_us();
        CK(vkQueueSubmit(c.q, 1, &si, fence));
        for (int i = 1; i <= K; i++) {
            double ts = now_us();
            while (__atomic_load_n(&fl[0], __ATOMIC_ACQUIRE) < (uint32_t) i) {
                if (now_us() - ts > 2e6) { ok = false; break; }   // the GPU's write never became visible
            }
            if (!ok) { __atomic_store_n(&fl[1], 0xffffffffu, __ATOMIC_SEQ_CST); break; }   // release every spin
            __atomic_store_n(&fl[1], (uint32_t) i, __ATOMIC_RELEASE);
        }
        double dt = now_us() - t0;
        CK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX));
        timeouts += __atomic_load_n(&fl[2], __ATOMIC_ACQUIRE);
        vkDestroyFence(c.dev, fence, nullptr);
        vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
        if (rep && ok) per.push_back(dt / K);
    }
    if (!ok || per.empty()) {
        printf("[rt_spin] GPU writes did not reach the host within one command buffer (timeouts %u)\n", timeouts);
    } else {
        std::sort(per.begin(), per.end());
        printf("[rt_spin] GPU spins on a host flag inside one command buffer, %d layers: median %.1f us per handshake "
               "(min %.1f), GPU-side timeouts %u\n", K, per[per.size() / 2], per[0], timeouts);
    }
    destroy_pipe(c, p);
    free_buf(c, f);
}

// The arena's case inside one command buffer: per layer the host rewrites a 4 MiB slot (an expert read from the
// SSD) after the GPU reached the layer and before it answers; the GPU then reads the slot and counts stale words.
// host_delay_us stands in for the SSD time of a layer's misses (the GPU spins that long).
static void test_coherence(Ctx& c, int K, double host_delay_us, bool read_before, bool split = false) {
    Buf f = make_buf(c, 4096);
    Buf slot = make_buf(c, 4u << 20);
    Pipe p = make_pipe(c, "coherence", {&f, &slot}, 20);
    const uint32_t n_words = (uint32_t) (slot.size / 4), groups = 64;
    const uint32_t max_iter = 50u * 1000u * 1000u;
    uint32_t* fl = (uint32_t*) f.p;
    uint32_t* sw = (uint32_t*) slot.p;
    memset(f.p, 0, f.size);
    for (uint32_t w = 0; w < n_words; w++) sw[w] = 0x12345678u;
    VkCommandBuffer cb = new_cb(c);
    begin(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.l, 0, 1, &p.ds, 0, nullptr);
    for (int i = 1; i <= K; i++) {
        uint32_t pc[5] = {(uint32_t) i, max_iter, n_words, read_before ? 1u : 0u, split ? 1u : 0u};
        vkCmdPushConstants(cb, p.l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, pc);
        vkCmdDispatch(cb, groups, 1, 1);
        barrier(cb);
        if (split) {
            pc[4] = 2;
            vkCmdPushConstants(cb, p.l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, pc);
            vkCmdDispatch(cb, groups, 1, 1);
            barrier(cb);
        }
    }
    CK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CK(vkCreateFence(c.dev, &fi, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    bool ok = true;
    double t0 = now_us();
    CK(vkQueueSubmit(c.q, 1, &si, fence));
    for (int i = 1; i <= K; i++) {
        double ts = now_us();
        while (__atomic_load_n(&fl[0], __ATOMIC_ACQUIRE) < (uint32_t) i)
            if (now_us() - ts > 5e6) { ok = false; break; }
        if (!ok) { __atomic_store_n(&fl[1], 0xffffffffu, __ATOMIC_SEQ_CST); break; }
        const uint32_t want = (uint32_t) i * 0x9e3779b9u;
        for (uint32_t w = 0; w < n_words; w++) sw[w] = want ^ w;
        double td = now_us();
        while (now_us() - td < host_delay_us) {}
        __atomic_store_n(&fl[1], (uint32_t) i, __ATOMIC_RELEASE);
    }
    double dt = now_us() - t0;
    VkResult wr = vkWaitForFences(c.dev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000);
    printf("[coherence%s] %d layers, host turn %.0f us + 4 MiB write, read_before %d: %s, window %.1f ms, "
           "stale words %u of %u, spin timeouts %u\n", split ? "_split" : "", K, host_delay_us, read_before ? 1 : 0,
           !ok ? "GPU never reached a layer" : wr != VK_SUCCESS ? "fence wait failed" : "ok", dt / 1000,
           __atomic_load_n(&fl[3], __ATOMIC_ACQUIRE), __atomic_load_n(&fl[4], __ATOMIC_ACQUIRE),
           __atomic_load_n(&fl[2], __ATOMIC_ACQUIRE));
    vkDestroyFence(c.dev, fence, nullptr);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    destroy_pipe(c, p);
    free_buf(c, f);
    free_buf(c, slot);
}

int main(int argc, char** argv) {
    std::vector<std::string> tests;
    size_t mib = 1024;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--shaders" && i + 1 < argc) g_shader_dir = argv[++i];
        else if (a == "--mib" && i + 1 < argc) mib = (size_t) atoi(argv[++i]);
        else tests.push_back(a);
    }
    if (tests.empty()) tests = {"bw", "dispatch", "rt"};
    Ctx c = init();
    for (auto& t : tests) {
        if (t == "bw") test_bw(c, mib);
        else if (t == "dispatch") test_dispatch(c);
        else if (t == "bw_q2") test_bw_q2(c, mib);
        else if (t == "rt") {
            test_rt_submit(c, 48);
            test_rt_timeline(c, 48, false);
            test_rt_timeline(c, 48, true);
            test_rt_spin(c, 48);
        } else if (t == "rt_spin") test_rt_spin(c, 48);
        else if (t == "coherence") {
            test_coherence(c, 48, 0, false);
            test_coherence(c, 48, 0, true);
            test_coherence(c, 48, 2000, true);    // ~1 ms SSD turn per layer, as the M4's 31-46 ms per token
            test_coherence(c, 8, 50000, true);    // a 50 ms turn: does a long GPU-side wait trip a watchdog?
            test_coherence(c, 48, 0, true, true);
            test_coherence(c, 48, 2000, true, true);
        }
        else fprintf(stderr, "unknown test %s\n", t.c_str());
    }
    vkDeviceWaitIdle(c.dev);
    return 0;
}
