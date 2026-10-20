# honeykrisp-strata

The Apple GPU Vulkan driver (Honeykrisp, Mesa) and the measurements behind a Vulkan port of
[Strata](https://github.com/Niko1221/Strata)'s Mac engine to Asahi Linux, with macOS speed as the target. Strata
runs Qwen3.8-Flash-Next (125B MoE). On a 32 GB M1 Max under macOS it streams the experts from the SSD: 7.9 tok/s
decode, 186 tok/s prompt (Q2_0, upstream's docs/MAC.md). Strata has no Vulkan engine, and Metal does not exist on
Linux. The port replaces Metal with Vulkan.

- `build-icd.sh`: builds the driver as a **private per-process ICD**. The desktop keeps the system Mesa. Opt in
  with `VK_DRIVER_FILES=~/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json`.
  - Base: joshuaswarren/mesa-1 `honeykrisp-omarchy-v3` @ `f144c7949b` (2026-10-12), the omarchy-mlx maintainer's
    M1/M1 Max compute set (upstream as omacom/mesa PRs #3-#6, not merged; #6 says "NOT READY TO MERGE").
  - `patches/` adds this repo's changes on top; it is empty so far.
- `bench/vkprobe`: GPU probes. `bench/ssdprobe`: expert-streaming read throughput.

## Measured (M1 Max 32 GB, Asahi, Linux 7.1.12, 2026-10-20)

Three drivers: the system Mesa 26.2.4, omarchy-mlx-vulkan 0.7.28's private ICD (`e7631595df`, 2026-10-01), and
v3 (`f144c7949b`).

| | stock 26.2.4 | private 10-01 | v3 |
|---|---|---|---|
| Device heap | 15.45 GiB (50%) | 15.45 GiB | **27.81 GiB (90%)** |
| VK_KHR_cooperative_matrix / bf16 | no / no | yes / no | yes / **yes** |
| Streaming read, uvec4 loads (best) | ~320 GB/s | ~320 GB/s | ~320 GB/s |
| Dependent tiny dispatch (barrier between) | 3.5 us | 2.7 us | **2.5 us** |
| Handshake: submit + fence per layer | 115 us | 99 us | 85 us |
| Handshake: timeline wait-before-signal | 97 us | 96 us | 95 us |
| **Handshake: GPU spins on a host flag in one command buffer** | **6.0 us** | **5.4 us** | **5.8 us** |
| Host rewrites a slot mid-command-buffer: stale words seen by GPU | 0 | 0 | 0 |
| 50 ms GPU-side wait per layer (8 layers) | ok | ok | ok |

Notes:
- Honeykrisp's compute timestamps are not usable (they gave 28,000 GB/s), so every time above is wall clock.
- The bandwidth figure varies with grid shape: 64 x 256 and 4096 x 256 threads reach 320 GB/s, other shapes drop
  to 180-230. Kernel tiling will have to be tuned per shape.
- The coherence test checks 50M words: the host writes a 4 MiB slot while the GPU waits, and a later dispatch
  reads it with plain loads after a barrier. This is the access pattern of the streaming arena.

**SSD**: random reads with O_DIRECT, 6-12 threads, 3 GiB per run.

| Chunk | raw partition | dm-crypt (LUKS) | file on btrfs (zstd, on LUKS) |
|---|---|---|---|
| 1792 KiB (one expert, gate+up+down) | 7.15 GB/s | 7.17 GB/s | 6.80-6.94 GB/s |
| 592 KiB (one matrix) | 6.49 GB/s | 6.11 GB/s | 5.39-5.66 GB/s |

LUKS costs nothing at expert-sized reads. Btrfs costs 4-7%; `chattr +C` on a model directory could recover that,
but it is not measured yet. For comparison, upstream measured 2.97 GB/s on the 32 GB M4 mini under macOS.

## What this means for the port

1. **The per-layer handshake** (the router's choice goes to the host, the host reads the missing experts, then the
   GPU goes on) must stay inside one command buffer. The GPU spins on a host-written flag, which mirrors Metal's
   shared-event wait. Per-segment submits cost ~100 us x 48 layers, about 4.8 ms per window. The spin costs ~0.3 ms.
2. **Memory**: on v3 the heap is no longer 50%. The arena's budget comes from `MemAvailable` minus a desktop
   reserve, filled in steps the way the Mac engine's `arena_budget.h` does it. The filled arena is locked with
   `mlock`, as upstream does after its freeze reports.
3. **Kernels**: read bandwidth is ~80% of the 400 GB/s peak. Prompt GEMMs use the v3 cooperative matrices; Metal's
   `simdgroup_matrix` is 8x8 too. The Mesa compiler's codegen versus Apple's is the open question. It is measured
   per kernel as each one is ported: parity first, then GB/s.
4. **SSD**: Asahi's NVMe is faster than the M4's. The experts' contiguous copy (one read per expert) matters more
   than the encryption does.

Ideas from DeepSeek's DeepInfer (released 2026-10-20) for later:
- **Confidence-scheduled verification**: upstream turns MTP drafts off on 32 GB Macs, because each verify step
  reads more experts. Verifying a draft only when its expected gain beats its expert-miss cost could make drafts pay.
- **Overlapping host scheduling with GPU work**: the read-ahead already does part of this.
