# bench/RESULTS.md - the driver's numbers, with the commands that produced them

M1 Max (32 GB), Asahi Linux 7.1.12, honeykrisp-strata v3 (Mesa 26.3.0-devel, `f144c7949b`), 2026-10-20. Every run
picks the private ICD:

```sh
export VK_DRIVER_FILES=$HOME/.local/lib/honeykrisp-strata/v3/share/vulkan/icd.d/asahi_icd.aarch64.json
```

## Load width (`bw_width`)

`./build/vkprobe --mib 512 bw_width` reads 512 MiB with one load width and counts the *useful* bytes (best of 6
dispatches, three grid sizes of 256 threads):

| the load | GB/s |
|---|---|
| 16 bytes (uvec4) | 294.9 / 299.9 / 299.5 |
| 8 bytes (uint2) | 300.7 / 290.7 / 286.4 |
| 4 bytes (uint) | 281.7 / 281.6 / 263.6 |
| 2 bytes (ushort) | **70.7 / 71.8 / 70.0** |
| 16-byte load, 4 bytes used (the piece at +2 of the chunk) | 273.7 / 270.7 / 268.5 |
| 16-byte load, 4 bytes used (the piece at 0) | 282.5 / 282.3 / 267.3 |
| 16-byte load, 2 bytes used (a scale) | 126.5 / 124.9 / 116.3 |
| the Q2_0 pattern with 4-byte-aligned loads | 275.7 / 269.2 / 270.4 |

**A 2-byte load reaches a quarter of the bandwidth; a 4-byte load is as good as a 16-byte one.** A 4-byte load may
sit at any 2-byte aligned address: the Q2_0 pattern (a 4-byte read of the unit's four code bytes at +2 of an 18-byte
block, then the block's 2-byte scale) reaches 322 GB/s on the Strata side (`vulkan/tests/bw_q2.cpp`).

The rows with "used" are what the earlier note ("the weights must be read with 16-byte loads") was built on: a 16-byte
load of which only 4 bytes are used is *slower* than a plain 4-byte load of the same data (269-283 against 264-282
GB/s, and the fetch is four times bigger).

The last row of the table, the block's 16 code bytes read as two 16-byte chunks, is not a bandwidth number: it counts
4 useful bytes a unit and reads 32 a block, so its fetch is 1.8 times the useful bytes.

## The Q2_0 weight reads, one pattern at a time (Strata's tree)

`build-vulkan/bw_q2 512 8` (the Strata repo) measures six patterns in Slang, with the Q2_0 layout:

| the pattern | GB/s |
|---|---|
| 4-byte codes at +2 + 2-byte scale (what the kernel does) | **322.7** |
| 4-byte codes at +2 + 4-byte scale | 291.7 |
| two 4-byte words for the codes + 4-byte scale | 217.7 |
| 16-byte load for the codes + 4-byte scale | 236.9 |
| two 4-byte words for the codes, no scale | 223.8 |
| two 16-byte loads a block | 288.3 |

The kernel's own pattern is the fastest of the six: the 4-byte read of the codes and the 2-byte read of the scale (one
a block) are not the problem. The Strata side's results and what to do next are in
`~/Work/strata-src/Strata/vulkan/RESULTS.md`.

## What was tried and dropped

- **Staging the weights through shared memory** (a probe that stages a row with 16-byte loads and then reads the
  units from shared memory): the numbers did not scale with the buffer size (the same GB/s for 1 MiB and for
  512 MiB), so they say nothing about the bandwidth and the probe was dropped. The Q2_0 pattern above answers the
  question anyway: the weight path is not the limit, so staging it would not help.
- **Reading a misaligned field as two 4-byte words** (to make every load 4-byte aligned): two loads and a 64-bit
  merge per field. The Q2_0 GEMV fell from 162 to 121 GB/s.

## The other probes (unchanged)

`bw` (uvec4 grid-stride reads) is 288-292 GB/s at 1 GiB; the handshake and coherence numbers are in the README's
"Measured" table.
