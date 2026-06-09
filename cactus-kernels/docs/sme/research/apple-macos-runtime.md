# Apple Silicon / macOS SME2 Runtime & Build Notes (M4 Pro, Apple clang 17)

Practical specifics for compiling and running hand-written SME2 code on an Apple M4 Pro
with Apple clang 17. Where possible, claims are corroborated against published sources AND
verified empirically on the target box.

## Verified test box (this machine)

```
Apple M4 Pro
  10 P-cores (hw.perflevel0.physicalcpu = 10), 4 E-cores (hw.perflevel1.physicalcpu = 4)
Apple clang version 17.0.0 (clang-1700.6.3.2)   Target: arm64-apple-darwin25.3.0
Xcode 26.2 (Build 17C52)
macOS (Darwin 25.3.0)
```

Local sysctl (all SME keys read directly):

```
hw.optional.arm.FEAT_SME       = 1
hw.optional.arm.FEAT_SME2      = 1
hw.optional.arm.FEAT_SME2p1    = 0
hw.optional.arm.FEAT_SME_I16I64 = 1
hw.optional.arm.FEAT_SME_F64F64 = 1
hw.optional.arm.FEAT_SME_F16F16 = 0
hw.optional.arm.FEAT_SME_B16B16 = 0
hw.optional.arm.SME_I8I32      = 1
hw.optional.arm.SME_I16I32     = 1
hw.optional.arm.SME_F32F32     = 1
hw.optional.arm.SME_F16F32     = 1
hw.optional.arm.SME_B16F32     = 1
hw.optional.arm.SME_BI32I32    = 1
hw.optional.arm.sme_max_svl_b  = 64        # 64 bytes -> SVL = 512 bits
```

---

## 1. The exact `-march` / `-mcpu` string for SME2 intrinsics (Apple clang 17)

**Recommendation: `-march=armv8-a+sme2`** (use `+sme` if you only need SME1 ACLE).
This is what the published Apple-M4 walkthroughs actually use and what runs here.

### Empirically tested on this box (compiling SME2 intrinsics: `svld1_f32`, `svmopa_za32_f32_m`, `__arm_locally_streaming __arm_new("za")`)

| Flag | Compiles SME2 intrinsics? | Notes |
|------|---------------------------|-------|
| (no flag, default) | **FAIL** | `error: function executed in streaming-SVE mode requires 'sme'` |
| `-march=armv8-a+sme`  | OK | SME1 ACLE only; SME2-only intrinsics (e.g. multi-vector) need `+sme2` |
| `-march=armv8-a+sme2` | OK | **Recommended.** Matches mod_poppo / dev.to articles' `+sme` form |
| `-march=armv9-a+sme2` | OK (compiles) | Also accepted; see caveat below |
| `-mcpu=apple-m4`      | OK (compiles) | Enables SME for the M4 target |
| `-mcpu=apple-m4+sme2` | OK (compiles) | |

The compiled SME2 binary **runs** on the M4 Pro and reports streaming vector length:

```
streaming VL bytes (svcntsb) = 64
streaming VL bits            = 512
```

i.e. SVL = 512 bits = 64 bytes, matching `hw.optional.arm.sme_max_svl_b = 64` and
tzakharko's "512-bits ... each vector register is 64 bytes ... ZA tile = 4096 bytes."

### Caveats

- **`-march=armv8-a+sme` is the form used by the published M4 sources.** The Zenn
  (mod_poppo) article and the dev.to port both compile with literally
  `clang -O2 -march=armv8-a+sme`. The architecture base is `armv8-a`, *not* `armv9`,
  on Apple's toolchain examples. SME is added as a `+sme`/`+sme2` feature on top, so the
  base version is largely irrelevant to whether SME intrinsics compile.
- **`-march=armv9-a+sme2` compiles, but armv9-a is misleading on Apple.** Apple does not
  publicly position M4 as a full ARMv9 part, and crucially `armv9-a` implies non-streaming
  SVE2 — which M4 does **not** implement (see §3). The base profile does not gate SME
  feature availability, so prefer the `armv8-a+sme2` spelling to avoid implying full SVE.
- **`-mcpu=apple-m4` is accepted by clang 17** and turns on SME, but ties the object to
  that CPU's tuning. For portable kernels prefer the explicit `+sme2` feature flag; gate
  the actual call at runtime (§2) so the binary still loads on non-SME Macs.
- clang validates the *intrinsics against the feature flag*, not against the real target
  CPU — so all the "compiles OK" rows above succeed even where the spelling is dubious.
  Runtime legality is a separate question (§3).

---

## 2. Runtime feature detection (sysctl key spellings)

Use `sysctlbyname` and treat a returned integer `1` as "present". Key spellings
**verified by direct read on this box** (all returned the stated values):

```c
// returns 1 on M4 Pro
sysctlbyname("hw.optional.arm.FEAT_SME",        ...) == 1   // = 1
sysctlbyname("hw.optional.arm.FEAT_SME2",       ...) == 1   // = 1
sysctlbyname("hw.optional.arm.FEAT_SME_I16I64", ...) == 1   // = 1
sysctlbyname("hw.optional.arm.sme_max_svl_b",   ...)        // = 64 (bytes -> 512-bit SVL)
```

- Key spellings confirmed: `hw.optional.arm.FEAT_SME`, `hw.optional.arm.FEAT_SME2`,
  `hw.optional.arm.FEAT_SME_I16I64`, `hw.optional.arm.sme_max_svl_b`. (The `FEAT_`
  prefix is capitalized; the SVL key is lowercase `sme_max_svl_b`.) These follow Apple's
  documented "Determining Instruction Set Characteristics" `hw.optional.arm.FEAT_*` scheme.
- `sme_max_svl_b` returns the **max Streaming Vector Length in bytes** (64 here). Multiply
  by 8 for bits (512). This matches the runtime `svcntsb()` result.
- Apple silicon also exposes per-datatype keys: `SME_I8I32`, `SME_I16I32`, `SME_F32F32`,
  `SME_F16F32`, `SME_B16F32`, `SME_BI32I32`, plus `FEAT_SME_F64F64` — all `1` here. Note
  these are *not* `FEAT_`-prefixed (e.g. `hw.optional.arm.SME_I16I32`), whereas the
  64-bit-accumulate variant is `FEAT_SME_I16I64`. `FEAT_SME2p1`, `FEAT_SME_F16F16`,
  `FEAT_SME_B16B16` are `0` on M4 (not implemented).
- macOS 15+ exposes a fast-path bitmask `hw.optional.arm.caps` (returned a packed integer
  here) if you want all feature bits in one call; individual keys remain the portable path.
- ACLE alternative: `#include <arm_sme.h>` and call `__arm_has_sme()` from non-streaming
  code as a guard before dispatching into a streaming kernel.

### Gotcha: don't rely on the Go stdlib for SME keys

`src/internal/cpu/cpu_arm64_darwin.go` (Go master) only probes
`armv8_1_atomics`, `armv8_crc32`, `armv8_2_sha512`, `armv8_2_sha3`, `FEAT_DIT`, `FEAT_SB`.
It does **not** enumerate any SME key — so it is not a citation for SME spellings, only a
template for the `sysctlbyname`-probe pattern. Apple's developer docs + the live sysctl on
this box are the authoritative source for the SME key names.

---

## 3. ZA / streaming-mode runtime behavior on macOS

### Non-streaming SVE is NOT available — it SIGILLs

M4 implements only the **streaming** SVE subset (inside SME), not regular SVE/SVE2.
Verified on this box: a plain non-streaming `svcntw()` call **crashes with SIGILL**
(process exit code 132 = 128 + SIGILL(4)). The published sources agree:
"the Apple M4 does *not* support non-streaming SVE, so calling `svcnt*` functions outside
streaming mode will cause a SIGILL." Practical rule: **every SVE/SME intrinsic must run
inside streaming mode** (a function marked `__arm_streaming`, `__arm_streaming_compatible`,
or `__arm_locally_streaming`). Do not call SVE intrinsics from ordinary code.

### Kernel auto save/restore of ZA / streaming state (XNU)

Per the XNU `doc/arm/sme.md`, the kernel **does** transparently manage SME state across
context switches — you do not save/restore ZA yourself:

- `machine_switch_context()` saves/restores `TPIDR2_EL0`, `ZA`, and `ZT0`.
- It is **lazy/validity-aware**: `machine_save_sme_context()` reads `SVCR.ZA` and skips
  saving when ZA/ZT0 are not actually live; lazy allocation defers ZA storage until an
  SME trap (clears `SCTLR_EL1.SMEN` in `machine_restore_sme_context()`).
- On every kernel entry from EL0 with `PSTATE.SM` set, XNU saves Z/P/SVCR and **clears
  `PSTATE.SM`** (in `locore.s`), because the kernel itself uses NEON SIMD that is illegal
  in streaming mode. Kernel code is forbidden from entering streaming SVE mode.
- Thread state is reachable via Mach flavors `ARM_SME_STATE` / `ARM_SME2_STATE`
  (`arm_sme_state_t`, holding SVCR, TPIDR2_EL0, SVL); stored in `thread->machine.usme`.

**Net:** preemption and normal context switches are safe and invisible — the OS preserves
ZA/streaming state for you.

### Is `__arm_new("za")` sufficient?

Yes, for ordinary use. With `__arm_new("za")` clang sets up a fresh ZA context on function
entry and disables it before return; if ZA is dormant it emits the lazy-save commit. With
`__arm_locally_streaming` clang toggles `PSTATE.SM` at the function boundary for you
("Clang manages PSTATE.SM automatically; it is not the source code's responsibility").
You generally never write `smstart`/`smstop` or touch TPIDR2 by hand. Use
`__arm_in/out/inout/preserves("za")` to thread ZA across a call boundary without re-init.

### Signal-handler hazards (the real caveat)

This is the sharp edge, and it is **under-documented** by Apple:

- XNU's own note flags the core hazard: "xnu has in-kernel SIMD instructions which become
  illegal while the CPU is in streaming SVE mode. This poses a problem if xnu interrupts
  EL0 while it is in the middle of executing SME-accelerated code." The kernel handles this
  for its own re-entry, but it means async interruption mid-streaming is a real code path.
- XNU states it **does not** send SME/SVE thread state in Mach exception messages, and the
  doc does not specify what `PSTATE.SM`/ZA look like inside a delivered POSIX signal handler
  or whether SME state lands in the signal `ucontext`/`mcontext`. **Treat it as unspecified.**
- Practical guidance for our kernels:
  - **Do not execute SME/streaming or SVE intrinsics inside a signal handler.** A handler
    may run in non-streaming mode where those instructions SIGILL, and ZA may not be the
    live tile you expect.
  - Keep streaming regions short and self-contained; don't `longjmp`/`setjmp` across a
    `__arm_locally_streaming` boundary (state-toggle code may be skipped).
  - Don't assume signal-interrupted ZA is recoverable from the handler; rely on the kernel
    to restore it on normal return, not on hand-inspecting the signal context.

### P-core vs E-core performance (and the ~5x)

- There is **one SME unit per CPU cluster**, shared by all cores in that cluster: a larger
  SME block in the P-core cluster, a smaller one in the E-core cluster. Throughput is a
  *cluster* resource — piling more threads onto the same cluster does **not** multiply SME
  throughput; it is gated by the shared unit.
- The P-core SME unit sustains ~**2.0–2.3 FP32 TFLOPS** (FMOPA outer product; ~2009 GFLOPS
  measured single-P-core). The E-core SME block is characterized as "much slower."
- The **~5x P-over-E** figure is an order-of-magnitude estimate consistent with the sources
  rather than a single published SME benchmark: M4 E-cores are roughly half the compute
  width and run at far lower clocks (general M4 measurements show E-cores ~4x slower on
  CPU work), and the E-cluster SME block is physically smaller. So **expect roughly 4–5x
  worse SME throughput on E-cores**; target the P-cluster for SME kernels.
- **Thread affinity on macOS:** there is **no `pthread_setaffinity_np`** and no public API
  to pin a thread to a specific core/cluster. The realistic lever is **QoS classes**: high
  QoS (`QOS_CLASS_USER_INTERACTIVE`, or `USER_INITIATED`) biases the scheduler toward
  **P-cores**; low QoS (`UTILITY`/`BACKGROUND`) pushes work to **E-cores**. Set it via
  `pthread_set_qos_class_self_np(...)` or a `dispatch_queue` with the desired QoS. This is a
  *hint*, not a guarantee — there is no hard pinning. (The deprecated Mach
  `THREAD_AFFINITY_POLICY` is advisory and effectively a no-op for core selection on Apple
  silicon.) For maximum SME throughput: run the streaming kernel on a high-QoS thread and
  avoid oversubscribing the cluster.

---

## 4. Minimum toolchain for SME ACLE; known Apple-clang bugs

- **Minimum: Apple clang 17 / Xcode 26 era.** Published working setups: macOS 15.7.1,
  Xcode 26.0.1, Apple clang 17.0.0 (dev.to / mod_poppo). This box: Apple clang 17.0.0
  (clang-1700.6.3.2), Xcode 26.2 — confirmed building & running SME2. SME ACLE
  (`<arm_sme.h>`, `__arm_streaming`, `__arm_new("za")`, `__arm_locally_streaming`) requires
  a recent LLVM/clang; older Xcode (pre-26) toolchains predate stable Apple SME ACLE support
  and should be considered unsupported for hand-written SME2.
- **Known LLVM/clang hazard (LLVM issue #86743):** with only `+sme` (no streaming context
  established), LLVM could emit SVE instructions that are illegal outside streaming mode —
  which on M4 means SIGILL. Mitigation: always wrap intrinsics in a properly attributed
  streaming function (`__arm_locally_streaming` / `__arm_streaming`) and build with `+sme2`;
  do not hand-emit SVE expecting non-streaming execution.
- Because clang accepts SME intrinsics under any of the §1 flags regardless of real target
  legality, **always gate the streaming entry point behind a runtime `FEAT_SME2` sysctl
  check (or `__arm_has_sme()`)** so the binary still loads/runs on non-SME Macs and you
  never hit a SIGILL on unsupported hardware.

---

## Quick reference (copy/paste)

```sh
# Build
clang -O2 -march=armv8-a+sme2 kernel.c -o kernel     # recommended
# (intrinsics also compile under armv9-a+sme2 / -mcpu=apple-m4, but prefer the above)
```

```c
#include <arm_sme.h>
#include <sys/sysctl.h>

static int has_sme2(void){
    int v=0; size_t sz=sizeof(v);
    return sysctlbyname("hw.optional.arm.FEAT_SME2",&v,&sz,0,0)==0 && v==1;
}

// All SVE/SME intrinsics MUST be inside a streaming + ZA scope.
__arm_locally_streaming __arm_new("za")
static void kernel(/*...*/){
    // svcntsb() == 64 here (512-bit SVL); ZA tile == 4096 bytes
    // ... svmopa_za32_f32_m(...), etc.
}

int main(void){
    if(!has_sme2()) return run_fallback();   // never SIGILL on non-SME hardware
    kernel(/*...*/);                          // run on a high-QoS thread for P-cluster
}
```

## Sources
- mod_poppo / Zenn, "Trying Out Arm's SME" (`-march=armv8-a+sme`, non-streaming SVE SIGILL, `__arm_locally_streaming`/`__arm_new("za")`, svcntw=16): https://zenn.dev/mod_poppo/articles/arm-scalable-matrix-extension?locale=en
- dev.to port of the above (versions: macOS 15.7.1 / Xcode 26.0.1 / clang 17.0.0; SIGILL quote): https://dev.to/aratamizuki/trying-out-arms-scalable-matrix-extension-with-apple-m4-or-qemu-1cgh
- tzakharko/m4-sme-exploration (512-bit SVL, 64-byte vectors, 4096-byte ZA, one SME block per cluster, P-core ~2 TFLOPS FP32, E-core "much slower", non-streaming SVE absent): https://github.com/tzakharko/m4-sme-exploration/blob/main/reports/01-sme-overview.md
- XNU `doc/arm/sme.md` (kernel save/restore of TPIDR2_EL0/ZA/ZT0, lazy save, PSTATE.SM cleared on kernel entry, ARM_SME_STATE/ARM_SME2_STATE, no SME state in Mach exceptions, streaming-vs-kernel-SIMD hazard): https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md
- Clang AttributeReference (SME attributes: __arm_streaming, __arm_streaming_compatible, __arm_locally_streaming, __arm_new("za"), __arm_in/out/inout/preserves): https://clang.llvm.org/docs/AttributeReference.html
- LLVM issue #86743 (SVE emitted under +sme without streaming -> illegal outside streaming): https://github.com/llvm/llvm-project/issues/86743
- Go `cpu_arm64_darwin.go` (sysctlbyname probe pattern; does NOT list SME keys): https://github.com/golang/go/blob/master/src/internal/cpu/cpu_arm64_darwin.go
- Apple Developer, "Determining Instruction Set Characteristics" (hw.optional.arm.FEAT_* sysctl scheme): https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
- Apple Developer Forums (no pthread_setaffinity_np on macOS; QoS classes drive P/E placement): https://developer.apple.com/forums/thread/674456
- Eclectic Light, "Inside M4 chips: E and P cores" (E-cores ~4x slower, ~half compute units, QoS-driven placement): https://eclecticlight.co/2024/11/18/inside-m4-chips-e-and-p-cores/
- Local verification: live `sysctl` reads + compiled/ran SME2 binary on this M4 Pro (see §Verified test box).
