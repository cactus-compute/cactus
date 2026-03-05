// Weak stubs for ARM SME2 ABI runtime helpers.
// On M4+ systems, libSystem provides the real implementations which override these.
// On M1/M2/M3 systems, these stubs satisfy the linker; the SME2 code path is never
// reached because cpu_has_sme2() returns false at runtime.

__attribute__((weak))
void __arm_tpidr2_save(void) {}

__attribute__((weak))
void __arm_tpidr2_restore(void) {}
