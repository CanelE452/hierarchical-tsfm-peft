# Measurement implementation notes

Recorded before the first new experimental measurement, after the protocol seal.

- Every CUDA facet performs a single synchronize → empty_cache → synchronize after its one model load/one-time merge and before the loaded baseline. This releases unused load/merge allocator cache uniformly. No empty_cache occurs between warmup and timed trials. Active tensor contents and the registered precision/schedule do not change.
- The legacy LoRA merge-parity helper calls F directly. The new wrapper explicitly uses the facet's actual row chunk for this one pre-timing check, then restores the F attribute; each normal whole-path call sets both route chunk keys as registered.
- An unrelated live Python GPU process, an older campaign's live lease, device utilization at least80%, or device memory usage at least90% prevents a new facet from starting. These preflight device values are resource context, never per-PID measurements. Desktop applications are preserved. Preflight occurs before each independent process.
- Invalid attempts remain recorded and excluded. A failed mandatory comparison makes Phase B ineligible; valid Phase A comparisons still finish. A same-facet implementation-repair retry archives the old receipt/log and verifies the old worker has terminated.
- Parent accuracy rows are SHA-bound before the worker's loader can enrich their metadata. The original history file may only be appended; its original bytes must remain a prefix.

These choices clarify cache, failure and resource handling; they do not alter the preregistered scientific candidates, score tolerances, confirmation rule, K recipe or gate.
