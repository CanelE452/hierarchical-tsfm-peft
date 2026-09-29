# Completed diagnostic; accuracy/generalization unresolved

Base repository: main at4c98b9228a9751542b2ff7bd61521c247911f946. User asked to solve the remaining problem. Diagnostic PLAN.md was written before replay; new fitting is a separate proposal in NEXT_REPAIR.md.

[Confirmed] replay01 completed with exit code0, GPU job occupancy44.3113305 seconds, fit0, optimizer update0, TEST access0. Existing v9 initial-score and best-trained VAL prediction replay checks passed, and128 pre-existing v9 public files were unchanged. No previous checkpoints were edited. No commit/push occurred.

Final verification also passed: all12 replayed states' VAL MSE/MAE match the original curve at the saved epoch, including the final restart epoch. Existing tracked files are unchanged. Local output size at verification was4,775,149 bytes. The executed diagnostic source is preserved in the cache with its original receipt hash; the current script adds a guard against overwriting an existing diagnostic on retry. No replay was repeated after that guard.

Same original TRAIN96 origins per seed and full VAL30 origins:

```text
P head/seed         Initial TRAIN   Trained-best TRAIN   Final TRAIN
92601                 0.177448604          0.173352622   0.170258704
92602                 0.172102559          0.169356034   0.166413096

P head/seed         Initial VAL     Trained-best VAL     Final VAL
92601                 0.374481996          0.377401221   0.380616314
92602                 0.374225103          0.376170494   0.380399835
```

Both P heads learned to lower the same TRAIN probe but failed to transfer that improvement to VAL. Final checkpoints amplify that pattern. RAW has the same direction except seed92602's trained-best VAL improvement of0.000063351, which is not evidence for broad recovery. All four runs' complete numbers, paths, hashes and replay checks are in diagnostic.json; arrays are under the ignored cache named in that JSON.

Interpretation: supports a generalization/transfer failure of these learned corrections, not a disconnected optimizer or a guarantee that simply running longer helps. It does not establish covariate-shift causality, optimality of the fitted family, or full-TRAIN behavior from a96-origin probe. The existing Robin P/Q limitation and Jena LoRA/direct-linear counterexamples remain.

Next concrete decision: whether to run the single matched ridge repair defined in NEXT_REPAIR.md (proposed16 basic CPU fits+2 technical recovery; no new neural training). This new rank/regularization/fitting contract has not been approved or executed. The goal of improved accuracy and practical cross-domain value is still unresolved.

Subsequent update: the paragraph above records the state when this diagnostic ended. The user later approved NEXT_REPAIR with `그렇게 해줘`; its separate16-fit execution and negative/limited results are now recorded in [v10 TOPIC_DECISION](../tsfm_peft_ridge_correction_v10_20260929/TOPIC_DECISION.md). The original proposal and diagnostic are preserved; they are published together with the authorized v10 work, whose publication receipt governs that later status.
