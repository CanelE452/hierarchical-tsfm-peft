# V9 prior principles and contribution boundary

This bounded review supports the staged-correction comparison in [protocol.json](protocol.json). It does not establish priority, efficacy or publication readiness. The new comparison concerns preservation of an already trained LEVEL predictor while a correction in its retained PCA space is fitted. Frozen-base adaptation, residual fitting, last-value centering and scalar shrinkage are established principles.

## Sources and actual reading depth

**Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks — Jeffrey O. Zhang, Alexander Sax, Amir Zamir, Leonidas Guibas and Jitendra Malik, ECCV 2020.** The official proceedings record and final PDF Sections 2 and 3.1–3.2 were read. The paper freezes the base, learns a side network and combines representations. Its representative alpha blending is `a B(x) + (1-a) S(x)`, with a learned mixing parameter; the described side initialization copies or distills the base. V9 instead preserves the coefficient of an already adapted parent at one, initializes the added head's output to zero, restricts one correction to P, and optionally chooses one constant strength after training. Those are the specific differences under test, not a new claim for adding a small network to a frozen one. The author project appeared in indexed search content, but direct access returned 404; no new author-code inspection was performed.

- [Official ECCV record](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1104_ECCV_2020_paper.php)
- [Final proceedings PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480698.pdf)
- [Author project](https://sidetuning.berkeley.edu/)

**Greedy Function Approximation: A Gradient Boosting Machine — Jerome H. Friedman, The Annals of Statistics 29(5), 1189–1232, 2001.** The official DOI resolved to Project Euclid, whose full text was unavailable through the attempted endpoints. The final publication's title, journal, year and pages were checked on the author-uploaded PDF cover. The readable method source was the author's **24 February 1999 preprint**, hosted by an academic course: Section 3/Algorithm 1, Section 4.1/LS Boost and Section 5/shrinkage were read. This is not recorded as a reading of the final 2001 method text. Stagewise addition to a fixed current function, squared-error residual fitting, and proportional shrinkage are direct antecedents. V9 uses one additional constrained head with the existing masked objective; its finite VAL strength selection is not the paper's general multi-stage boosting procedure or a new shrinkage principle.

- [Official publication DOI](https://doi.org/10.1214/aos/1013203451)
- [Final-paper cover accessed through the author's uploaded copy](https://www.researchgate.net/profile/Jerome-Friedman/publication/2424824_Greedy_Function_Approximation_A_Gradient_Boosting_Machine/links/6589e1ae0bb2c7472b0fc00e/Greedy-Function-Approximation-A-Gradient-Boosting-Machine.pdf)
- [Preprint method text actually read](https://www.cse.iitb.ac.in/~soumen/readings/papers/Friedman1999GreedyFuncApprox.pdf)

**AdaPTS, ICML 2025; NLinear, AAAI 2023.** The existing [v7 primary-source record](../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md) is reused. AdaPTS's latent transform → foundation model → inverse-transform and forecast-trained adapters overlap with the parent main path. The v7 review read its accessible preprint and author adapter/prediction code, not the inaccessible final proceedings PDF. NLinear's final method passage and author model, loader and training code had been read: subtracting the last input, sharing a temporal linear map, and external TRAIN standardization are prior operations. V9 does not add another P last-value restoration term: the parent already supplies its main forecast. This bounded comparison makes no claim of absence from all AdaPTS versions or related methods.

- [AdaPTS official proceedings](https://proceedings.mlr.press/v267/benechehab25a.html)
- [AdaPTS reviewed adapter code](https://raw.githubusercontent.com/abenechehab/AdaPTS/main/src/adapts/adapters.py)
- [AdaPTS reviewed prediction code](https://raw.githubusercontent.com/abenechehab/AdaPTS/main/src/adapts/adapts.py)
- [NLinear final paper](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089)
- [NLinear author model](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)

**On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation — Gavin C. Cawley and Nicola L. C. Talbot, JMLR 11, 2079–2107, 2010.** The official publication entry and abstract were rechecked; the full paper was not reread for v9. Finite-sample selection criteria can themselves be overfit. Reusing VAL for the parent, head checkpoint and alpha therefore adds selection opportunities; alpha zero does not create a TEST non-degradation guarantee. The two development datasets remain exposed.

- [Official JMLR record and abstract](https://jmlr.org/papers/v11/cawley10a.html)

## What the design can and cannot establish

Let `f*` denote the selected, already trained LEVEL parent, `P=UU^T`, `Q=I-P`, and `Delta X=X-repeat(last(X))`. The staged comparisons are

```text
STAGED_P:   f*(X) + alpha H_P(Delta X P)
STAGED_RAW: f*(X) + alpha H_RAW(Delta X)
```

The basic comparison trains with alpha one and selects its checkpoint including step zero. The auxiliary policy chooses alpha from the fixed five-value grid on one best-trained checkpoint. Both heads receive the same parent, initialization, parameter count and selection opportunities. Alpha zero is parent fallback, not evidence that a learned correction helps.

Under an orthogonal fixed P, a channel-shared bias-free temporal linear head commutes with the channel projection. Thus `H_P(Delta X P) Q=0`, and STAGED_P preserves the parent's Q output in exact arithmetic. Numerical preservation must still be checked. With complete targets and equal-weight squared error, the P and Q loss components separate; a masked or differently weighted objective need not preserve that separation. Consequently v8's shared clipping, scheduler and checkpoint selection are possible sources of coupling, not proof of unavoidable conflicting P/Q gradients.

The strongest competing explanation is ordinary side correction with extra fitting and VAL selection. STAGED_RAW is the direct matched control for the P restriction. Comparison with v8 JOINT_PQ changes initialization, the frozen learned Q function, adaptation stages and selection history, so it is a practical procedure comparison rather than pure causal isolation. Preserving Q does not guarantee better total accuracy. A useful result would support this particular staged procedure within the evaluated scope; it would not establish new normalization, generic boosting, a universal replacement, or a lower total adaptation cost from head-only timing.
