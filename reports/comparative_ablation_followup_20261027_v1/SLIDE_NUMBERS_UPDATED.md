# Slide numbers updated

V is development validation; T was not evaluated. Registered median/IQR decisions are unchanged.

## Slide 16 — P ablation

- jena: OURS=0.697770, A1=0.775362, effect=11.12%, n=10/10, extended=True, undertrained_at_10=False. English verdict: P contribution verdict = CONTRIBUTES; A1 removes P.
- ETTh1: OURS=0.279643, A1=0.279226, effect=-0.15%, n=10/10, extended=False, undertrained_at_10=False. English verdict: P contribution verdict = UNCLEAR; A1 removes P.
- ETTh2_safe17400: OURS=0.277231, A1=0.263365, effect=-5.00%, n=10/10, extended=False, undertrained_at_10=False. English verdict: P contribution verdict = UNCLEAR_PROVISIONAL; A1 removes P.
- appliances: OURS=0.282019, A1=0.283612, effect=0.56%, n=10/10, extended=False, undertrained_at_10=False. English verdict: P contribution verdict = UNCLEAR; A1 removes P.

## Slide 18 — Whole D ablation and output-space floor

- jena: OURS=0.697770, A4=0.685574, effect=-1.75%, n=10/10, D1 median floor/actual=0.293 (n=10/10). English verdict: Whole-D contribution verdict = UNCLEAR; last-value and lag effects remain inseparable.
- ETTh1: OURS=0.279643, A4=0.715743, effect=155.95%, n=10/10, D1 median floor/actual=0.645 (n=10/10). English verdict: Whole-D contribution verdict = CONTRIBUTES; last-value and lag effects remain inseparable.
- ETTh2_safe17400: OURS=0.277231, A4=0.901934, effect=225.34%, n=10/10, D1 median floor/actual=0.767 (n=10/10). English verdict: Whole-D contribution verdict = CONTRIBUTES; last-value and lag effects remain inseparable.
- appliances: OURS=0.282019, A4=0.334393, effect=18.57%, n=10/10, D1 median floor/actual=0.425 (n=10/10). English verdict: Whole-D contribution verdict = CONTRIBUTES; last-value and lag effects remain inseparable.

Primary aggregate: jena, ETTh1, appliances; ETTh2_safe17400 is reported separately.
Comparison verdicts remain provisional where required C2/C3 extensions are out of scope.
Followup execution scope was selected after inspecting the first campaign results.
