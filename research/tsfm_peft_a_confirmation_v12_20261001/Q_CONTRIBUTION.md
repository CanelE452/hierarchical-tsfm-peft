# Fixed A Q contribution on existing development data

Previously exposed Robin/Jena/Hog development data; same selected v11 A checkpoints. Q removal/replacement is a fit-zero post-hoc ablation, not independent main-only training. The actual approved v12 PLAN and its new confirmation units have not been recovered; this artifact does not execute or substitute for those units.

[확인] Saved full/main predictions were reused. Only the actual paired LEVEL parent's E/D/G was evaluated on CPU; no TSFM load/forward, fitting, or benchmark was performed.

`A_MAIN_LAST = saved A main + last context residual x - D(E(x))`, broadcast over 48 steps. The learned-G contrast compares this last-only Q contribution with the parent's learned Q.

Scores use observed targets, pool per-channel SSE/count across test_a/test_b, then average channels. The combined rows below average seed losses, not predictions.

```text
dataset family                 combined_MSE combined_MAE signed_bias
robin   A_FULL                   0.27307070   0.32305936  0.03005222
robin   A_MAIN                   0.97489112   0.65284526  0.04272410
robin   A_MAIN_LAST              0.34691168   0.38274119  0.03477719
robin   base_level               0.27519585   0.32459320  0.00713893
robin   f0                       0.22909530   0.26207394  0.00245977
robin   lora_native              0.22361027   0.25634704  0.01332037
robin   full_lora_mse            0.21851705   0.25871818  0.03286625
robin   direct_nlinear           0.32772411   0.35341024  0.01939647
jena    A_FULL                   0.25309313   0.27504651  0.00617871
jena    A_MAIN                   0.30992049   0.33588618 -0.01641053
jena    A_MAIN_LAST              0.28445359   0.29220632  0.00670311
jena    base_level               0.32435916   0.32044992 -0.01215891
jena    f0                       0.29230003   0.26475763 -0.02704049
jena    lora_native              0.24058990   0.22658423 -0.02167063
jena    full_lora_mse            0.23289162   0.23894342 -0.00074745
jena    direct_nlinear           0.27398417   0.28671400  0.01126887
hog     A_FULL                   2.65855236   0.79614867 -0.05087650
hog     A_MAIN                   4.58304434   1.33134527 -0.08666210
hog     A_MAIN_LAST              3.12390918   0.89109001 -0.06528502
hog     base_level               2.66680936   0.80264902 -0.06404066
hog     f0                       2.73804422   0.63610210 -0.12236443
hog     lora_native              2.68355968   0.61890242 -0.13109481
hog     full_lora_mse            2.62530403   0.62391908 -0.13063111
hog     direct_nlinear           3.05935150   0.77547482 -0.10043757
```

Positive gain below means the left model improves the reference. Every seed and both periods remain available in JSON/CSV.

```text
dataset contrast                         combined_MSE_gain combined_MAE_gain
robin   whole_Q_gain_vs_main                    0.70182042        0.32978591
robin   Q_last_gain_vs_main                     0.62797944        0.27010407
robin   learned_G_gain_over_Q_last              0.07384098        0.05968184
jena    whole_Q_gain_vs_main                    0.05682736        0.06083967
jena    Q_last_gain_vs_main                     0.02546689        0.04367986
jena    learned_G_gain_over_Q_last              0.03136047        0.01715982
hog     whole_Q_gain_vs_main                    1.92449198        0.53519660
hog     Q_last_gain_vs_main                     1.45913516        0.44025526
hog     learned_G_gain_over_Q_last              0.46535682        0.09494134
```

Complete-vector P/Q diagnostics use exactly the same observed-vector subset for every model within each dataset/period. Their coverage is recorded; these diagnostics are separate from the masked macro score.

CPU reconstructed main+Q must match the saved GPU full prediction at fixed atol=1e-5, rtol=1e-4. Maximum discrepancies and all tolerance violations are retained in JSON; the saved GPU full prediction remains the scoring reference.

All baselines reuse existing prediction bytes and reproduce their published v11 scores. Original v11 files are preserved. This is neither an independent confirmation nor evidence that separately training main-only would produce the same model.

Numerical source: `q_contribution02.json`. Prediction arrays: `E:\CODING\proj\hierarchical-tsfm-peft\.cache\tsfm_peft_a_confirmation_v12_20261001\q_contribution02`.
