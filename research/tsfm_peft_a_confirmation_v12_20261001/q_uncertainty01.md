# Conditional Q contribution intervals

Previously exposed Robin/Jena/Hog development data; same selected v11 A checkpoints. Q removal/replacement is a fit-zero post-hoc ablation, not independent main-only training. The actual approved v12 PLAN and its new confirmation units have not been recovered; this artifact does not execute or substitute for those units.

Percentile block95 conditional on fixed selected model/parent seeds and observed development periods. No fitting/selection uncertainty, no seed-population uncertainty, no multiplicity adjustment, no independent confirmation.

candidate minus reference; negative favors candidate (opposite sign to positive Q gains)

The same non-circular block resampling as v11 is used: 2000 draws, fixed seed 9262026, block 7 origins for Robin/Hog and 42 for Jena. test_a and test_b are independently resampled without crossing boundaries, and combined pools their per-channel error sums/counts. Both fixed seeds share all resamples.

```text
dataset candidate/reference         metric delta      conditional block95 (combined, mean fixed seeds)
robin   A_FULL/A_MAIN               mse  -0.70182042 [-0.95775449, -0.45774217]
robin   A_FULL/A_MAIN               mae  -0.32978591 [-0.39860492, -0.26111184]
robin   A_FULL/A_MAIN_LAST          mse  -0.07384098 [-0.10669045, -0.03503208]
robin   A_FULL/A_MAIN_LAST          mae  -0.05968184 [-0.07852819, -0.04101751]
robin   A_MAIN_LAST/A_MAIN          mse  -0.62797944 [-0.86645419, -0.40449419]
robin   A_MAIN_LAST/A_MAIN          mae  -0.27010407 [-0.32417196, -0.21210621]
jena    A_FULL/A_MAIN               mse  -0.05682736 [-0.07426585, -0.03754946]
jena    A_FULL/A_MAIN               mae  -0.06083967 [-0.07174071, -0.04769900]
jena    A_FULL/A_MAIN_LAST          mse  -0.03136047 [-0.03730654, -0.02437831]
jena    A_FULL/A_MAIN_LAST          mae  -0.01715982 [-0.01881179, -0.01330707]
jena    A_MAIN_LAST/A_MAIN          mse  -0.02546689 [-0.04303503, -0.00864042]
jena    A_MAIN_LAST/A_MAIN          mae  -0.04367986 [-0.05545362, -0.03258724]
hog     A_FULL/A_MAIN               mse  -1.92449198 [-2.41556019, -1.68470019]
hog     A_FULL/A_MAIN               mae  -0.53519660 [-0.63530217, -0.48978304]
hog     A_FULL/A_MAIN_LAST          mse  -0.46535682 [-0.74531977, -0.32298522]
hog     A_FULL/A_MAIN_LAST          mae  -0.09494134 [-0.13260801, -0.06905849]
hog     A_MAIN_LAST/A_MAIN          mse  -1.45913516 [-1.91145981, -1.17099922]
hog     A_MAIN_LAST/A_MAIN          mae  -0.44025526 [-0.51952526, -0.40555479]
```

Per-seed and per-period absolute and relative intervals are retained in JSON/CSV. An interval excluding zero describes this conditional resampling analysis only; it does not establish fresh-unit generalization or account for model/learning-rate selection.
