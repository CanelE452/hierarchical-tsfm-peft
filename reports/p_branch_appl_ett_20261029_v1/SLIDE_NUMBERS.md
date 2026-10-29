# P branch numbers for slide16

Development-V exploratory results; signed effect vs Full P.

Jena unchanged: Time branch only +0.2% (UNCLEAR); Channel branch only +8.9% (FULL_P_LOWER).

```text
Method               Dataset             Effect   Decision                    cap3_selected n
P_TIME_ONLY          appliances                -0.4% UNCLEAR                      0/10 10
P_TIME_ONLY          ETTh1                     +0.5% UNCLEAR                      0/10 10
P_TIME_ONLY          ETTh2_safe17400           -3.5% UNCLEAR                      0/10 10
P_CHANNEL_ONLY       appliances                -1.3% UNCLEAR                      0/10 10
P_CHANNEL_ONLY       ETTh1                     -0.7% UNCLEAR                      1/10 10
P_CHANNEL_ONLY       ETTh2_safe17400           -3.6% UNCLEAR                      0/10 10
```

```text
Dataset             Time_only_params Channel_only_params
appliances                      35392                3802
ETTh1                           33536                 711
ETTh2_safe17400                 33536                 711
```

Positive effect means the retained-branch variant has higher MSE. Incomplete families are not ten-seed primary results.
PROVISIONAL propagates from either comparison arm; no cap10 extension was run. Unequal capacity and joint LayerNorm/E/D coupling remain.
