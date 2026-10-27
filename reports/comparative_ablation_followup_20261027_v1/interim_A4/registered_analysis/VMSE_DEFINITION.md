# V MSE definition

V MSE is the mean across original channels of squared forecast errors over observed validation targets, measured after normalization using statistics fitted only on the A60 training interval.

[확인] 근거: research/informs_broad_validation_20261015_v1/evaluation.py:39–61; data_contract.py:61–76; fit_base.py:46–49. V는 개발 검증 구간이며 보호 테스트 T가 아니다.
