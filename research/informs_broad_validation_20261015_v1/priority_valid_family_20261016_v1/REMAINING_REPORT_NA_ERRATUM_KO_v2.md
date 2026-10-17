# 남은 방법군 보고: 미측정 메모리 표기 호환 수정

[확인] 원 보고 builder의 `b2_reference`는 고정 기존 B2 비용 CSV의 `F_resident_bytes`가 `None` 또는 빈 문자열일 때만 미측정으로 받았다. 실제 [B2 참조 CSV](B2_REFERENCE_COSTS_v1.csv)의 네 행은 모두 정확한 문자열 `NA`, 범위는 `NOT_SEPARATELY_RECORDED`다. 나머지 자료·역할·장치·CPU2/RSS3GiB·V43·관측 라벨·batch/phase metadata와 원 B2 BASE의 run/family/selected epoch는 기존 계약과 일치했다. 이는 보고서의 missing-value 표기 호환 오류이며 학습·성능 실패로 해석하지 않는다.

원 builder `67d19ed46ee94192de20303aeb8480cb69641fdaf9246c507a5fccf68d12e35d`, 원 plot `4da404549722734e482375eedb416860acb686c429c0602f18d7b097a6c85fba`, 원 tests·CSV·BASE·감사·실패 영수증·과학 source48는 그대로 보존한다. 기존 B2 CSV SHA는 `e87b84a1639322d6f8cee91f6cf2dc7c048e8e0121101002ffbcf82cdde121fa`, BASE SHA는 `0100fa0551f4865cfb047716ed5ab12bea3315612f3d11126856bdb6ebed2acf`다.

새 [builder v2](build_remaining_family_report_v2.py)는 B2 참조의 해당 sentinel 검사 한 곳에 정확한 `NA`만 추가하고, 실제 자기 파일명과 입력 source 결속 두 곳을 새 이름으로 바꾼다. `na`, `N/A`, 앞 공백이 있는 ` NA`, 수치 0·문자열 0·다른 수치는 받지 않는다. 미측정 범위 검사도 유지한다. 입력 행을 고치지 않으며 정규화 MSE/MAE·비용·학습 활동·참조 SHA·출력 schema·CreateNew 정책은 그대로다.

수용한 미측정값의 내부 표현은 기존 `None`이고 CSV 출력은 기존 writer의 빈 셀이다. 이를 0이나 추정 메모리로 바꾸지 않는다. Route 전체 weights/RSS, 모델 파일 크기, F-only resident bytes는 서로 다른 범위이며 F-only 메모리는 계속 별도 미기록이다.

새 [plot v2](plot_remaining_family_v2.py)는 새 builder의 정확한 SHA와 파일명 및 자기 파일명 결속만 바꾼다. 원 canonical CSV/MD와 audit/report bindings의 정확한 대사, display 수치·artist 결속, actual selected refit 0 허용, 두 데이터셋별 좌표, 기존 참조 재사용, 출력 이름과 자원 제한은 유지한다. 첫 성공 canonical 결과 파일이 생성되기 전의 수정이므로 `REMAINING_FAMILY_*_v1` 출력 이름은 바꾸지 않는다. 실제 출력이 있으면 덮어쓰지 않고 거절한다.

## 검증 범위

실제 참조의 header와 허용 metadata를 읽고, 같은 네 batch/phase 행 구조·미측정 sentinel·selected endpoint를 가진 합성 fixture를 만들었다. 합성 정확도·비용 숫자는 새로 지어낸 값이며 실제 결과가 아니다. 수정 전 해당 fixture에서 원 `ValueError` RED가 재현됐고, 수정 후 builder 10개·plot 11개, 총 21개 stdlib 검사에서 PASS했다. 마지막 실행은 0.041초였고 NumPy/Torch/Matplotlib import는 모두 false였다.

검사는 정확 sentinel만 수용, 원 memory scope 불변, 실제 입력 행 무변경, timed와 stage 분리, 메모리 None 유지, 동등한 원 수치, 기존 출력 거부를 포함한다. 두 byte-diff 검사로 새 builder의 변경이 sentinel 한 곳+자기 결속 두 곳뿐이고, 새 plot은 builder SHA/파일명과 자기 파일명만 달라졌음을 확인한다. 원 검사 파일은 수정하지 않는다.

이 수정 단계의 실제 모델·Task·NPZ·CP·통계/trial 재채점·학습·추론·비용·프로세스·Git/network·canonical 보고 생성·그림 렌더는 0이다. 실제 감사 PASS와 원 실패 영수증은 ROOT가 보존하며, 보고와 렌더의 실제 재실행·원정밀도/links/이미지 검토는 ROOT가 별도로 수행한다. 이 문서는 두 실제 실험의 완료나 전체 MASTER 완료를 새로 주장하지 않는다.

## 실제 재실행 접점

ROOT는 원 builder CLI와 같은 인수를 새 `build_remaining_family_report_v2.py`에 전달하고 `--builder-sha`에 새 source SHA를 지정한다. plot은 `plot_remaining_family_v2.py`와 그 정확한 `--plot-sha`, 실제 성공 report binding/audit/CSV SHA를 사용한다. 기본 호출은 계속 입력 읽기·파일 생성·렌더 0이며 `--execute --root-triggered`와 실제 결속 없이는 실행하지 않는다. 이 수정은 원 감사 또는 과학 입력을 교체하는 우회가 아니다.
