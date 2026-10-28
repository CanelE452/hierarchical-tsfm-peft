# 발표 자료 수정본

첨부된 **Temporal–Channel Input Adaptation for Frozen Time-Series Foundation Models**의 19장 구성을 기준으로 문제정의·방법론 설명, 절제 결과, 결론과 발표 대본을 정리한다. 원본 첨부 PDF와 기존 PPTX는 수정하지 않는다.

## 전달 파일

- `Temporal_Channel_Input_Adaptation_revised.pptx`: 실결과 검증이 끝난 뒤 생성되는 최종 19장 발표 파일.
- `Temporal_Channel_Input_Adaptation_revised.pdf`: 최종 PPTX를 Microsoft PowerPoint로 내보낸 19장 PDF.
- `SPEAKER_NOTES_KO.md`: 장별 한국어 대본과 19장 질의응답. 같은 대본이 PPTX의 발표자 노트에도 들어 있다.
- `CHANGELOG_KO.md`: 장별 기존 표현과 수정 내용을 대조한 기록.
- `PRESENTATION_VALIDATION.json`: 실결과 연결, slide count, 산출물 해시, 렌더 경로, 기술 검증 결과. 최종 시각 검토는 캠페인 `FINAL_REVIEW.json`에서 별도로 확인한다.

이름에 `DRAFT`가 붙은 파일은 결과가 아직 반영되지 않은 작업용 초안이다. 최종 전달본에는 초안을 포함하지 않는다. 최종 파일은 `ANALYSIS_VALIDATION.json`이 PASS이고 기본 20회와 다섯 구성별 10 seed 결과가 검증된 경우에만 만들 수 있다.

## 내용과 해석 범위

- 3–10장: F 고정은 연구 조건, P는 압축 이전의 비선형 문맥 처리, K는 초기 학습 입력 PCA의 설명분산에 따른 휴리스틱으로 설명한다. 입력 분산과 예측 정보는 같은 개념이 아니다.
- 9장 설명분산 표는 `k_tau_rule_20261023_v1/K_TAU.csv`의 `cumulative_A_fit`이며 A_fit 0–50% 기준이다. 최종 refit의 PCA·정규화는 A60 0–60%에서 새로 적합하며 네 데이터의 K는 동일하다.
- D의 마지막 관측값 경로는 현재 수준을 C개 채널 그대로 보존한다. 모든 압축 손실을 복원한다는 주장이 아니다. 고정 lag 각각의 기여는 아직 분리해 검증하지 않았다.
- 12장 대본: A_fit 0–50% 탐색 학습 → A_stop 50–60% epoch 선택·봉인 → A60 0–60% 새 refit → V 70–80% 평가. M_cal 60–70%는 이번 선택에 사용하지 않는다.
- 16장: 기존 네 데이터의 P/D 절제와 새 Jena 분기 실험을 분리한다. 새 차트는 native PowerPoint 차트이며 막대는 중앙값, 수염은 Q1–Q3다. 정확한 선택 epoch·상한·연장·잠정 상태는 `SUMMARY.csv`와 `EXECUTION_COMPLETION.json`, 발표 검증 기록에 남긴다.
- 시간만/채널만은 유지한 P 분기를 가리킨다. joint LayerNorm과 E/D의 결합이 남으므로 모델 전체가 그 축에 독립적이라는 뜻이 아니다. 분기 용량도 다르다. A2는 파라미터 수 맥락을 제공하며 연산량·구조를 완전히 통제하지 않는다.
- UNCLEAR는 차이를 확인하지 못했다는 뜻이며 효과 없음이나 동등성의 증명이 아니다. 기술통계 판정은 유의성 검정이 아니다. 분기 간 상쇄 가능성은 검증된 결과로 주장하지 않는다.
- 18장과 대본: Raw 대비 MSE 개선과 MAE 악화, C2/C3 비교의 잠정성, K 규칙의 한계, 별도 구현·예산이 필요한 F 대조, 독립 테스트 미완료를 함께 설명한다.
- D0의 약 90%는 고정 PCA 출력 공간에 대한 조건부 하한 중 평균 이동 성분이다. 전체 예측 오차나 개선량의 90%가 아니다.
- 모두 개발 V 결과다. 기존 P 기여를 본 뒤 Jena를 선택한 탐색 분석이라는 사실을 공개한다. 보호 T 수치는 읽지 않는다.

## 원본과 편집성

원본 PDF는 64개 노트 페이지이며 실제 슬라이드는 19장이다. 각 페이지가 래스터 이미지와 숨은 OCR로 구성되어 편집 객체가 남아 있지 않았다. Downloads에서 확인한 최신 PPTX는 10월 19일 v17로 첨부 PDF와 내용·순서가 달랐으며, 연결된 Drive/Spaces에서도 일치하는 최신 편집 원본을 찾지 못했다.

따라서 첨부 PDF의 슬라이드 영역을 복원하고 다음과 같이 편집 가능한 요소를 다시 작성했다.

- **3·5·6·8·9·10·16·18장**: 텍스트, 표, 도형을 새로 작성한 편집 가능한 슬라이드. 최종 16장 차트도 편집 가능하다.
- **7·14·15·17장**: 원래 그림·표를 보존하고 목표 문구, q99와 MAPE 범위, 비용 측정 조건, K–메모리 해석을 편집 가능한 텍스트로 보완했다. 보존한 그림의 내부 글자·그림은 개별 편집 객체가 아니다.
- **1·2·4·11·12·13·19장**: 첨부 PDF의 시각 자료를 이미지로 보존했다. 내부 글자·그림은 개별 편집 객체가 아니며 이미지 교체나 재작성으로 수정해야 한다. 대본은 모든 장에서 편집 가능하다.

이는 원본 PPTX를 직접 수정한 파일이 아니라 최신 첨부 PDF를 충실히 복원한 새 발표 파일이다. 복원 이미지와 PowerPoint 재내보내기 때문에 원본과 글꼴·줄바꿈·픽셀 단위 모양이 완전히 같지는 않다. `SOURCE_RECOVERY.json`에 원본 SHA-256, 페이지 대응, crop 영역을 기록했다.

## 다시 생성하기

이 폴더의 `finalize.ps1`이 PowerPoint 내보내기, native 차트 IQR, 정상 열기, 19장 렌더 및 기술 검증을 실행한다. 현재 호스트의 Codex 번들 Node/Python/Poppler와 설치된 PowerPoint를 사용한다.

```powershell
# 새 결과 없이 문구·배치를 검토하는 초안
& ./finalize.ps1

# 분석 검증이 PASS이고 실험이 완료된 경우에만 최종본 생성
& ./finalize.ps1 -Final
```

`build_deck.cjs`는 캠페인의 `ANALYSIS_VALIDATION.json`, `EXECUTION_COMPLETION.json`, `SUMMARY.csv`, `CONTRASTS.csv`를 읽는다. 최종 산출물에 쓰인 데이터 해시는 `BUILD_FINAL.json`과 `PRESENTATION_VALIDATION.json`에 기록된다. `validate_deck.py`가 CSV와 차트 값을 대조하며, 최종 16·18장에 DRAFT/TBD/결과 대기 문구가 남아 있으면 실패한다.

PowerPoint가 생성 OOXML을 정규화하도록 새 파일에 `OpenAndRepair`를 적용한 뒤 저장하며, 사용자 원본이나 열린 파일은 변경하지 않는다. 정규화 뒤 일반 열기 성공을 `OFFICE_OPEN_FINAL.json`에 기록한다. 모든 최종 렌더를 별도로 시각 검토한 후에만 전달한다.

`assets/source_slide_*.png`와 `assets/source_branding.png`는 편집 원본이 없는 보존 장을 재생성하는 데 필요한 소스다. `SOURCE_RECOVERY.json`, `BUILD_FINAL.json`, `CHANGE_MAP.json`과 생성·내보내기·검증 코드를 함께 보관하면 수정본을 추적하기 쉽다. `prepare_source.py`는 최초 복원에 쓰인 도구이고, 이미 복원한 source assets가 있으면 다시 실행하지 않는다.

소스 이미지에서 PPTX/PDF를 다시 만드는 작업과 원본 첨부 PDF의 보존 검사는 별개다. 현재 호스트에서는 `SOURCE_RECOVERY.json`의 원본 절대 경로를 읽어 SHA-256을 확인한다. 다른 컴퓨터에서는 소스 이미지로 재생성할 수 있어도 원본 PDF가 없으면 보존 검사까지 PASS라고 기록할 수 없다. 전체 검증을 재현하려면 동일 SHA-256의 원본 PDF도 제공하고 해당 경로를 올바르게 연결해야 한다.

`renders/`, `.validation_dependencies/`, `assets/draft*`, `probe*`, `repaired.pptx`, `*DRAFT*`는 렌더·검증용 작업 파일이다. 원본 소스 이미지와 구별하며 최종 전달 ZIP에는 포함할 필요가 없다.
