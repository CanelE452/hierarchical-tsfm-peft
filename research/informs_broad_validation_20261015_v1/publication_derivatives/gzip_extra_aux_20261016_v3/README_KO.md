# 부속 JSON/CSV 원바이트 무손실 포장

원본 내용·모든 숫자·행·조건을 바꾸지 않고 gzip(level9,mtime0)으로 보존했다. 원파일은 로컬에서 변경하지 않았다. [모든 원경로·원 SHA·gzip SHA](GZIP_MANIFEST.json), [동결 원 목록](ORIGINAL_FILE_LIST.csv), [전수 임시·독립 복원 및 반복 검증](BUILD_RESULT.json)을 함께 확인한다. 원본의 JSON 의미를 다시 직렬화하거나 CSV 숫자를 반올림하지 않았다.

저장소 루트에서 `python <이 폴더>/restore_originals.py --destination .`를 실행하면 정확한 원경로로 복원한다. 기존 byte가 같으면 skip하고 다르면 실패하며 덮어쓰지 않는다. 이 helper는 기존 e88b5cf8…ab5b의 정확한 복사본이다. 재포장 소스는 [BUILD_RECIPE.py](BUILD_RECIPE.py)이며 명시된 입력 SHA·source39·registry·lock을 확인하고 --execute --root-triggered로만 새 출력에 쓴다.

루트 canonical MD/CSV, stats pack manifest/archive, 원 복원 bootstrap은 이 부속 포장의 대상이 아니다. 공개 Markdown은 원경로 직접 링크 대신 이 manifest/index 또는 실제 gzip 경로를 사용한다. 복원 PASS는 새 학습·예측·독립 일반화·원격 게시 완료를 뜻하지 않는다.
