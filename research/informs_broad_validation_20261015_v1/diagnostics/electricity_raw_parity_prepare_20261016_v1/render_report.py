"""Render the saved original parity returns; reads aggregate JSON, never target arrays."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PUBLIC = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--gpu-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actual, gpu_path, output = (path.resolve() for path in (args.actual, args.gpu_receipt, args.output))
    for path in (actual, gpu_path, output):
        if not path.is_relative_to(PUBLIC / 'diagnostics'):
            raise ValueError('Use campaign diagnostics paths')
    result = read(actual / 'RESULT.json')
    if result['status'] != 'DIAGNOSTIC_COMPLETE_ORIGINAL_PARITY_RETURNS_PRESERVED':
        raise ValueError('Incomplete executions require a separate partial report')
    gpu = read(gpu_path)
    records, parities = [], {}
    for batch in (1, 8):
        parity = read(actual / ('RAW_PARITY_B%d.json' % batch))
        if (parity['atol'], parity['rtol']) != (1e-5, 1e-4):
            raise ValueError('Original tolerances changed')
        if [row['row_microbatch'] for row in parity['variants']] != ['full', 32, 8, 4]:
            raise ValueError('Original variant order changed')
        any_success = any(row['status'] in {'PASS', 'FAIL'} for row in parity['variants'])
        gate = any_success and not any(row['status'] == 'FAIL' for row in parity['variants'])
        if gate != parity['successful_variants_parity']:
            raise ValueError('Saved gate and original variant statuses disagree')
        parities[batch] = parity
        for row in parity['variants']:
            status = row['status']
            if status in {'PASS', 'FAIL'}:
                if row['F_logical_streams'] != batch * 321 or not math.isfinite(row['max_abs']) or row['max_abs'] < 0:
                    raise ValueError('Actual inference counts or delta invalid')
            records.append(dict(batch_size=batch, row_microbatch=row['row_microbatch'], status=status,
                                max_abs=row.get('max_abs', ''), F_logical_streams=row.get('F_logical_streams', ''),
                                prediction_sha256=row.get('prediction_sha256', ''),
                                reference_row_microbatch=parity['reference_row_microbatch'],
                                atol=parity['atol'], rtol=parity['rtol']))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'PARITY_VARIANTS.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 4.3))
    fig.subplots_adjust(left=.12, right=.98, top=.78, bottom=.22)
    for r, batch in enumerate((1, 8)):
        for col, row in enumerate(parities[batch]['variants']):
            reference = row['row_microbatch'] == parities[batch]['reference_row_microbatch']
            color = '#E1E5E8' if reference else {'PASS': '#C7E6DF', 'FAIL': '#F1B59D', 'BLOCKED_CUDA_OOM': '#F6E6A5'}[row['status']]
            ax.add_patch(Rectangle((col, r), .96, .88, facecolor=color, edgecolor='white', linewidth=2))
            label = 'REFERENCE' if reference else row['status'].replace('BLOCKED_CUDA_', '')
            delta = 'max |delta| = %.6g' % row['max_abs'] if 'max_abs' in row else 'no prediction returned'
            ax.text(col+.48, r+.44, label+'\n'+delta, ha='center', va='center', fontsize=10)
    ax.set(xlim=(0, 4), ylim=(1.93, -.08), xticks=[.48, 1.48, 2.48, 3.48], xticklabels=['full', '32', '8', '4'],
           yticks=[.44, 1.44], yticklabels=['Batch 1', 'Batch 8'])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Electricity Raw: original row-microbatch parity', fontsize=15, y=.96)
    fig.text(.5, .85, 'Chronos-Bolt-small FP32 | C=321, L=512, H=48 | same V past histories', ha='center', fontsize=10)
    fig.text(.12, .1, 'Original allclose: atol=1e-5, rtol=1e-4. Deltas use physical output units.\nReference compares with itself. This is an inference check; accuracy and timed cost are not measured.', fontsize=9)
    fig.savefig(output / 'PARITY_MATRIX.svg')
    fig.savefig(output / 'PARITY_MATRIX.png', dpi=160)
    plt.close(fig)

    lines = ['# Electricity Raw 비용 실패: 원 parity 반환 진단', '',
             '[확인] 기존 비용 gate 실패 기록을 보존하고 같은 원 함수를 새 audit 작업에서 다시 실행했다. 원 공차, row 후보, 모델, 정규화, V 과거 입력 구성은 유지했다. 정답 생성·채점·학습은 수행하지 않았다.', '',
             '![원 parity 진단](PARITY_MATRIX.svg)', '',
             '## 실제 관찰', '', '```text', 'batch  row_microbatch  status              max_abs']
    for row in records:
        value = repr(row['max_abs']) if row['max_abs'] != '' else 'NA'
        lines.append(f"{row['batch_size']:<6} {str(row['row_microbatch']):<14} {row['status']:<19} {value}")
    lines.extend(['```', '',
                  '배치 1과 8의 원 successful_variants_parity는 각각 ' + str(parities[1]['successful_variants_parity']) + ', ' + str(parities[8]['successful_variants_parity']) + '다. 원 반환값과 전체 정밀도 수치는 아래 JSON과 CSV에 보존했다.', '',
                  '## 판정 범위', '',
                  'PASS 또는 FAIL 상태는 해당 variant의 실제 추론이 반환됐음을 뜻한다. FAIL은 처음 성공한 reference와 원 allclose 기준이 맞지 않았다는 의미다. BLOCKED_CUDA_OOM은 별도로 구분한다. 이 진단의 정상 종료는 비용 gate 통과나 예측 성능 개선을 뜻하지 않는다.', '',
                  '배치 분할에 따른 출력 차이의 연산 단계는 이 반환값만으로 확정하지 못한다. PyTorch 2.10 공식 문서는 배치 연산과 분할 연산의 부동소수점 결과가 비트 단위로 같다고 보장하지 않는다고 설명한다. 이는 가능한 설명의 근거이며, 이번 사례의 직접 원인 증명은 아니다. [공식 수치 정확도 문서](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html#batched-computations-or-slice-computations)', '',
                  '기존 실패 8건과 COST.csv는 변경하지 않았다. warmup, timed trial, stage trial은 모두 0회다. 학습 미완료 192조건과 그 안의 GPU 3 epoch 미달 48조건도 그대로다. 현재 T 잠금은 유지한다.', '',
                  '## 자원과 재현', '',
                  f"새 audit GPU process-wall 정산은 {gpu['GPU_charge_seconds']!r}초다. 누적 {gpu['GPU_used_after_seconds']!r}/43200초, 잔여 {gpu['GPU_remaining_seconds']!r}초다. 독점 owner 아래 새 자식 작업만 실행했고 bounded wait와 종료·join 여유를 두었다.", '',
                  '입력은 기존 worker와 같은 V 구간의 과거 창이다. 원 safeFIT 자산 로더는 전체 자산을 역직렬화할 수 있으므로 T 행의 byte가 메모리에 올라오지 않았다고 주장하지 않는다. T target 생성·적합·선정·평가 호출은 없다.', '',
                  '[집계 CSV](PARITY_VARIANTS.csv), [생성 코드](../electricity_raw_parity_prepare_20261016_v1/render_report.py), [고정 실행 계획](../electricity_raw_parity_prepare_20261016_v1/PLAN.json)', '',
                  '[원 실제 진단 결과](../' + actual.name + '/RESULT.json), [배치 1 원 반환](../' + actual.name + '/RAW_PARITY_B1.json), [배치 8 원 반환](../' + actual.name + '/RAW_PARITY_B8.json)', '',
                  '[GPU audit 영수증](../electricity_raw_parity_prepare_20261016_v1/GPU_AUDIT.json), [전체 미완료와 재개 조건](../../RESUME.md)', ''])
    (output / 'REPORT_KO.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    sources = [actual / 'RESULT.json', actual / 'RAW_PARITY_B1.json', actual / 'RAW_PARITY_B8.json', gpu_path, Path(__file__)]
    receipt = dict(status='RENDERED_SAVED_ORIGINAL_PARITY_RETURNS', original_variant_rows=8,
                   source_files={path.relative_to(PUBLIC).as_posix(): sha(path) for path in sources},
                   outputs={path.name: sha(path) for path in output.iterdir() if path.is_file()},
                   new_training=False, new_target_or_accuracy_scoring=False, new_GPU_calls=False,
                   original_cost_gate_modified=False, full_campaign_completed=False)
    with (output / 'RENDER_RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
