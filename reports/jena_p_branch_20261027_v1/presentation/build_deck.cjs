/* Reconstruct the supplied 19-slide deck; write only this presentation folder. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const pptxgen = require('pptxgenjs');
const ROOT = __dirname;
const ASSETS = path.join(ROOT, 'assets');
const source = JSON.parse(fs.readFileSync(path.join(ROOT, 'SOURCE_RECOVERY.json'), 'utf8'));
const reportRoot = path.dirname(ROOT);
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const branch = loadVerifiedResults();
const final = process.argv.includes('--final');
if (final && branch.status !== 'COMPLETE') throw new Error('Final deck requires verified COMPLETE branch results.');

function loadVerifiedResults() {
  const validationPath = path.join(reportRoot, 'ANALYSIS_VALIDATION.json');
  if (!fs.existsSync(validationPath)) return { status: 'PENDING', rows: [] };
  const v = JSON.parse(fs.readFileSync(validationPath, 'utf8'));
  const completionPath = path.join(reportRoot, 'EXECUTION_COMPLETION.json');
  const completion = JSON.parse(fs.readFileSync(completionPath, 'utf8'));
  if (v.status !== 'PASS' || v.base_complete !== 20 || v.primary_rows !== 50 ||
      completion.status !== 'TRAINING_COMPLETE' || v.T_numeric_values_parsed !== 0 ||
      sha(completionPath) !== v.inputs.completion_sha256) throw new Error('Analysis/completion validation gate is incomplete or stale.');
  const order = ['A1','P_TIME_ONLY','P_CHANNEL_ONLY','OURS','A2'];
  const label = {A1:'No P',P_TIME_ONLY:'Time only',P_CHANNEL_ONLY:'Channel only',OURS:'Full P',A2:'A2'};
  const rows = order.map(id => {
    const r = v.summary.find(s => s.method_id === id);
    if (!r || r.n !== 10 || !Number.isFinite(r.MSE_median) || !Number.isFinite(r.MSE_Q1) || !Number.isFinite(r.MSE_Q3) ||
        JSON.parse(r.selected_epochs).length !== 10) throw new Error('Missing ten-seed verified summary: '+id);
    return {id,label:label[id],chart_label:label[id]+'\ncap '+r.cap+(r.extended?' / ext':' / base')+(r.provisional?'\nPROVISIONAL':''),median_mse:r.MSE_median,q1:r.MSE_Q1,q3:r.MSE_Q3,
      trainable_parameters:r.parameters,selected_epochs:JSON.parse(r.selected_epochs),cap:r.cap,extended:r.extended,provisional:r.provisional};
  });
  const c = Object.fromEntries(v.contrasts.map(x=>[x.method_id,x]));
  const fmt = x => (x >= 0 ? '+' : '')+x.toFixed(2)+'%';
  const metricLine = 'Time only '+fmt(c.P_TIME_ONLY.effect_pct)+'; channel only '+fmt(c.P_CHANNEL_ONLY.effect_pct)+' vs full P.';
  function explain(id,removed) {
    const decision = c[id].descriptive_decision;
    const provisional = decision.includes('PROVISIONAL') ? ' 상한·예산 상태 때문에 잠정 결과입니다.' : '';
    const effect = fmt(c[id].effect_pct);
    if (decision.startsWith('FULL_P_LOWER')) return removed+' 분기를 제거하면 중앙값 MSE가 '+effect+'이고 등록된 기술통계 판정 규칙에서 Full P가 더 낮습니다.'+provisional;
    if (decision.startsWith('VARIANT_LOWER')) return removed+' 분기를 제거한 구성이 중앙값 MSE '+effect+'로 더 낮아, 두 분기를 모두 쓰는 구성이 유리하다는 주장은 지지하지 않습니다.'+provisional;
    return removed+' 분기 제거 시 중앙값 MSE 차이는 '+effect+'이며 등록된 기술통계 판정 규칙에서 차이는 불명확합니다. 동등하다는 뜻은 아닙니다.'+provisional;
  }
  const noProvisional = !rows.filter(r=>r.id.startsWith('P_')).some(r=>r.provisional);
  const decisionText = x => x.startsWith('FULL_P_LOWER') ? 'full P lower' : x.startsWith('VARIANT_LOWER') ? 'variant lower' : 'unclear';
  const conclusion = 'Jena: vs time-only, '+decisionText(c.P_TIME_ONLY.descriptive_decision)+'; vs channel-only, '+decisionText(c.P_CHANNEL_ONLY.descriptive_decision)+'.'+(noProvisional ? '' : ' Provisional.');
  const schedules = rows.map(r=>{
    const frequencies = {};
    for(const e of r.selected_epochs) frequencies[e]=(frequencies[e]||0)+1;
    return r.label+': cap '+r.cap+', 연장 '+(r.extended?'있음':'없음')+', 잠정 '+(r.provisional?'PROVISIONAL':'아님')+', 선택 epoch 분포 '+
      Object.keys(frequencies).sort((a,b)=>Number(a)-Number(b)).map(e=>e+' epoch '+frequencies[e]+'개').join(', ')+'.';
  }).join('\n');
  return {status:'COMPLETE',n:10,rows,contrasts:v.contrasts,takeaway_en:metricLine,
    takeaway_ko:explain('P_TIME_ONLY','채널')+' '+explain('P_CHANNEL_ONLY','시간'),
    conclusion_en:conclusion,conclusion_ko:explain('P_TIME_ONLY','채널')+' '+explain('P_CHANNEL_ONLY','시간'),schedule_notes_ko:schedules,
    provenance:{analysis_validation_sha256:sha(validationPath),summary_sha256:sha(path.join(reportRoot,'SUMMARY.csv')),
      contrasts_sha256:sha(path.join(reportRoot,'CONTRASTS.csv')),completion_sha256:sha(completionPath),total_new_runs:v.new_result_count}};
}
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Minjae Kim; revision prepared with Codex';
pres.subject = 'Frozen TSFM external adaptation; exploratory Jena branch follow-up';
pres.title = 'Temporal–Channel Input Adaptation for Frozen Time-Series Foundation Models';
pres.company = 'Kwangwoon University, AIaaS Lab';
pres.lang = 'en-US';
pres.theme = { headFontFace: 'Times New Roman', bodyFontFace: 'Times New Roman', lang: 'en-US' };
const W = 13.333333, H = 7.5;
const C = { wine: '740800', blue: '0057A6', pale: 'F8EEEE', ice: 'EBF2F8', ink: '202020', gray: '666666', line: 'B7B7B7', green: '3C6F51' };
const ST = pres.ShapeType;
const notes = {};
const changes = [];

function tx(s, text, x, y, w, h, opt = {}) {
  s.addText(text, { x, y, w, h, fontFace: 'Times New Roman', fontSize: 18, color: C.ink,
    margin: 0, breakLine: false, valign: 'mid', isTextBox: true, ...opt });
}
function rect(s, x, y, w, h, fill = 'FFFFFF', stroke = C.line, radius = true) {
  s.addShape(radius ? ST.roundRect : ST.rect, { x, y, w, h, radius: 0.08, fill: { color: fill },
    line: { color: stroke, width: 1 }, rectRadius: 0.08 });
}
function arrow(s, x1, y1, x2, y2, color = C.gray) {
  s.addShape(ST.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color, width: 1.3, endArrowType: 'triangle' } });
}
function header(s, section, number, title, references = '') {
  s.background = { color: 'FFFFFF' };
  tx(s, section, 0.4, 0.12, W - 0.8, 0.64, { fontSize: 36, bold: true, align: 'center' });
  // This red rule belongs to the supplied AIaaS template and is deliberately retained.
  s.addShape(ST.line, { x: 0, y: 0.97, w: W, h: 0, line: { color: C.wine, width: 2.5 } });
  tx(s, '▪', 0.6, 1.35, 0.22, 0.3, { fontSize: 23 });
  tx(s, title, 0.98, 1.29, 11.7, 0.42, { fontSize: 23, bold: true });
  s.addImage({ path: path.join(ASSETS, 'source_branding.png'), x: 0.08, y: 7.06, w: 1.72, h: 0.34 });
  if (references) tx(s, references, 2.4, 7.03, 9.95, 0.34, { fontSize: 8.4, align: 'right', color: C.gray });
  s.addShape(ST.rect, { x: 12.48, y: 7.06, w: 0.85, h: 0.44, fill: { color: C.wine }, line: { color: C.wine } });
  tx(s, String(number), 12.48, 7.12, 0.85, 0.25, { fontSize: 11, color: 'FFFFFF', align: 'center' });
  if (!final) tx(s, branch.status==='COMPLETE' ? 'DRAFT — VERIFIED DATA; FINAL REVIEW PENDING' : 'DRAFT — NEW BRANCH RESULTS PENDING', 8.3, 6.86, 4.2, 0.14,
    { fontSize: 8, color: C.wine, align: 'right' });
}
function bullet(s, text, y, opt={}) {
  tx(s, '▪', 0.72, y, 0.2, 0.32, { fontSize: 19 });
  tx(s, text, 1.03, y, 11.4, opt.h || 0.5, { fontSize: 18, ...opt });
}
function labelBox(s, text, x, y, w, h, opt={}) {
  rect(s, x, y, w, h, opt.fill || 'FFFFFF', opt.stroke || C.line);
  tx(s, text, x+0.12, y+0.1, w-0.24, h-0.2, { align: 'center', fontSize: 17, ...opt });
}
function note(number, text) { notes[number] = text; }
function record(number, before, after) { changes.push({slide:number,before,after}); }
function existingNotes(number) {
  const text=source.slides[number].original_notes;
  return text.split(/\[제작자 근거|·범위 — 대본 아님\]/)[0].replace(/\n/g,' ').replace(/\s{2,}/g,' ').trim()
    .replace('발 표하겠습니다','발표하겠습니다').replace('이 연구 에서는','이 연구에서는');
}

for (let number=1; number<=19; number++) {
  const s=pres.addSlide();
  const img=path.join(ASSETS, source.slides[number].image);
  if (![3,5,6,8,9,10,16,18].includes(number)) s.addImage({path:img,x:0,y:0,w:W,h:H});
  note(number, existingNotes(number));
  if(number===3){
    header(s,'Background',number,'Problem 1: Reusing a Frozen Univariate Forecaster on Multivariate Data',
      'Ansari et al., Chronos, TMLR (2024).  Benechehab et al., AdaPTS, ICML (2025).');
    bullet(s,'Setting: reuse the pretrained forecaster without updating its weights.',1.96);
    bullet(s,'Raw applies the same forecaster to each channel separately; it has no cross-channel input.',2.63);
    bullet(s,'Goal: adapt to the target data while balancing forecast error and inference cost.',3.30);
    labelBox(s,'Multivariate history\nC observed channels',0.96,4.25,3.4,1.38,{fill:'F4F4F4',fontSize:20});
    labelBox(s,'Raw: shared, frozen F\nApplied separately to each channel',4.95,4.25,3.4,1.38,{fill:C.ice,stroke:C.blue,fontSize:20});
    labelBox(s,'Raw forecasts\nOne forecast per original channel',8.94,4.25,3.4,1.38,{fill:C.pale,stroke:C.wine,fontSize:20});
    arrow(s,4.4,4.94,4.88,4.94); arrow(s,8.42,4.94,8.86,4.94);
    tx(s,'Cross-channel dependence is a motivation to test—not a guarantee of additional predictive information.',1.04,6.08,11.2,0.5,{fontSize:18,color:C.wine,align:'center'});
    note(number,'이 연구는 기존 단변량 시계열 파운데이션 모델의 가중치를 그대로 유지하며 재사용하는 조건을 다룹니다. Raw는 같은 모델에 채널을 하나씩 넣으므로 다른 채널의 관측값을 직접 입력으로 사용하지 않습니다. 대상 데이터의 여러 채널이 관련될 수 있다는 점은 외부 적응을 시도할 동기입니다. 그러나 채널이 함께 움직인다는 사실만으로 예측이 반드시 좋아진다고 보장할 수는 없습니다. 우리의 목표는 이 조건에서 예측 오차와 추론 비용의 균형을 개선하는 것입니다. 일반적인 모든 도메인 적응 문제를 해결한다는 주장은 아닙니다.');
    record(number,'도메인 차이와 채널 관계에서 외부 적응의 필요성을 넓게 결론냄','F 고정은 연구 조건, 오차·비용 균형은 목표로 분리. 채널 관계는 예측 이득의 보장이 아님을 명시.');
  }
  if(number===4){
    note(number,'모델을 대상 데이터에 적응시키는 방법으로 전체 파인튜닝, 일부 층만 학습하는 부분 파인튜닝, LoRA나 내부 어댑터 같은 파라미터 효율적 적응이 있습니다. 이 장은 가능한 대안을 설명합니다. 내부 튜닝이 불가능하거나 항상 더 나쁘다는 뜻은 아닙니다. 이번 연구는 기존 F의 가중치를 유지하며 외부 모듈로 적응한다는 조건을 선택했습니다. F를 고정해도 앞단 P와 E를 학습하려면 F를 통과하는 입력 기울기가 필요하므로, 전체 학습 비용이 자동으로 작아진다고 말하지 않습니다.');
    record(number,'내부 튜닝의 소개','기존 시각 자료를 보존하고 대본에서 외부 적응은 연구 설정이며 내부 튜닝의 보편적 열등성을 뜻하지 않음을 명시.');
  }
  if(number===5){
    header(s,'Background',number,'External Adaptation: Learn the Interface, Keep the Forecaster',
      'Benechehab et al., AdaPTS: Adapting Univariate Foundation Models, ICML (2025).');
    bullet(s,'An input adapter mixes the original channels into K latent series.',1.96);
    bullet(s,'The same frozen F predicts each latent series; an output adapter returns C channels.',2.63);
    bullet(s,'K < C reduces the number of series sent through F; total speed is measured end to end.',3.30);
    labelBox(s,'C original\nchannels',0.8,4.4,2.05,1.28,{fontSize:21});
    labelBox(s,'Input adapter\nC → K',3.25,4.4,2.05,1.28,{fill:C.pale,stroke:C.wine,fontSize:21});
    labelBox(s,'Frozen F\nK series',5.70,4.4,2.05,1.28,{fill:C.ice,stroke:C.blue,fontSize:21});
    labelBox(s,'Output adapter\nK → C',8.15,4.4,2.05,1.28,{fill:C.pale,stroke:C.wine,fontSize:21});
    labelBox(s,'C-channel\nforecast',10.60,4.4,2.05,1.28,{fontSize:21});
    for(const x of [2.88,5.33,7.78,10.23]) arrow(s,x,5.04,x+0.29,5.04);
    tx(s,'Frozen weights preserve the checkpoint; they do not make adapter training free.',1.05,6.11,11.25,0.4,{fontSize:19,align:'center',color:C.wine});
    note(number,'외부 적응은 F의 가중치를 바꾸지 않고 입력과 출력의 인터페이스를 학습하는 방식입니다. 입력 어댑터가 C개 채널을 K개 잠재 시계열로 바꾸고, 같은 고정 예측기가 각 잠재 시계열을 예측합니다. 출력 어댑터는 예측을 원래 채널로 돌립니다. K가 C보다 작으면 F에 보내는 시계열 수는 줄지만 어댑터 연산도 있으므로 실제 속도는 전체 경로를 측정해야 합니다. 학습 때에는 F를 통과하는 입력 기울기가 필요합니다. 용어는 외부 파라미터 효율적 적응으로 통일하며, F 내부에 모듈을 삽입하는 PEFT와 구조적 위치가 다릅니다.');
    record(number,'채널 혼합과 K<C가 관계 활용 및 비용 감소를 바로 보장하는 인상','F 가중치 보존·입출력 인터페이스 학습을 설명하고 F 시계열 수와 전체 실측 비용을 분리.');
  }
  if(number===6){
    header(s,'Problem',number,'Problem 2: Limited Latent Capacity Must Carry Useful Forecast Changes',
      'Eckart and Young, Psychometrika (1936): reconstruction result only.  Benechehab et al., ICML (2025).');
    bullet(s,'A small K limits the representation available to the latent forecasting path.',1.96);
    bullet(s,'Input reconstruction variance is not the same as information needed to predict the future.',2.63);
    bullet(s,'Our design preserves the last observed level separately and learns forecast changes.',3.30);
    labelBox(s,'Input X\nC channels',0.95,4.32,2.2,1.1,{fontSize:21});
    labelBox(s,'Latent forecast path\nP → E → F → D\nK series through F',4.05,4.1,5.1,1.55,{fill:C.pale,stroke:C.wine,fontSize:21});
    labelBox(s,'Forecast\nlast value + change',10.06,4.32,2.4,1.1,{fontSize:20});
    arrow(s,3.22,4.87,3.98,4.87); arrow(s,9.23,4.87,9.99,4.87);
    s.addShape(ST.line,{x:2.05,y:5.45,w:0,h:0.70,line:{color:C.blue,width:1.4}});
    s.addShape(ST.line,{x:2.05,y:6.15,w:9.2,h:0,line:{color:C.blue,width:1.4}});
    s.addShape(ST.line,{x:11.25,y:5.48,w:0,h:0.67,line:{color:C.blue,width:1.3,beginArrowType:'triangle'}});
    tx(s,'Last observation: a separate C-channel path',3.70,6.26,6.2,0.28,{fontSize:17,color:C.blue,align:'center'});
    note(number,'차원을 줄이면 잠재 예측 경로가 사용할 표현이 제한됩니다. 다만 입력의 분산을 잃는다는 사실만으로 예측에 필요한 정보를 그만큼 잃는다고 단정할 수는 없습니다. PCA의 재구성 성질은 미래 예측 오차의 하한이 아닙니다. 우리의 구조에서는 마지막 관측값이 원래 C개 채널 그대로 출력에 전달됩니다. 따라서 모든 정보가 K개 시계열을 거쳐야 하는 엄격한 병목은 아닙니다. 현재 수준은 별도 경로에 두고, 잠재 경로가 미래 변화를 표현하도록 역할을 나누려는 설계입니다. 버린 입력 정보를 모두 복구한다는 보장은 없습니다.');
    record(number,'입력 분산 손실에서 예측 병목을 직접 도출; 모든 정보가 K를 통과','표현 제한은 구조적 사실, 예측 정보 손실은 가설로 분리. 마지막 값 우회와 변화량 경로를 명시.');
  }
  if(number===7){
    // Preserve the detailed source diagram; replace only the claim-bearing goal strip.
    rect(s,0.70,6.19,11.9,0.58,'FFFFFF','FFFFFF',false);
    tx(s,'Goal: reuse frozen F with fewer latent series and a useful accuracy–cost trade-off',0.82,6.22,11.65,0.24,{fontSize:17,bold:true,color:C.wine,align:'center'});
    tx(s,'P adjusts context before projection; the last observed level bypasses the latent forecast path.',0.82,6.49,11.65,0.22,{fontSize:16,align:'center'});
    note(number,'전체 구조는 P, E, 고정 예측기 F, 디코더 D로 이루어집니다. P는 압축하기 전에 과거 구간과 채널을 이용해 표현을 조정합니다. E는 C개 채널을 K개 잠재 시계열로 투영합니다. F는 각 잠재 시계열을 같은 가중치로 예측하고, D는 잠재 과거와 예측에서 얻은 변화 특징을 원래 채널의 변화량으로 변환합니다. 마지막 관측값은 별도 경로로 그대로 더합니다. 목표는 고정 F의 재사용 조건에서 정확도와 비용의 균형을 얻는 것이며, 모든 데이터나 지표에서 개선을 보장하는 구조는 아닙니다.');
    record(number,'더 정확하고 더 저렴하다는 목표 문구','정확도·비용 균형이라는 목표로 범위를 조정. 상세 다이어그램은 원본 이미지, 수정 문구는 편집 가능.');
  }
  if(number===8){
    header(s,'Proposed Method',number,'Context Module P: Adjust the Representation Before Compression',
      'Chen et al., TSMixer, TMLR (2023).  Current implementation: temporal + channel residual MLPs.');
    bullet(s,'F reads an already compressed history; P can use the full observed window before E.',1.96);
    bullet(s,'E is a shared linear projection; P adds nonlinear, input-dependent context processing.',2.63);
    bullet(s,'This motivates the design; it does not prove that both branches are necessary.',3.30);
    labelBox(s,'X\nL × C',0.80,4.52,1.3,1.12,{fontSize:21});
    labelBox(s,'Time residual branch\nL → 32 → L\nshared across channels',2.60,4.16,3.05,1.78,{fill:C.pale,stroke:C.wine,fontSize:20});
    labelBox(s,'Channel residual branch\nC → 32 → C\nshared across time steps',6.15,4.16,3.05,1.78,{fill:C.pale,stroke:C.wine,fontSize:20});
    labelBox(s,'E\nC → K',9.7,4.52,1.45,1.12,{fontSize:21});
    labelBox(s,'Z\nL × K',11.65,4.52,1.0,1.12,{fontSize:21});
    for(const x of [2.13,5.68,9.23,11.18]) arrow(s,x,5.08,x+0.38,5.08);
    tx(s,'Residual outputs start at zero: P(X) = X initially. Joint LayerNorm still couples time and channels.',0.95,6.31,11.6,0.36,{fontSize:17,color:C.gray,align:'center'});
    note(number,'P의 목적은 과거를 처음 사용하거나 채널을 처음 섞는 것이 아닙니다. F는 이미 압축된 과거를 읽고, E와 D도 채널을 혼합합니다. P는 압축하기 전 전체 관측 구간을 보고 비선형적으로 표현을 조정할 수 있게 합니다. 제한된 K에 예측에 유용한 내용을 담도록 하려는 설계 가설입니다. 시간 분기는 채널마다 같은 MLP로 과거 구간을 처리하고, 채널 분기는 각 시점에서 같은 MLP로 채널을 처리합니다. 두 분기의 LayerNorm은 전체 L×C 통계를 쓰므로 한 분기를 제거해도 순수한 축별 독립성이 되는 것은 아닙니다. 출력층을 0으로 초기화해서 시작할 때 P는 identity입니다.');
    record(number,'P가 문맥을 더한다는 설명','F·E 대비 추가 역할을 압축 이전의 비선형 문맥 처리로 구체화. 두 분기 필연성·축별 독립성 주장을 피함.');
  }
  if(number===9){
    header(s,'Proposed Method',number,'Projection E: A Reproducible Rule for Initial Capacity K',
      'Source: k_tau_rule_20261023_v1/K_TAU.csv, τ = 0.9, cumulative_A_fit.  Rea and Rea (2016): heuristic context.');
    bullet(s,'Choose the smallest K retaining at least τ = 0.9 of the A_fit input PCA variance.',1.96);
    bullet(s,'Initialize E from PCA, then update E with the final forecast loss.',2.63);
    bullet(s,'This is an initial-capacity heuristic—not a guarantee of optimal predictive dimension.',3.30);
    const rows=[['Dataset','C','K at τ = 0.9',{text:'A_fit PCA variance\nretained',options:{fontSize:14,bold:true}}],['Jena','21','6','91.1%'],['Appliances','26','10','90.7%'],['ETTh1','7','4','93.0%'],['ETTh2','7','4','93.6%']];
    s.addTable(rows,{x:1.20,y:4.00,w:10.9,colW:[4.3,1.6,2.5,2.5],border:{type:'solid',pt:0.5,color:'C5C5C5'},fontFace:'Times New Roman',fontSize:18,
      margin:0.06,color:C.ink,rowH:0.36,fill:'FFFFFF',bold:false,autoPage:false,
      boldFirstRow:true});
    tx(s,'K uses A_fit (0–50%) PCA variance. Refit PCA/normalization use A60 (0–60%); K is unchanged in all four datasets.',0.98,6.31,11.5,0.52,{fontSize:16,color:C.wine,align:'center'});
    note(number,'K는 A_fit, 즉 시간순 0–50% 구간 입력의 PCA 누적 설명분산이 0.9 이상이 되는 최소 차원으로 정합니다. 표의 91.1%, 90.7%, 93.0%, 93.6%는 K 선택에 사용한 A_fit 누적분산이며 K_TAU.csv의 cumulative_A_fit에서 왔습니다. 최종 refit 때 PCA와 정규화는 A60, 즉 0–60%에서 새로 적합하지만 네 데이터의 K는 동일합니다. 같은 기준으로 데이터별 채널 중복을 반영하고, 검증 성능을 보고 K를 고르는 과정을 피하려는 재현 가능한 초기 용량 규칙입니다. E는 PCA 방향에서 시작하지만 이후 예측 오차로 학습됩니다. P도 표현을 바꾸고 D는 변화량을 예측하므로, 원래 입력의 90% 분산이 학습된 표현이나 예측 정보의 90%라는 뜻은 아닙니다. 0.9는 사전에 고정한 경험적 설정이며 최적 예측 차원을 이론적으로 보장하지 않습니다. 실제 K 민감도와 비용은 17장에서 제한된 범위로 확인합니다.');
    record(number,'설명분산 규칙의 절차 중심','초기 입력 PCA 기준·재현 가능한 휴리스틱임을 명시. 학습 후 표현·변화량·예측 정보 90% 보장과 구분.');
  }
  if(number===10){
    header(s,'Proposed Method',number,'Decoder D: Preserve the Current Level and Predict a Change',
      'Zeng et al., LTSF-Linear / NLinear, AAAI (2023): last-value residual formulation.  Registered D implementation.');
    bullet(s,'The last observed value reaches the output directly in all C original channels.',1.96);
    bullet(s,'D combines latent change features at several past reference positions.',2.63);
    bullet(s,'Lag positions {0, 1, 2, 4, 8, 16, 32} are a fixed design choice; their added value is not yet isolated.',3.30,{h:0.58});
    labelBox(s,'Latent history + F forecast\nchange features at fixed lag positions',0.92,4.36,4.75,1.22,{fill:C.ice,stroke:C.blue,fontSize:21});
    labelBox(s,'D: linear map\n7K → C',6.15,4.36,2.5,1.22,{fill:C.pale,stroke:C.wine,fontSize:21});
    labelBox(s,'Channel-wise change\nΔŶ',9.13,4.36,3.2,1.22,{fontSize:21});
    arrow(s,5.72,4.97,6.08,4.97); arrow(s,8.7,4.97,9.06,4.97);
    tx(s,'Forecast = last observed value + predicted change',1.1,5.95,11.15,0.45,{fontSize:25,bold:true,align:'center',color:C.wine});
    tx(s,'Illustration: 15.0 °C + 0.8 °C = 15.8 °C',2.0,6.48,9.2,0.28,{fontSize:19,align:'center'});
    note(number,'D에서는 각 채널의 마지막 관측값을 출력에 직접 더합니다. 현재 수준은 압축을 거치지 않고 보존하고, 나머지 경로가 앞으로의 변화량을 표현하도록 역할을 나눕니다. 잠재 과거와 F의 예측을 시간축으로 연결해 고정된 lag 위치의 변화 특징을 만들고, 선형 층으로 원래 채널의 변화량으로 변환합니다. lag 집합은 현재 구현의 고정 설계값이며, 특정 실제 주기나 인과 시차를 찾아냈다는 뜻이 아닙니다. 여러 위치의 변화 특징을 조합할 수 있지만 그것이 반드시 도움이 되는지는 별도 질문입니다. 기존 A4는 D 전체를 바꾼 실험이므로 마지막 값·변화량·lag 각각의 효과로 나눠 해석하지 않습니다. 마지막 값 경로가 모든 압축 손실을 복구하거나 모든 분포 이동에 강건함을 보장하지도 않습니다.');
    record(number,'계산 방식 중심, 여러 lag의 이유 불명확','현재 수준 보존의 역할을 명확히 하고 lag는 고정 설계값·개별 기여 미확인으로 표시.');
  }
  if(number===11){
    note(number,'학습 손실은 원래 채널에서 최종 예측의 평균제곱오차 MSE 하나입니다. P, E, D를 함께 학습하고 F의 가중치는 바꾸지 않습니다. 오차의 기울기는 F를 거쳐 E와 P까지 전달됩니다. 따라서 잠재 시계열은 입력 재구성을 최대화하도록 학습되는 것이 아니라, 이 예측기를 거친 최종 예측에 맞춰 학습됩니다. 다음 실험에서는 Raw와 비교 어댑터, P와 D의 절제, K 민감도를 차례로 보겠습니다.');
  }
  if(number===12){
    note(number,'실험은 비교, 절제, K 민감도의 세 부분입니다. F는 고정된 Chronos-Bolt-small입니다. 입력은 512시점이고 예측 길이는 48시점입니다. 데이터는 Jena 21채널, Appliances 26채널, ETTh1과 ETTh2 각 7채널입니다. 시간순 첫 0–50%의 A_fit으로 탐색 학습하고 50–60%의 A_stop에서 epoch를 선택해 봉인합니다. 그 뒤 선택한 길이로 0–60%의 A60에서 새로 refit하고 70–80%의 V에서 평가합니다. 60–70%의 M_cal은 이번 학습 길이 선택에 사용하지 않습니다. 보호 T는 열지 않습니다. K와 표준화 통계는 등록된 학습 구간에서 정합니다. 주지표는 표준화된 채널별 V MSE의 평균이며, MAE와 다른 지표도 함께 봅니다. 학습 방법은 사전에 정한 10개 seed를 사용합니다. 비교 방법마다 실제 상한과 선택 epoch는 다르며, 결과 보고서에 공개합니다. V를 이미 반복해서 확인했으므로 이번 Jena 분기는 개발 V에서의 탐색 결과입니다.');
    record(number,'실험 설정 대본','A_fit 0–50% 탐색 → A_stop 50–60% epoch 선택·봉인 → A60 0–60% 새 refit → V 70–80% 평가를 명확히 함. M_cal은 이번 선택에 쓰지 않음.');
  }
  if(number===13){
    note(number,'먼저 비교 실험입니다. 표는 기존 검증된 10개 seed의 V MSE 중앙값이며, 학습하지 않는 Raw와 PCA는 단일 값입니다. 우리 방법은 Raw보다 Jena에서 17.8%, Appliances에서 18.9%, ETTh1에서 1.3% 낮고 ETTh2에서는 6.8% 높았습니다. AdaPTS식 PCA, LinearAE, dropoutLinearAE는 같은 F와 분할에 이식한 비교 방법으로, 원 논문 성능의 재현이라는 뜻은 아닙니다. 우리 방법의 중앙값이 이 어댑터들보다 낮지만 Jena는 등록된 중앙값·IQR 규칙에서 차이가 불명확합니다. ETTh1과 Appliances의 우위도 비교 어댑터가 epoch 상한을 자주 선택했기 때문에 잠정적입니다. 이번 Jena P 분기 실험은 C2·C3의 학습 연장을 대신하지 않습니다. ETT에서 선형 출력 공간과 수준 변화가 관련될 수 있다는 진단은 있지만, 이것만으로 모든 성능 차이의 원인을 확정하지 않습니다.');
    record(number,'비교 결과와 제작 메모가 섞인 대본','비교 중앙값은 보존하고 C2/C3 잠정성, Jena 차이 불명확, 이식 비교 범위를 명확히 함. 제작 메모 제거.');
  }
  if(number===14){
    s.addShape(ST.rect,{x:1.27,y:2.54,w:11.43,h:0.36,fill:{color:'FFFFFF'},line:{color:'FFFFFF'}});
    tx(s,'Error quantiles: q90/q95/q99 lower on Jena/Appliances; on ETT, only q99 is lower among those shown.',1.32,2.55,11.35,0.31,{fontSize:16});
    s.addShape(ST.rect,{x:6.99,y:5.65,w:5.67,h:0.52,fill:{color:'FFFFFF'},line:{color:'FFFFFF'}});
    tx(s,'R²: ours / Raw; other rows: % change.\nMAPE in original units: |y| ≥ 0.1 × σ_A60,c.',7.01,5.68,5.64,0.47,{fontSize:14,color:'404040'});
    note(number,'MSE만으로 성능을 평가하면 놓치는 부분이 있습니다. MSE는 큰 오차를 더 강하게 반영합니다. 우리 방법의 MSE는 세 데이터에서 Raw보다 낮지만 MAE와 MASE는 네 데이터 모두 더 높았습니다. Jena와 Appliances에서는 절대오차의 상위 분위수가 줄고 중앙값은 커졌습니다. 따라서 흔한 작은 오차와 드문 큰 오차 사이의 교환이 관찰됩니다. 이 분위수 분석은 전체 오차 분포의 요약이며, 동일한 시점의 오차가 각각 개선됐다는 짝지어진 인과 설명은 아닙니다. ETT는 표시한 분위수 중 q99만 낮았습니다. 상위 1%에 속하는 모든 오차가 줄었다는 뜻은 아닙니다. MAPE는 원 단위의 |y|가 해당 채널 A60 표준편차의 0.1배 이상인 값만 사용한 조건부 지표입니다. 결론을 모든 시점이나 지표의 성능 개선으로 확대하지 않겠습니다.');
    record(number,'오차 분위수 설명의 top 1% 표현과 불명확한 MAPE 제외 조건','시각 문구를 표시한 분위수 중 q99만 감소로 고치고 MAPE |y|≥0.1σ_A60,c 조건을 표시. 분포 요약을 시점별 개선으로 확대하지 않음.');
  }
  if(number===15){
    tx(s,'Inference cost: batch 8; median latency over 20 repeats; OURS = mean of two seeds.\nTable: default ≤128 F rows/call. Jena 3.6×: separate fixed-48-row setting. Memory: peak allocated GPU memory.',1.03,6.02,11.36,0.66,{fontSize:14.5,color:'404040'});
    note(number,'이 표는 기존에 측정한 추론 비용입니다. batch 8에서 지연 시간을 20회 측정한 중앙값을 사용하고 우리 방법은 두 seed의 값을 평균했습니다. 메모리는 GPU peak allocated memory입니다. 표는 F 호출당 최대 128행의 기본 정책에서 측정했고, Jena 메모리는 Raw보다 약 26%, ETT는 약 10% 적으며 속도는 약 1.3–2.1배였습니다. Jena의 약 3.6배는 F 호출을 고정 48행으로 맞춘 별도 조건이므로 두 측정 조건을 섞지 않습니다. Appliances 비용은 아직 측정하지 않았습니다. 학습 파라미터는 약 3만 4천–3만 7천 개이며 대부분 P에 있고, AdaPTS식 선형 어댑터는 더 작습니다. 다른 기존 비교에서 LoRA, Ridge 또는 Direct MLP가 더 정확한 경우도 있었으나 학습·측정 조건이 완전히 같지는 않습니다. 이번 실험만으로 F가 항상 필요한지, 내부 튜닝보다 항상 우수한지, 전체 학습 비용이 더 작은지는 결론내리지 않습니다.');
    record(number,'기존 비용 결과 대본','시각 각주와 대본에 추론 peak allocated memory, batch 8·20회 중앙값·두 seed 평균, 기본128행과 별도고정48행 조건을 표시.');
  }
  if(number===16){
    header(s,'Experiments',number,'Ablation: Component Roles and the Jena P-Branch Follow-up',
      'Sources: comparative_ablation_followup_20261027_v1; jena_p_branch_20261027_v1.  V = development validation.');
    tx(s,'Registered ablations: change in median V MSE',0.75,1.98,5.55,0.60,{fontSize:20,bold:true});
    tx(s,'10 seeds; positive = worse after removal/replacement',0.75,2.59,5.55,0.25,{fontSize:14,color:C.gray});
    const rows=[['Dataset','No P','Linear D'],['Jena','+11.12%','−1.75%'],['Appliances','+0.56%','+18.57%'],['ETTh1','−0.15%','+155.95%'],['ETTh2*','−5.00%*','+225.34%']];
    s.addTable(rows,{x:0.75,y:2.97,w:5.52,colW:[2.25,1.6,1.67],rowH:0.40,fontFace:'Times New Roman',fontSize:17,
      margin:0.06,border:{type:'solid',pt:0.5,color:'C5C5C5'},fill:'FFFFFF',autoPage:false,boldFirstRow:true});
    tx(s,'P contributes on Jena; D contributes on Appliances and ETTh1 under the registered rule.',0.8,5.46,5.35,0.70,{fontSize:18,bold:true,color:C.wine});
    tx(s,'* ETTh2 is auxiliary; its No-P result remains provisional.\nA2 (parameter-count reference): UNCLEAR on all four datasets.',0.8,6.27,5.35,0.50,{fontSize:13,color:C.gray});
    tx(s,'Jena: exploratory P-branch comparison',6.77,1.98,5.75,0.36,{fontSize:19,bold:true});
    tx(s,'10 seeds; median V MSE, whiskers Q1–Q3; lower is better',6.77,2.37,5.75,0.25,{fontSize:13.5,color:C.gray});
    if(branch.status==='COMPLETE'){
      const rs=branch.rows;
      s.addChart(pres.ChartType.bar,[{name:'Median V MSE',labels:rs.map(r=>r.chart_label),values:rs.map(r=>r.median_mse)}],
        {x:6.74,y:2.78,w:5.82,h:3.10,barDir:'col',showLegend:false,showTitle:false,showValue:true,
         catAxisLabelFontFace:'Times New Roman',catAxisLabelFontSize:10,valAxisLabelFontFace:'Times New Roman',valAxisLabelFontSize:11,
         valAxisMinVal:0,valAxisMaxVal:Math.ceil(Math.max(...rs.map(r=>r.q3))*10)/10+0.1,
         valAxisTitle:'V MSE ↓',showValAxisTitle:true,valAxisTitleFontSize:13,
         chartColors:[C.wine],showBorder:false,showCatName:false,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.000',dataLabelColor:C.ink,dataLabelFontSize:11,
         valGridLine:{color:'DDDDDD',width:0.5},catAxisLineColor:'999999',showCatName:false});
      tx(s,branch.takeaway_en || 'Exploratory branch effects are conditional on this architecture and training rule.',6.78,6.10,5.68,0.52,{fontSize:16,color:C.wine,bold:true});
    } else {
      rect(s,6.8,2.58,5.65,3.30,'F6F6F6','CCCCCC');
      tx(s,'NEW EXPERIMENTS IN PROGRESS\nTime-only P and Channel-only P\n10 seeds each; results will replace this panel',7.08,3.43,5.1,1.4,{fontSize:22,align:'center',color:C.gray});
    }
    tx(s,'Unequal branch capacity: time MLP 33,312; channel MLP 1,397. Branch effects ≠ intrinsic superiority of an axis.',6.78,6.65,5.68,0.28,{fontSize:10.5,color:C.gray});
    note(number,'기존 절제 실험에서 Jena의 P 제거는 10 epoch까지 연장 탐색한 뒤 MSE가 11.12% 증가해 등록 기준상 기여가 확인됐습니다. D 전체를 선형 디코더로 바꾸면 Appliances에서 18.57%, ETTh1에서 155.95%, 보조 ETTh2에서 225.34% 증가했습니다. Jena에서 D 대체의 차이는 불명확했습니다. A2는 네 데이터 모두 불명확합니다. 오른쪽 Jena 분기 실험은 기존 P 기여가 관찰된 데이터를 선택한 탐색 분석입니다. '+(branch.status==='COMPLETE' ? branch.takeaway_ko : '시간 분기만과 채널 분기만의 20회 새 결과는 아직 실행 중이므로 초안에는 수치를 넣지 않았습니다.')+' 시간 MLP의 파라미터는 33,312개, 채널 MLP는 1,397개이므로 시간만과 채널만의 순위가 축 자체의 우월성을 뜻하지 않습니다. A2는 파라미터 수의 맥락을 제공하지만 연산량과 구조까지 완전히 통제한 대조군은 아닙니다. 실제 선택 epoch와 연장 여부, 미해결 상한을 결과와 함께 공개합니다.');
    if(branch.status==='COMPLETE') notes[number]+='\n\n[선택 일정 참고: 발표 중 필요한 경우 설명]\n'+branch.schedule_notes_ko;
    record(number,'Jena No-P 및 D 결과 TBD; Figure pending','완료된 A1/A4 결과로 교체하고 Jena 새 두 분기를 기존 OURS/No-P/A2와 비교. 새 결과는 검증 완료 후에만 확정본 반영.');
  }
  if(number===18){
    header(s,'Conclusion',number,'What the Evidence Supports—and What Remains Open',
      'All accuracy claims here are development-V results. Comparator extensions and independent test remain open.');
    labelBox(s,'Design\nAdapt before compression;\npreserve the current level separately',0.78,2.05,3.77,1.45,{fill:C.ice,stroke:C.blue,fontSize:21});
    labelBox(s,'Observed benefit\nRaw MSE reduced on\nJena (17.8%) and Appliances (18.9%)',4.78,2.05,3.77,1.45,{fill:C.pale,stroke:C.wine,fontSize:21});
    labelBox(s,'Component evidence\nP contributes on Jena;\nD contributes on ETTh1 and Appliances',8.78,2.05,3.77,1.45,{fill:'F2F5F2',stroke:C.green,fontSize:21});
    tx(s,branch.status==='COMPLETE' ? (branch.conclusion_en || branch.takeaway_en) : 'Jena branch follow-up: results pending in this draft.',0.97,3.91,11.36,0.75,{fontSize:22,bold:true,color:C.wine,align:'center'});
    bullet(s,'Scope: ETTh2 MSE and all four MAE results remain worse than Raw; comparator wins are provisional.',4.98,{fontSize:17,h:0.62});
    bullet(s,'K from PCA is a heuristic: Jena benefits from a larger tested K. Accuracy–cost trade-offs are conditional.',5.70,{fontSize:17,h:0.62});
    bullet(s,'Next: isolate the value of F, finish comparator optimization, then evaluate a frozen design independently.',6.42,{fontSize:17,h:0.37});
    note(number,'정리하겠습니다. 이 연구는 고정된 단변량 TSFM을 재사용하는 조건에서 압축 전 문맥 처리와 출력 복원 구조를 설계했습니다. 현재 수준은 마지막 값 경로로 보존하고 잠재 경로는 변화량을 표현합니다. 개발 검증 MSE는 Raw보다 Jena에서 17.8%, Appliances에서 18.9%, ETTh1에서 1.3% 낮았지만 ETTh2에서는 높았습니다. 기존 절제에서는 Jena의 P, ETTh1과 Appliances의 D 전체 기여가 확인됐습니다. '+(branch.status==='COMPLETE' ? (branch.conclusion_ko || branch.takeaway_ko) : '새 Jena 분기 결과는 아직 검증 전이므로 확정 결론에 포함하지 않았습니다.')+' 비교 어댑터의 연장이 남아 순위는 잠정적이고, MAE는 네 데이터 모두 Raw보다 높습니다. K 규칙도 예측 최적성을 보장하지 않습니다. 고정 F를 쓰는 추가 가치와 더 충분히 최적화한 비교 방법, 독립 테스트를 후속 과제로 남깁니다. 특정 백본·설정·개발 구간에서 얻은 결과로 한정합니다.');
    record(number,'Jena 및 D 기여 TBD; 폭넓은 효율성 서술','확인된 효과·조건부 비용·미해결 비교·F 가치·독립 테스트를 구분해 결론 완성. 새 분기 결과의 범위와 불확실성 반영.');
  }
  if(number===17){
    s.addShape(ST.rect,{x:1.75,y:4.61,w:4.35,h:0.93,fill:{color:'FFFFFF'},line:{color:'FFFFFF'}});
    tx(s,'Measured under default batching:\n217 / 243 / 294 MiB at K = 3 / 6 / 12.\nFixed 48 rows/call: K = 12 ≈ 243 MiB.',1.77,4.62,4.31,0.85,{fontSize:14.5,color:'404040'});
    note(number,'마름모는 A_fit 입력 설명분산 0.9 규칙이 고른 K이고, 회색 점은 이전에 학습한 다른 K입니다. 이 민감도 그림은 공통 seed 두 개의 평균이며 막대는 최소–최대입니다. 10 seed 비교·절제의 중앙값과 IQR 기준과 구분합니다. Appliances와 ETTh1에서는 시도한 K 중 규칙이 고른 K보다 더 나은 값이 관찰되지 않았습니다. 최적 K를 찾았다는 증명은 아닙니다. Jena는 K가 12일 때 규칙의 6보다 더 낮은 MSE여서 규칙의 한계를 기록했습니다. ETTh2는 시도한 모든 K가 Raw보다 나빴습니다. Jena의 217, 243, 294 MiB는 기본 행 배치 정책에서 K가 3, 6, 12일 때 측정한 값입니다. 고정 48행 정책에서는 K=12도 약 243 MiB이므로 K가 커질 때 메모리가 항상 증가한다고 일반화하지 않습니다. 규칙은 재현 가능한 출발점이지만 예측 최적성을 보장하지 않습니다. 이번 새 Jena 분기는 K를 6으로 고정하며 이 K 비교를 다시 선택하지 않습니다.');
    record(number,'K 규칙 민감도 대본과 Larger K uses more memory 문구','공통 두 seed 평균·min/max 범위를 명시. 메모리 문구는 기본 배치 측정과 고정48행 결과로 제한하여 항상 증가한다고 주장하지 않음.');
  }
  if(number===19){
    note(number,'들어 주셔서 감사합니다. 질문 받겠습니다.\n\n[예상 질문과 짧은 답]\n1. 왜 F를 고정하나? 기존 체크포인트의 가중치를 유지하며 외부 적응하는 연구 조건입니다. 내부 튜닝보다 항상 낫다는 뜻은 아닙니다.\n2. PEFT인가? 넓게는 작은 외부 파라미터만 학습하는 효율적 적응입니다. 이 발표에서는 외부 적응이라고 부르며, F 내부에 삽입하는 모듈과 위치를 구분합니다.\n3. P가 무엇을 추가하나? E는 선형 투영이고 F는 압축된 과거를 읽습니다. P는 압축 전에 전체 관측 문맥을 이용하는 비선형 표현 조정을 추가합니다.\n4. 시간 분기가 우월한가? 현재 시간·채널 분기의 용량이 다르고 정규화와 E/D도 채널을 혼합합니다. 분기 결과는 현재 구조의 추가 기여로만 해석합니다.\n5. Jena를 고른 이유는? 기존에 P 기여를 관찰한 데이터를 마감 내에 더 자세히 보는 탐색입니다. 결과를 보고 범위를 선택했다는 사실을 공개합니다.\n6. PCA 90%가 예측 정보 90%인가? 아닙니다. 초기 학습 입력의 설명분산 기준이며 학습된 표현이나 미래 변화 정보 비율이 아닙니다.\n7. D가 압축 손실을 복구하나? 현재 수준을 직접 보존하지만 잃은 모든 정보를 복구하는 것은 아닙니다. A4는 D 전체의 비교이며 lag만의 효과는 아직 미확인입니다.\n8. 하한의 약 90%가 수준 이동인가? 고정 PCA 출력 공간의 조건부 하한에서 평균 이동 항의 비율입니다. 전체 예측 오차나 개선량의 90%가 아닙니다.\n9. F가 고정이면 학습이 공짜인가? 아닙니다. 앞단 P/E 학습에는 F를 통과하는 입력 기울기가 필요합니다.\n10. Ridge나 Direct MLP보다 왜 F인가? 이번 연구는 고정 F의 외부 적응 조건을 다룹니다. 같은 어댑터에서 F를 대체한 추가 가치 검증은 별도 구현과 실행 예산이 필요해 후속 질문으로 남아 있습니다.\n11. 모든 비교에서 이겼나? 아닙니다. 비교 어댑터 연장이 남아 잠정적이며, ETTh2 MSE와 네 데이터의 MAE는 Raw보다 높습니다.\n12. 이번 결과를 테스트 성능이라고 부를 수 있나? 아닙니다. 이미 여러 번 살펴본 개발 V 결과이며, 보호 T는 접근하지 않았습니다.');
    record(number,'Jena 연장 진행 중 표현, PEFT 용어 불일치가 있는 Q&A','대본·Q&A에서 완료 상태와 연구 범위·용량·하한·독립 테스트 한계를 통일. 감사 인사 시각 자료는 보존.');
  }
  s.addNotes(notes[number]);
}

const label=final?'FINAL':'DRAFT';
const out=path.join(ROOT,final?'Temporal_Channel_Input_Adaptation_revised.pptx':'Temporal_Channel_Adaptation_Jena_DRAFT.pptx');
pres.writeFile({fileName:out}).then(()=>{
  const script = '# 발표 대본 및 질의응답\n\n'+(final?'검증된 Jena 분기 결과 반영본.':branch.status==='COMPLETE'?'검증된 Jena 분기 결과를 반영한 초안이며 최종 검수 전입니다.':'초안: 새 Jena 분기 실험 결과는 아직 반영 전입니다.')+'\n\n'+
    Object.entries(notes).map(([n,t])=>`## ${n}장\n\n${t}\n`).join('\n');
  fs.writeFileSync(path.join(ROOT,final?'SPEAKER_NOTES_KO.md':'SPEAKER_NOTES_KO_DRAFT.md'),script,'utf8');
  fs.writeFileSync(path.join(ROOT,'CHANGE_MAP.json'),JSON.stringify(changes,null,2),'utf8');
  fs.writeFileSync(path.join(ROOT,`BUILD_${label}.json`),JSON.stringify({output:out,slide_count:19,branch_status:branch.status,
    source_sha256:source.source_pdf_sha256,fully_rewritten_slides:[3,5,6,8,9,10,16,18],partially_rewritten_slide:7,partially_rewritten_slides:[7,14,15,17],
    notes_rewritten_slides:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],raster_preservation_other_slides:true,
    branch_results:branch,created_at:new Date().toISOString()},null,2),'utf8');
  fs.writeFileSync(path.join(ROOT,final?'CHANGELOG_KO.md':'CHANGELOG_KO_DRAFT.md'),
    '# 발표 자료 변경 대조\n\n'+(final?'검증된 Jena 분기 결과를 반영했습니다.':'초안이며 새 Jena 분기 수치는 아직 반영하지 않았습니다.')+'\n\n'+changes.map(c=>
      `## ${c.slide}장\n\n기존: ${c.before}\n\n수정: ${c.after}\n`).join('\n'),'utf8');
  console.log(out);
});
