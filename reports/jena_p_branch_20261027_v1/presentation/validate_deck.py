"""Check the generated presentation and attach native IQR error bars before export."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import zipfile

from lxml import etree as ET
from pypdf import PdfReader
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def cached_values(node, name):
    if node is None:
        raise ValueError('Missing native chart element: '+name)
    points = node.xpath('.//c:numCache/c:pt|.//c:numLit/c:pt|.//c:strCache/c:pt|.//c:strLit/c:pt', namespaces=NS)
    points.sort(key=lambda p: int(p.get('idx')))
    if not points:
        raise ValueError('Missing native chart cache: '+name)
    return [p.find('c:v',NS).text or '' for p in points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--decorate', action='store_true')
    parser.add_argument('--renders')
    args = parser.parse_args()
    label = 'FINAL' if args.final else 'DRAFT'
    build = json.loads((ROOT / f'BUILD_{label}.json').read_text(encoding='utf8'))
    pptx = Path(build['output'])
    pdf = pptx.with_suffix('.pdf')
    branch = build['branch_results']
    if args.decorate:
        if not args.final:
            return
        if branch['status'] != 'COMPLETE':
            raise ValueError('Cannot decorate a final chart without actual results')
        with zipfile.ZipFile(pptx) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        chart_parts = [n for n in parts if n.startswith('ppt/charts/chart') and n.endswith('.xml')]
        if len(chart_parts) != 1:
            raise ValueError('Expected one native Jena chart')
        chart = ET.fromstring(parts[chart_parts[0]])
        series = chart.find('.//c:barChart/c:ser', NS)
        if series is None:
            raise ValueError('Native Jena bar series missing')
        for old in series.findall('c:errBars', NS):
            series.remove(old)
        def el(name, **attrs):
            return ET.Element('{'+NS['c']+'}'+name, **attrs)
        bars = el('errBars')
        for tag, value in [('errDir','y'),('errBarType','both'),('errValType','cust'),('noEndCap','0')]:
            bars.append(el(tag, val=value))
        for side, values in [('plus',[r['q3']-r['median_mse'] for r in branch['rows']]),
                             ('minus',[r['median_mse']-r['q1'] for r in branch['rows']])]:
            error_side = el(side)
            literals = el('numLit')
            fmt = el('formatCode'); fmt.text = '0.000000'; literals.append(fmt)
            literals.append(el('ptCount', val=str(len(values))))
            for i, value in enumerate(values):
                point = el('pt', idx=str(i)); v = el('v'); v.text = str(value); point.append(v); literals.append(point)
            error_side.append(literals); bars.append(error_side)
        series.insert(series.index(series.find('c:cat', NS)), bars)
        parts[chart_parts[0]] = ET.tostring(chart, xml_declaration=True, encoding='UTF-8', standalone=True)
        temporary = pptx.with_suffix('.iqr.pptx')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for n, data in parts.items():
                z.writestr(n, data)
        temporary.replace(pptx)
        print('Native Q1–Q3 error bars inserted from verified data.')
        return
    with zipfile.ZipFile(pptx) as z:
        if z.testzip() is not None:
            raise ValueError('PPTX CRC failure')
        all_parts = z.namelist()
        slides = [n for n in all_parts if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
        note_parts = [n for n in all_parts if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')]
        assert len(slides) == 19 and len(note_parts) == 19
        for n in all_parts:
            if n.endswith('.xml') or n.endswith('.rels'):
                ET.fromstring(z.read(n))
        content = {}
        for n in range(1,20):
            xml = ET.fromstring(z.read(f'ppt/slides/slide{n}.xml'))
            content[n] = '\n'.join(xml.xpath('.//a:t/text()', namespaces=NS))
        for n in build['fully_rewritten_slides']:
            if not content[n].strip():
                raise ValueError('Rewritten slide has no native editable text: '+str(n))
        if args.final:
            if branch['status'] != 'COMPLETE':
                raise ValueError('Final presentation lacks verified branch data')
            for n in [16,18]:
                if any(word in content[n].lower() for word in ['tbd','pending','in progress','draft','will replace']):
                    raise ValueError('Unresolved placeholder in final result slide '+str(n))
            charts = [n for n in all_parts if n.startswith('ppt/charts/chart') and n.endswith('.xml')]
            chart = ET.fromstring(z.read(charts[0]))
            series = chart.find('.//c:barChart/c:ser', NS)
            if series is None:
                raise ValueError('Final native bar series missing')
            expected_rows = branch['rows']
            categories = cached_values(series.find('c:cat',NS),'categories')
            if [normalized_text(t) for t in categories] != [normalized_text(r['chart_label']) for r in expected_rows]:
                raise ValueError('Office chart labels differ from verified method/schedule labels')
            for actual, expected, name in [
                (cached_values(series.find('c:val',NS),'medians'), [r['median_mse'] for r in expected_rows], 'median'),
                (cached_values(series.find('c:errBars/c:plus',NS),'IQR plus'), [r['q3']-r['median_mse'] for r in expected_rows], 'Q3-median'),
                (cached_values(series.find('c:errBars/c:minus',NS),'IQR minus'), [r['median_mse']-r['q1'] for r in expected_rows], 'median-Q1')]:
                if len(actual) != len(expected) or any(abs(float(a)-b)>1e-12 for a,b in zip(actual,expected)):
                    raise ValueError('Office chart values differ from verified '+name)
            note16 = ET.fromstring(z.read('ppt/notesSlides/notesSlide16.xml'))
            actual_note = normalized_text(' '.join(note16.xpath('.//a:t/text()',namespaces=NS)))
            if normalized_text(branch['schedule_notes_ko']) not in actual_note:
                raise ValueError('Actual PPTX notes lost selection epoch/schedule metadata')
    pdf_reader = PdfReader(pdf)
    if len(pdf_reader.pages) != 19:
        raise ValueError('Expected 19 PDF slides')
    source = json.loads((ROOT/'SOURCE_RECOVERY.json').read_text(encoding='utf8'))
    if sha(source['source_pdf']) != source['source_pdf_sha256']:
        raise ValueError('Original attached PDF changed')
    if args.final:
        page16 = normalized_text(pdf_reader.pages[15].extract_text())
        page18 = normalized_text(pdf_reader.pages[17].extract_text())
        if 'exploratory' not in page16.lower():
            raise ValueError('Final PDF 16 omits exploratory status')
        for row in branch['rows']:
            if normalized_text(row['chart_label']) not in page16:
                raise ValueError('Final PDF 16 omits method/cap/extension/provisional label: '+row['id'])
        if normalized_text(branch['conclusion_en']) not in page18:
            raise ValueError('Final PDF 18 conclusion differs from verified contrasts')
        if normalized_text(branch['schedule_notes_ko']) not in normalized_text((ROOT/'SPEAKER_NOTES_KO.md').read_text(encoding='utf8')):
            raise ValueError('Final speaker notes omit actual schedule metadata')
        with (ROOT.parent/'SUMMARY.csv').open(encoding='utf-8-sig', newline='') as f:
            summary = {r['method_id']:r for r in csv.DictReader(f)}
        for r in branch['rows']:
            s = summary[r['id']]
            assert int(s['n']) == 10
            for source_key, deck_key in [('MSE_median','median_mse'),('MSE_Q1','q1'),('MSE_Q3','q3')]:
                assert abs(float(s[source_key])-r[deck_key]) < 1e-12
            assert json.loads(s['selected_epochs']) == r['selected_epochs']
    render_dir = Path(args.renders) if args.renders else None
    renders = sorted(render_dir.glob('slide-*.png')) if render_dir else []
    if len(renders) != 19:
        raise ValueError('Expected 19 final-resolution rendered slides')
    sheet = Image.new('RGB', (1600, 1250), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for i, render in enumerate(renders):
        with Image.open(render) as frame:
            thumb = frame.convert('RGB'); thumb.thumbnail((400,225))
        x, y = i%4*400, i//4*250
        sheet.paste(thumb, (x,y+25)); draw.text((x+5,y+5), str(i+1), fill='black')
    sheet.save(render_dir/'contact_sheet.png')
    files = [pptx,pdf,ROOT/('SPEAKER_NOTES_KO.md' if args.final else 'SPEAKER_NOTES_KO_DRAFT.md'),
             ROOT/('CHANGELOG_KO.md' if args.final else 'CHANGELOG_KO_DRAFT.md')]
    report = {'status':'PASS','scope':'Technical validation; final visual review is a separate root-agent gate.',
              'slide_count':19,'notes_count':19,'final_actual_data':args.final,
              'source_pdf_preserved':True,'native_text_slides':build['fully_rewritten_slides'],
              'native_chart_with_Q1_Q3':args.final,'branch_rows':branch['rows'],
              'office_chart_data_and_labels_verified':args.final,
              'pdf16_schedule_labels_verified':args.final,'pdf18_conclusion_verified':args.final,
              'actual_pptx_note16_epoch_distribution_verified':args.final,
              'branch_provenance':branch.get('provenance'),
              'output_hashes':{p.name:sha(p) for p in files},
              'render_paths':[str(p) for p in renders],
              'contact_sheet':str(render_dir/'contact_sheet.png'),
              'visual_review_status':'PENDING_ROOT_REVIEW'}
    target = ROOT/('PRESENTATION_VALIDATION.json' if args.final else 'PRESENTATION_VALIDATION_DRAFT.json')
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(str(target))


if __name__ == '__main__':
    main()
