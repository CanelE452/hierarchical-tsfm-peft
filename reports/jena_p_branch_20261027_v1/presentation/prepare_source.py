"""Recover the supplied 19-slide deck from its notes-page PDF, read-only source."""
from pathlib import Path
import json
import hashlib
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF = Path('C:/Users/User/Downloads/Temporal–Channel Input Adaptation for Frozen Time-Series Foundation Models.pdf')
EXTRACT = ROOT.parents[2] / '.cache/claim_experiment_review_20261027_v1/pdf_pages.json'

def main():
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    records = json.loads(EXTRACT.read_text(encoding='utf-8'))
    groups = {}
    for item in records:
        first = item['text'].splitlines()[0].strip()
        number = int(first) if re.fullmatch(r'\d+', first) else (19 if first == 'Thank you!' else 1)
        groups.setdefault(number, []).append(item)
    assert sorted(groups) == list(range(1, 20))
    reader = PdfReader(PDF)
    source = {}
    for number, pages in groups.items():
        page_number = pages[0]['pdf_page']
        page = reader.pages[page_number - 1]
        im = page.images[0].image.convert('RGB')
        assert im.size == (3840, 2160)
        # The notes export places the original 16:9 slide at 40% scale.
        crop = im.crop((1152, 130, 2688, 994))
        filename = f'source_slide_{number:02d}.png'
        crop.save(assets / filename)
        content = []
        for item in pages:
            text = item['text']
            if '[발표 대본]' in text:
                text = text.split('[발표 대본]', 1)[1]
            elif '[제작자 근거' in text:
                text = text.split('[제작자 근거', 1)[1]
            else:
                # The full extracted page remains in the original source archive.
                continue
            content.append(text.strip())
        source[str(number)] = {'pdf_pages': [p['pdf_page'] for p in pages], 'image': filename,
                               'original_notes': '\n\n'.join(content)}
    # Branding is an unchanged raster element from the supplied slide.
    from PIL import Image
    logo = Image.open(assets / 'source_slide_07.png').crop((5, 818, 205, 858))
    logo.save(assets / 'source_branding.png')
    manifest = {'source_pdf': str(PDF), 'source_pdf_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
                'source_kind': 'raster notes PDF with hidden OCR; no matching editable source found',
                'crop_box_pixels': [1152, 130, 2688, 994], 'slide_count': 19, 'slides': source}
    (ROOT / 'SOURCE_RECOVERY.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'slides': 19, 'assets': str(assets), 'source_sha256': manifest['source_pdf_sha256']}))

if __name__ == '__main__':
    main()
