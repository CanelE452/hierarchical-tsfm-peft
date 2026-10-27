"""Publication ZIP integrity and interrupted-finalization recovery."""
import json
import zipfile

import pytest

from research.comparative_ablation_20261025_v1 import bundle, runtime


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    report=tmp_path/'reports'/runtime.NAME
    source=tmp_path/'research'/runtime.NAME
    report.mkdir(parents=True)
    source.mkdir(parents=True)
    monkeypatch.setattr(runtime,'REPO',tmp_path)
    monkeypatch.setattr(runtime,'REPORT',report)
    monkeypatch.setattr(runtime,'CACHE',tmp_path/'.cache'/runtime.NAME)
    old=tmp_path/'original.txt'
    old.write_text('preserved',encoding='utf8')
    runtime.write_new(report/'PRE_HASHES.json',dict(scope='one fixture file',
        files=[dict(path=str(old),bytes=old.stat().st_size,sha256=runtime.sha(old))]))
    (source/'models.py').write_text('# fixture\n',encoding='utf8')
    (report/'RESULTS_KO.md').write_text('Synthetic report.\n',encoding='utf8')
    return report,source,old


def test_archive_contains_exact_report_source_and_replays_without_rewrite(fixture):
    report,source,old=fixture
    first=bundle.build_bundle()
    path=report/(runtime.NAME+'_bundle.zip')
    before=path.stat().st_mtime_ns
    assert first['zip_crc']=='PASS' and first['existing_files_unchanged']
    with zipfile.ZipFile(path) as archive:
        assert 'research/'+runtime.NAME+'/models.py' in archive.namelist()
        assert not any(name.endswith('.zip') for name in archive.namelist())
        assert archive.testzip() is None
    assert bundle.build_bundle()==first
    assert path.stat().st_mtime_ns==before
    assert old.read_text()=='preserved'


def test_packaging_resumes_after_manifest_but_before_zip_creation(fixture,monkeypatch):
    actual=bundle.zipfile.ZipFile
    def interrupted(*args,**kwargs):
        raise OSError('simulated interruption before ZIP creation')
    with monkeypatch.context() as scope:
        scope.setattr(bundle.zipfile,'ZipFile',interrupted)
        with pytest.raises(OSError):
            bundle.build_bundle()
    assert bundle.zipfile.ZipFile is actual
    assert bundle.build_bundle()['zip_crc']=='PASS'


def test_preservation_failure_is_visible_in_diagnostic_bundle(fixture):
    report,source,old=fixture
    old.write_text('external change',encoding='utf8')
    receipt=bundle.build_bundle()
    assert not receipt['existing_files_unchanged']
    post=json.loads((report/'POST_HASHES.json').read_text())
    assert post['status']=='FAIL_EXISTING_FILE_CHANGED'
    assert post['changed']==[str(old)]
