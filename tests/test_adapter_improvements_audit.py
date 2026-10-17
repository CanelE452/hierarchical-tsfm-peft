from types import SimpleNamespace
from research.adapter_improvements_20261018_v1 import audit


def test_pems_loader_unpacks_pinned_task_and_retains_access_proof(monkeypatch):
    task = dict(H=12,L=512,split_ranges=dict(A_fit=(0,8928)))
    module = SimpleNamespace(load_task=lambda public, private: (task,{"protected_T_numeric_parse_count":0},{"sealed":True}))
    monkeypatch.setattr(audit.importlib,"import_module",lambda name:module)
    ctx = {}
    actual = audit.load_task(ctx,"pems32",dict(campaign="public",old_cache="private"))
    assert actual is task
    assert ctx["task_access"]["pems32"]["protected_T_numeric_parse_count"] == 0
