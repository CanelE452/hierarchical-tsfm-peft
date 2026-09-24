#!/usr/bin/env python
"""P1/P2 period construction (contract 5.1) and PERIODS.json.

The official loader always puts the evaluation windows at the end of each series. A period is built
by truncating every series so that the period's evaluation windows become the final ones, then
reusing the official gluonts split so that window geometry, stride and metric masking stay identical.
"""
from __future__ import annotations
import numpy as np
import ga


def truncate(entry: dict, drop: int) -> dict:
    """Return a copy of a gluonts entry with the last `drop` steps removed."""
    out = dict(entry)
    t = np.asarray(entry['target'])
    out['target'] = t[..., :t.shape[-1]-drop] if drop else t
    for key in ('past_feat_dynamic_real',):
        if key in out and out[key] is not None:
            v = np.asarray(out[key])
            out[key] = v[..., :v.shape[-1]-drop] if drop else v
    return out


class Period:
    """One evaluation period with its own training segment, validation block and test windows."""

    def __init__(self, cfg, name: str):
        assert name in ('P1', 'P2')
        self.cfg, self.name = cfg, name
        ds = ga.dataset(cfg)
        self.freq = ds.freq
        self.H = ds.prediction_length
        self.W = ds.windows
        self.V = min(3, self.W)
        self.target_dim = ds.target_dim
        self.entries = [ga_entry for ga_entry in ds.gluonts_dataset]
        self.drop = self.W*self.H if name == 'P1' else 0     # P1 hides P2 entirely
        self.series_count = len(self.entries)

    # ---- datasets -------------------------------------------------------
    def _base(self):
        return [truncate(e, self.drop) for e in self.entries]

    def test_data(self):
        from gluonts.dataset.split import split
        _, template = split(self._base(), offset=-self.H*self.W)
        return template.generate_instances(prediction_length=self.H, windows=self.W, distance=self.H)

    def validation_data(self):
        """The V windows immediately before this period's evaluation windows."""
        from gluonts.dataset.split import split
        base = [truncate(e, self.drop + self.W*self.H) for e in self.entries]
        _, template = split(base, offset=-self.H*self.V)
        return template.generate_instances(prediction_length=self.H, windows=self.V, distance=self.H)

    def training_series(self):
        """Everything strictly before this period's validation block."""
        cut = self.drop + (self.W + self.V)*self.H
        return [truncate(e, cut) for e in self.entries]

    def summary(self):
        lens = [np.asarray(e['target']).shape[-1] for e in self.training_series()]
        return dict(period=self.name, key=self.cfg['key'], H=self.H, windows=self.W,
                    validation_windows=self.V, series=len(self.entries), target_dim=int(self.target_dim),
                    drop_from_end=self.drop, train_length_min=int(min(lens)), train_length_max=int(max(lens)),
                    eval_offset_from_series_end=[-(self.drop + self.W*self.H), -self.drop or None],
                    required_train_length=ga.CONTEXT_FIT + 5*self.H,
                    feasible=bool(min(lens) >= ga.CONTEXT_FIT + 5*self.H))


def build():
    out = {}
    for cfg in ga.CONFIGS:
        g = ga.geometry(cfg)
        rec = dict(geometry=g)
        if g['feasible']:
            for name in ('P1', 'P2'):
                rec[name] = Period(cfg, name).summary()
        else:
            rec['status'] = 'INFEASIBLE'
            rec['reason'] = (f"P1 training segment {g['train_length_P1']} < required "
                             f"{g['required_train_length']} (contract 5.2); excluded and not replaced")
        out[cfg['id']] = rec
        print(f"{cfg['id']} {cfg['key']:32s} {'INFEASIBLE' if not g['feasible'] else 'OK'}", flush=True)
    ga.write_json(ga.OUT/'PERIODS.json', dict(utc=__import__('time').time(), per_config=out,
                  rule='P2 = official test windows; P1 = the W windows immediately before them; '
                       'each period holds out its own validation block of min(3, W) windows'))
    return out


if __name__ == '__main__':
    build()
