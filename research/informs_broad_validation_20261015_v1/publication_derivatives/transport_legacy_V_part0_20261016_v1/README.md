# Legacy public-statistic archive byte transport

This is a third transport format. The frozen publication inspector does not support it; independent publication admission is required.
The original archive and original MANIFEST remain unchanged locally. No removal, staging, or publication is performed.

Restore the exact original repository-relative part into a clone or separate private directory:

```text
python research/informs_broad_validation_20261015_v1/stats_pack_transport.py restore --manifest TRANSPORT_DIR/MANIFEST.json --destination CLONE_ROOT
```

Restored part: `research/informs_broad_validation_20261015_v1/stats_packs/pack_1792065325663656600/part_0000.tar.xz`.
The embedded original MANIFEST, each original member, canonical tar header/order/padding, and original XZ bytes are SHA-verified.
The original preset6/CRC64 XZ bytes require a reproducible liblzma environment; a mismatch fails without writing.
Exact existing destinations are read and skipped; mismatches are never overwritten. Original sufficient-statistic gzips are restored separately by the original stats_pack.py and original MANIFEST.
Only already-public aggregate members are transported, with exact numeric/literal or raw common9 fallback and opaque gzip bytes. No rounding, row removal, new targets, model access, or scientific re-evaluation occurs.
Archive-only savings exclude JSON/README sidecars. This byte-transport PASS does not replace the old publication or scientific gates.
