import numpy as np
from statsforecast.models import SeasonalNaive, AutoETS
from hier_peft.data import load_and_audit


def test_statistical_baselines_finite_on_real_train_contexts():
    ds = load_and_audit("Labour", write_artifacts=False)
    for cls in [SeasonalNaive, AutoETS]:
        prediction = np.stack([cls(season_length=12).forecast(y=y[:48].copy(), h=12)["mean"] for y in ds.values])
        assert prediction.shape == (57, 12)
        assert np.isfinite(prediction).all()
