import numpy as np
import torch

from gtm_detector.ml_freq.config import DecoderConfig, ExperimentConfig, TrainingConfig, TransformerConfig
from gtm_detector.ml_freq.data import Interval
from gtm_detector.ml_freq.evaluation import CrossValidator, EventMatcher, SegmentDecoder
from gtm_detector.ml_freq.models import (
    SklearnSegmenter,
    TimeSeriesTransformer,
    TransformerSegmenter,
    classic_models,
    load_model,
)
from gtm_detector.ml_freq.tests.conftest import day, make_well

SMALL = TransformerConfig(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
QUICK = TrainingConfig(epochs=3, crop_length=64)


def test_decoder_merges_gaps_and_marks_open_end():
    proba = np.zeros(40)
    proba[5:15] = 0.9
    proba[16:20] = 0.9  # разрыв в 1 день — склеится
    proba[35:] = 0.9  # доходит до конца — эффект продолжается
    decoder = SegmentDecoder(DecoderConfig(smooth_days=1, max_gap=2, min_length=2))
    events = decoder.decode(proba, [day(i) for i in range(40)])
    assert events == [Interval(day(5), day(19)), Interval(day(35), None)]


def test_matcher_one_to_one_by_start():
    labels = [Interval(day(10), day(20)), Interval(day(60), day(70))]
    predicted = [Interval(day(11), day(20)), Interval(day(12), day(25)), Interval(day(100), None)]
    score = EventMatcher(start_tolerance=7).score(predicted, labels)
    assert (score.true_positives, score.false_negatives, score.false_positives) == (1, 1, 2)
    assert score.start_errors == [1] and score.end_errors == [0]


def test_transformer_forward_shape():
    network = TimeSeriesTransformer(n_channels=5, config=SMALL)
    logits = network(torch.randn(2, 30, 5))
    assert logits.shape == (2, 30)


def test_transformer_fit_predict_save_load(tmp_path, wells):
    model = TransformerSegmenter(SMALL, QUICK).fit(wells)
    proba = model.predict_proba(wells[0])
    assert proba.shape == (len(wells[0]),) and ((proba >= 0) & (proba <= 1)).all()
    path = tmp_path / "transformer.pt"
    model.save(path)
    assert np.allclose(load_model(path).predict_proba(wells[0]), proba, atol=1e-6)


def test_sklearn_models_learn_the_effect(tmp_path, wells):
    model = classic_models()["gboost"]().fit(wells)
    test = make_well(step=50, effect=30, seed=99)
    proba = model.predict_proba(test)
    assert proba[60:75].mean() > 0.7 and proba[:40].mean() < 0.3
    path = tmp_path / "gboost.joblib"
    model.save(path)
    assert isinstance(load_model(path), SklearnSegmenter)


def test_cross_validation_runs_for_all_models(wells):
    models = {"logreg": classic_models()["logreg"], "transformer": lambda: TransformerSegmenter(SMALL, QUICK)}
    results = CrossValidator(ExperimentConfig(folds=2)).run(wells, models)
    assert [r.name for r in results] == ["logreg", "transformer"]
    for result in results:
        assert result.score.true_positives + result.score.false_negatives == len(wells)
        assert 0.0 <= result.day_auc <= 1.0
