import pytest
from multigen_evaluation import evaluate_predictions,prediction_report

def sample_rows():
    return [
        {"id":"a","label":"Accept","metadata":{"cluster_id":"one","delta":.3}},
        {"id":"b","label":"Accept","metadata":{"cluster_id":"one","delta":.1}},
        {"id":"c","label":"Reject","metadata":{"cluster_id":"two","delta":-.2}},
    ]

def test_report_counts_sources_and_paired_identical_models_have_zero_difference():
    report=prediction_report(sample_rows(),{"a":["Accept"]*3,"b":["Accept"]*3})
    assert report["pairs"]==3 and report["independent_sources"]==2
    assert report["bootstrap"]["draws"]==2000
    assert report["models"]["a"]["accuracy"]["estimate"]==2/3
    assert report["models"]["a"]["accuracy"]["ci95"]==[0.,1.]
    for metric in report["paired_differences"]["a - b"].values():
        assert metric=={"estimate":0.,"ci95":[0.,0.]}

def test_sealed_test_fails_before_reading_missing_predictions(tmp_path):
    with pytest.raises(PermissionError,match="sealed"):
        evaluate_predictions(tmp_path,"test",{"model":tmp_path/"does-not-exist.jsonl"})
    assert not list(tmp_path.iterdir())

def test_incomplete_or_invalid_predictions_are_rejected():
    with pytest.raises(ValueError):
        prediction_report(sample_rows(),{"model":["Accept"]})
    with pytest.raises(ValueError):
        prediction_report(sample_rows(),{"model":["Accept","Reject","Better"]})
