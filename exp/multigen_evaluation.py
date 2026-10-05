"""Frozen-protocol evaluation of aligned classification predictions.

This stage is NOT run by data collection. Test access is denied before opening
prediction files or exported data unless all four evaluation artifacts are locked.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import numpy as np
from multigen_data import common as C
from multigen_data.processing import clustered_bootstrap, evaluation_rows

METRICS = ("accuracy", "net_delta_per_pair", "accept_precision", "accept_recall", "macro_f1")

def point_metrics(rows, decisions):
    if len(rows) != len(decisions) or any(x not in ("Accept","Reject") for x in decisions):
        raise ValueError("Predictions must align with every pair")
    truth = np.array([r["label"] for r in rows])
    pred = np.array(decisions)
    accepted = pred == "Accept"
    positive = truth == "Accept"
    tp = int((accepted & positive).sum())
    fp = int((accepted & ~positive).sum())
    fn = int((~accepted & positive).sum())
    tn = int((~accepted & ~positive).sum())
    safe = lambda a,b: a/b if b else 0.
    delta = np.array([r["metadata"]["delta"] for r in rows], dtype=float)
    return {"accuracy":float((pred == truth).mean()),
            "net_delta_per_pair":float((accepted*delta).mean()),
            "accept_precision":safe(tp,tp+fp),"accept_recall":safe(tp,tp+fn),
            "macro_f1":(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2}

def prediction_report(rows, predictions, *, draws=2000, seed=C.MASTER_SEED):
    if not rows or not predictions:
        raise ValueError("Rows and comparison predictions must be nonempty")
    samples = clustered_bootstrap(rows,predictions,draws=draws,seed=seed)
    points = {name:point_metrics(rows,values) for name,values in predictions.items()}
    result = {"pairs":len(rows),
              "independent_sources":len({r["metadata"]["cluster_id"] for r in rows}),
              "bootstrap":{"unit":"source","draws":draws,"seed":seed,"confidence":.95,
                           "interval":"percentile","paired_resampling":True},
              "models":{},"paired_differences":{},
              "undefined_precision_recall_convention":0.0}
    for name in predictions:
        result["models"][name] = {
            metric:{"estimate":points[name][metric],
                    "ci95":np.quantile(samples[name][metric],[.025,.975]).tolist()}
            for metric in METRICS}
    for a,b in itertools.combinations(predictions,2):
        result["paired_differences"][a+" - "+b] = {
            metric:{"estimate":points[a][metric]-points[b][metric],
                    "ci95":np.quantile(samples[a][metric]-samples[b][metric],[.025,.975]).tolist()}
            for metric in METRICS}
    return result

def evaluate_predictions(directory, split, prediction_paths):
    directory = Path(directory)
    C.require_unsealed(directory,split)  # Must precede every data/prediction read.
    if split == "test":
        lock = C.read(directory/"evaluation_lock.json")
        frozen = lock["evaluation"]
        if set(prediction_paths) != set(frozen["comparisons"]):
            raise PermissionError("Comparison set differs from the frozen evaluation protocol")
        codepath = str(Path(__file__).resolve())
        if frozen["artifacts"].get(codepath) != C.file_hash(codepath):
            raise PermissionError("Freeze this evaluation implementation before unsealing Test")
        if frozen.get("metrics") != list(METRICS):
            raise PermissionError("Evaluation metrics were not frozen")
    rows = evaluation_rows(directory,split)
    expected = {r["id"] for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Evaluation pairs must be unique")
    predictions, artifacts = {}, {}
    for name,path in prediction_paths.items():
        values = {}
        for r in C.jsonl(path):
            if r["id"] in values:
                raise ValueError("Duplicate prediction pair ID")
            values[r["id"]] = r["decision"]
        if set(values) != expected:
            raise ValueError("Predictions do not exactly cover the frozen evaluation pairs")
        predictions[name] = [values[r["id"]] for r in rows]
        artifacts[name] = {"path":str(path),"sha256":C.file_hash(path)}
    result = prediction_report(rows,predictions)
    result.update(task=C.read(directory/"protocol.json")["task"],split=split,
                  protocol_sha256=C.file_hash(directory/"protocol.json"),
                  predictions=artifacts,
                  evaluation_code_sha256=C.file_hash(__file__),
                  at=C.utc())
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--split",choices=("dev","test"),required=True)
    parser.add_argument("--predictions",action="append",required=True,help="name=/path/to/id-decision.jsonl")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.predictions:
        name,path = item.split("=",1)
        if name in paths:
            raise ValueError("Repeated comparison name")
        paths[name] = Path(path)
    report = evaluate_predictions(args.data_dir,args.split,paths)
    C.dump(args.output,report,immutable=True)
    print(C.canonical(report))

if __name__ == "__main__":
    main()
