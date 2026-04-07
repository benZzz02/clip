import json
import os

import numpy as np
import pandas as pd


PER_VIDEO_SKIP_KEYS = {
    "accuracy",
    "f1_average",
    "f1_per_class",
    "video_ids",
    "mAP",
    "AP_per_class",
}

NON_METRIC_COLUMNS = {
    "dataset",
    "ckpt",
    "model_family",
    "text_model",
    "tokenizer_name",
    "vision_weights",
    "output_dir",
    "batch_size",
    "num_workers",
    "embed_dim",
    "num_frames",
    "model_num_frames",
    "frame_stride",
    "temporal_layers",
    "temporal_heads",
    "temporal_dropout",
    "image_size",
    "result_json",
}


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_numeric_scalar(value):
    value = _to_python_scalar(value)
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def flatten_metrics(value, prefix="", skip_keys=None, max_sequence_len=256):
    skip_keys = PER_VIDEO_SKIP_KEYS if skip_keys is None else set(skip_keys)
    flat = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(
                flatten_metrics(
                    child,
                    prefix=child_prefix,
                    skip_keys=skip_keys,
                    max_sequence_len=max_sequence_len,
                )
            )
        return flat

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        key_name = prefix.split(".")[-1] if prefix else ""
        if key_name in skip_keys or len(value) > max_sequence_len:
            return flat

        for idx, child in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flat.update(
                flatten_metrics(
                    child,
                    prefix=child_prefix,
                    skip_keys=skip_keys,
                    max_sequence_len=max_sequence_len,
                )
            )
        return flat

    scalar = _to_python_scalar(value)
    if _is_numeric_scalar(scalar):
        flat[prefix] = float(scalar)
    return flat


def _build_metric_table(metrics):
    rows = [{"metric": metric, "value": value} for metric, value in sorted(metrics.items())]
    return pd.DataFrame(rows)


def _upsert_summary_row(summary_path, row, key_columns):
    row_df = pd.DataFrame([row])
    if not os.path.exists(summary_path):
        row_df.to_csv(summary_path, index=False)
        return

    summary_df = pd.read_csv(summary_path)
    all_columns = list(dict.fromkeys(list(summary_df.columns) + list(row_df.columns)))
    summary_df = summary_df.reindex(columns=all_columns)
    row_df = row_df.reindex(columns=all_columns)

    valid_keys = [col for col in key_columns if col in summary_df.columns and col in row_df.columns]
    if valid_keys:
        mask = pd.Series(True, index=summary_df.index)
        for col in valid_keys:
            mask &= summary_df[col].astype(str) == str(row[col])
        summary_df = summary_df.loc[~mask]

    summary_df = pd.concat([summary_df, row_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)


def load_reference_metrics(reference_path, dataset=None):
    suffix = os.path.splitext(reference_path)[1].lower()

    if suffix == ".json":
        with open(reference_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if dataset and isinstance(payload, dict) and dataset in payload and isinstance(payload[dataset], dict):
            payload = payload[dataset]
        return flatten_metrics(payload)

    if suffix == ".csv":
        df = pd.read_csv(reference_path)

        if {"metric", "value"}.issubset(df.columns):
            if dataset and "dataset" in df.columns:
                dataset_df = df[df["dataset"].astype(str) == str(dataset)]
                if not dataset_df.empty:
                    df = dataset_df
            return {
                str(metric): float(value)
                for metric, value in zip(df["metric"], df["value"])
                if _is_numeric_scalar(value)
            }

        if dataset and "dataset" in df.columns:
            dataset_df = df[df["dataset"].astype(str) == str(dataset)]
            if not dataset_df.empty:
                df = dataset_df

        if df.empty:
            return {}

        row = df.iloc[0].to_dict()
        metrics = {}
        for key, value in row.items():
            if key in NON_METRIC_COLUMNS:
                continue
            if _is_numeric_scalar(value):
                metrics[str(key)] = float(value)
        return metrics

    raise ValueError(f"Unsupported reference file format: {reference_path}")


def build_sota_comparison_table(metrics, reference_metrics):
    shared_metrics = sorted(set(metrics) & set(reference_metrics))
    rows = []
    for metric in shared_metrics:
        ours = float(metrics[metric])
        sota = float(reference_metrics[metric])
        rows.append(
            {
                "metric": metric,
                "ours": ours,
                "sota": sota,
                "gap_to_sota": ours - sota,
                "ratio_to_sota": (ours / sota) if abs(sota) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def export_evaluation_reports(results, dataset, output_dir, metadata=None, sota_file=None):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {} if metadata is None else dict(metadata)

    dataset_results = results.get(dataset, results) if isinstance(results, dict) else results
    metrics = flatten_metrics(dataset_results)

    wide_row = {**metadata, **metrics}
    summary_wide_path = os.path.join(output_dir, f"summary_{dataset}.csv")
    pd.DataFrame([wide_row]).to_csv(summary_wide_path, index=False)

    metrics_table = _build_metric_table(metrics)
    summary_long_path = os.path.join(output_dir, f"summary_{dataset}_metrics.csv")
    metrics_table.to_csv(summary_long_path, index=False)

    aggregate_path = os.path.join(output_dir, "summary_all.csv")
    _upsert_summary_row(
        aggregate_path,
        wide_row,
        key_columns=["dataset", "model_family", "ckpt"],
    )

    report_paths = {
        "summary_wide_csv": summary_wide_path,
        "summary_metrics_csv": summary_long_path,
        "summary_all_csv": aggregate_path,
    }

    if sota_file:
        reference_metrics = load_reference_metrics(sota_file, dataset=dataset)
        comparison_table = build_sota_comparison_table(metrics, reference_metrics)
        comparison_path = os.path.join(output_dir, f"summary_{dataset}_vs_sota.csv")
        comparison_table.to_csv(comparison_path, index=False)
        report_paths["sota_comparison_csv"] = comparison_path

    return report_paths
