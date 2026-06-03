"""
根据 legacy_eeg_classifier_eval 得到的
paper_results/classification_legacy/classification_legacy_results.json
计算三种场景的均值和标准差，方便更新论文表格。
"""
import json
import os
import numpy as np


def summarize_scenario(scenario_data, scenario_name):
    subjects = [k for k in scenario_data.keys() if k.startswith("S")]
    print(f"\n=== {scenario_name} ===")
    for metric in ["baseline_9ch", "baseline_22ch", "dttd_augmented"]:
        vals = []
        for sid in subjects:
            v = scenario_data[sid][metric]["accuracy"]
            vals.append(v)
        vals = np.asarray(vals)
        mean = vals.mean() * 100
        std = vals.std() * 100
        print(f"{metric}: {mean:.2f} ± {std:.2f} %")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        root,
        "paper_results",
        "classification_legacy",
        "classification_legacy_results.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summarize_scenario(data["within_subject"], "within_subject")
    summarize_scenario(data["cross_session"], "cross_session")
    summarize_scenario(data["cross_subject"], "cross_subject")


if __name__ == "__main__":
    main()


