"""
PhysioNet MI 数据预提取脚本

一次性从EDF文件读取所有被试数据，保存为.npy格式，后续加载只需1秒。
输出文件: E:/data/PhysioNetMI/physionet_mi_preprocessed.npz
  - data: (N_subjects, ) object array, 每个元素为 (n_trials, 64, 640) float32
  - labels: (N_subjects, ) object array, 每个元素为 (n_trials,) int64
  - subject_ids: 有效被试ID列表
"""

import os
import sys
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.physionet_mi import (
    PhysioNetMIDataset, CHANNEL_NAMES_64, INPUT_CHANNEL_INDICES_16,
    MI4C_RUNS, parse_subject_run_from_name, load_physionet_events,
    map_run_desc_to_label
)
import mne


def extract_all_subjects(data_path, output_path, max_subject=109):
    """提取所有被试数据并保存为npz"""
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_path}")
    print(f"最大被试数: {max_subject}")
    print("=" * 60)

    all_data = {}
    valid_ids = []

    t0 = time.time()

    for sid in range(1, max_subject + 1):
        subject_str = f'S{sid:03d}'
        data_dir = os.path.join(data_path, subject_str)

        if not os.path.exists(data_dir):
            continue

        # 找EDF文件
        edf_files = []
        for f in os.listdir(data_dir):
            if f.lower().endswith('.edf'):
                try:
                    _, run = parse_subject_run_from_name(f)
                    if run in MI4C_RUNS:
                        edf_files.append(os.path.join(data_dir, f))
                except Exception:
                    continue

        if not edf_files:
            continue

        edf_files.sort()

        # 提取trials
        subject_trials = []
        subject_labels = []

        for edf_file in edf_files:
            try:
                _, run = parse_subject_run_from_name(os.path.basename(edf_file))
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose="ERROR")

                # 选择EEG通道
                eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, exclude="bads")
                raw.pick(eeg_picks)

                # 重排通道到标准10-10顺序
                edf_ch_names = [ch.upper() for ch in raw.ch_names]
                reorder_idx = []
                for std_ch in CHANNEL_NAMES_64:
                    std_upper = std_ch.upper()
                    if std_upper in edf_ch_names:
                        reorder_idx.append(edf_ch_names.index(std_upper))
                if len(reorder_idx) == len(raw.ch_names):
                    raw.reorder_channels([raw.ch_names[i] for i in reorder_idx])

                # 预处理
                target_sfreq = 160
                if raw.info['sfreq'] != target_sfreq:
                    raw.resample(target_sfreq, verbose=False)
                raw.notch_filter(50.0, verbose=False)
                raw.filter(4.0, 30.0, fir_design='firwin', verbose=False)
                raw.set_eeg_reference('average', verbose=False)

                # 获取事件
                event_pairs = load_physionet_events(raw, edf_file)

                # 提取trials
                sfreq = raw.info["sfreq"]
                tmin, tmax = 0.0, 4.0
                start_offset = int(round(tmin * sfreq))
                end_offset = int(round(tmax * sfreq))
                time_steps = 640
                data_arr = raw.get_data().astype(np.float32, copy=False)

                for onset, desc in event_pairs:
                    if desc not in {"T1", "T2"}:
                        continue
                    label = map_run_desc_to_label(run, desc)
                    if label is None:
                        continue
                    start = int(onset) + start_offset
                    stop = int(onset) + end_offset
                    if start < 0 or stop > data_arr.shape[1] or stop <= start:
                        continue
                    seg = data_arr[:, start:stop]
                    if seg.shape[1] > time_steps:
                        seg = seg[:, :time_steps]
                    elif seg.shape[1] < time_steps:
                        pad_width = time_steps - seg.shape[1]
                        seg = np.pad(seg, ((0, 0), (0, pad_width)), mode='edge')
                    if seg.shape[1] == time_steps:
                        subject_trials.append(seg)
                        subject_labels.append(label)

                print(f"  {os.path.basename(edf_file)}: done", flush=True)

            except Exception as e:
                print(f"  {os.path.basename(edf_file)}: 错误 - {e}")
                continue

        if subject_trials:
            all_data[subject_str] = (
                np.stack(subject_trials, axis=0).astype(np.float32),
                np.array(subject_labels, dtype=np.int64)
            )
            valid_ids.append(subject_str)
            elapsed = time.time() - t0
            print(f"被试 {sid}/{max_subject}: {len(subject_trials)} trials, "
                  f"耗时 {elapsed:.0f}s", flush=True)
        else:
            print(f"被试 {sid}: 无有效数据")

    # 保存为npz
    print(f"\n共 {len(valid_ids)} 个有效被试")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 用object数组保存不等长数据
    data_list = np.empty(len(valid_ids), dtype=object)
    label_list = np.empty(len(valid_ids), dtype=object)
    for i, sid in enumerate(valid_ids):
        data_list[i] = all_data[sid][0]
        label_list[i] = all_data[sid][1]

    np.savez_compressed(output_path,
                        data=data_list,
                        labels=label_list,
                        subject_ids=np.array(valid_ids),
                        channel_names=np.array(CHANNEL_NAMES_64),
                        input_channel_indices=np.array(INPUT_CHANNEL_INDICES_16))

    total_time = time.time() - t0
    print(f"\n保存完成! 文件: {output_path}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='E:/data/PhysioNetMI')
    parser.add_argument('--output', default='f:/DTTD-DDPM/data_cache/physionet_mi_preprocessed.npz')
    args = parser.parse_args()

    extract_all_subjects(args.data_path, args.output)
