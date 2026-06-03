import os
import re
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
import time

DATA_PATH = 'E:/data/PhysioNetMI'
CACHE_PATH = 'paper_results/physionet_mi/physionet_mi_cache.npz'
MI4C_RUNS = [4, 8, 12]
TIME_STEPS = 640
LABEL_MAP = {(4, 'T1'): 0, (4, 'T2'): 1, (8, 'T1'): 2, (8, 'T2'): 3,
             (12, 'T1'): 0, (12, 'T2'): 1}

FS_TARGET = 160
L_FREQ = 4.0
H_FREQ = 30.0
NOTCH_FREQ = 50.0

CHANNEL_NAMES_64 = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2', 'Iz'
]


def parse_edf_annotations(ann_bytes):
    events = []
    text = ann_bytes.decode('latin-1', errors='replace')
    pattern = r'([+-]?\d+\.?\d*)\x15(\d+\.?\d*)\x14([^\x14\x00]+)\x14'
    matches = re.findall(pattern, text)
    for onset_str, dur_str, desc in matches:
        try:
            onset = float(onset_str)
            desc = desc.strip()
            if desc in ('T1', 'T2'):
                events.append((onset, desc))
        except ValueError:
            continue
    return events


def apply_preprocessing(eeg_data, sfreq):
    if sfreq != FS_TARGET:
        from scipy.signal import resample_poly
        gcd = np.gcd(int(FS_TARGET), int(sfreq))
        up = int(FS_TARGET) // gcd
        down = int(sfreq) // gcd
        eeg_data = resample_poly(eeg_data, up, down, axis=1)

    sfreq = FS_TARGET

    if NOTCH_FREQ is not None and NOTCH_FREQ < sfreq / 2:
        w0 = NOTCH_FREQ / (sfreq / 2)
        Q = 30.0
        b, a = iirnotch(w0, Q)
        eeg_data = sosfilt(np.array([b[0], b[1], b[2], a[0], a[1], a[2]]).reshape(1, -1), eeg_data, axis=1)

    if L_FREQ is not None and H_FREQ is not None:
        nyq = sfreq / 2.0
        low = L_FREQ / nyq
        high = H_FREQ / nyq
        sos = butter(5, [low, high], btype='band', output='sos')
        eeg_data = sosfiltfilt(sos, eeg_data, axis=1)

    eeg_data = eeg_data - eeg_data.mean(axis=0, keepdims=True)

    return eeg_data.astype(np.float32)


def read_edf_with_events(edf_path):
    with open(edf_path, 'rb') as f:
        header = f.read(256)
        n_data_records = int(header[236:244].decode('ascii').strip())
        duration = float(header[244:252].decode('ascii').strip())
        n_signals = int(header[252:256].decode('ascii').strip())

        labels = [f.read(16).decode('ascii', errors='ignore').strip() for _ in range(n_signals)]
        transducer = [f.read(80).decode('ascii', errors='ignore').strip() for _ in range(n_signals)]
        phys_dim = [f.read(8).decode('ascii', errors='ignore').strip() for _ in range(n_signals)]
        phys_min = [float(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        phys_max = [float(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        dig_min = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        dig_max = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        prefilter = [f.read(80).decode('ascii', errors='ignore').strip() for _ in range(n_signals)]
        n_samp = [int(f.read(8).decode('ascii').strip()) for _ in range(n_signals)]
        reserved = [f.read(32) for _ in range(n_signals)]

        # 识别EEG通道和注释通道
        ann_index = -1
        eeg_indices = []
        for i, label in enumerate(labels):
            if label == 'EDF Annotations':
                ann_index = i
            elif i != ann_index:
                # PhysioNet MI的phys_dim可能是'BCI2000'或空，不能只靠'uV'判断
                # 排除注释通道，其余都是EEG
                eeg_indices.append(i)

        n_eeg = len(eeg_indices)
        samples_per_record = n_samp[eeg_indices[0]]
        total_samples = n_data_records * samples_per_record

        # 正确读取：按通道组织数据，避免reshape导致通道数据混乱
        eeg_data = np.zeros((n_eeg, total_samples), dtype=np.float32)
        ann_bytes = b''

        for rec in range(n_data_records):
            ch_idx = 0
            for i in range(n_signals):
                ns = n_samp[i]
                raw = f.read(ns * 2)
                if i == ann_index:
                    ann_bytes += raw
                    continue
                if i not in eeg_indices:
                    continue
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
                pmin, pmax = phys_min[i], phys_max[i]
                dmin, dmax = dig_min[i], dig_max[i]
                if dmax != dmin:
                    samples = (samples - dmin) / (dmax - dmin) * (pmax - pmin) + pmin
                start = rec * samples_per_record
                eeg_data[ch_idx, start:start + ns] = samples.astype(np.float32)
                ch_idx += 1

    sfreq = samples_per_record / duration if duration > 0 else 160.0

    events = parse_edf_annotations(ann_bytes) if ann_index >= 0 else []

    edf_ch_names = [labels[i].rstrip('.').upper() for i in eeg_indices]
    reorder_idx = []
    for std_ch in CHANNEL_NAMES_64:
        std_upper = std_ch.upper()
        if std_upper in edf_ch_names:
            reorder_idx.append(edf_ch_names.index(std_upper))
    if len(reorder_idx) == len(eeg_indices):
        eeg_data = eeg_data[reorder_idx, :]

    return eeg_data, eeg_indices, events, sfreq, labels


all_segments = []
all_labels = []
loaded_subjects = 0
t_start = time.time()

for sid in range(1, 110):
    subj_dir = os.path.join(DATA_PATH, f"S{sid:03d}")
    if not os.path.isdir(subj_dir):
        continue

    for run in MI4C_RUNS:
        edf_path = os.path.join(subj_dir, f"S{sid:03d}R{run:02d}.edf")
        if not os.path.exists(edf_path):
            continue
        try:
            t0 = time.time()
            eeg_data, eeg_indices, events, sfreq, ch_labels = read_edf_with_events(edf_path)

            eeg_data = apply_preprocessing(eeg_data, sfreq)
            sfreq = FS_TARGET

            n_t1 = sum(1 for _, d in events if d == 'T1')
            n_t2 = sum(1 for _, d in events if d == 'T2')

            for onset, desc in events:
                label = LABEL_MAP.get((run, desc))
                if label is None:
                    continue
                start = int(onset * sfreq)
                stop = start + int(4.0 * sfreq)
                if start < 0 or stop > eeg_data.shape[1]:
                    continue
                seg = eeg_data[:, start:stop]
                if seg.shape[1] > TIME_STEPS:
                    seg = seg[:, :TIME_STEPS]
                elif seg.shape[1] < TIME_STEPS:
                    seg = np.pad(seg, ((0, 0), (0, TIME_STEPS - seg.shape[1])), mode='edge')
                all_segments.append(seg)
                all_labels.append(label)

            elapsed = time.time() - t0
            print(f"  S{sid:03d}R{run:02d}: {elapsed:.1f}s, {len(eeg_indices)} EEG ch, T1={n_t1} T2={n_t2}, total segs: {len(all_segments)}", flush=True)
        except Exception as e:
            print(f"  S{sid:03d}R{run:02d}: ERROR - {e}", flush=True)
            continue

    if all_segments:
        loaded_subjects += 1

    if len(all_segments) >= 300:
        break

elapsed_total = time.time() - t_start
all_data = np.array(all_segments, dtype=np.float32)
all_labels_arr = np.array(all_labels, dtype=np.int64)

print(f"\nTotal: {len(all_data)} segments, shape: {all_data.shape}, elapsed: {elapsed_total:.1f}s", flush=True)
print(f"Data range: [{all_data.min():.4f}, {all_data.max():.4f}]", flush=True)
print(f"Data mean: {all_data.mean():.4f}, std: {all_data.std():.4f}", flush=True)

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
np.savez(CACHE_PATH, data=all_data, labels=all_labels_arr)
print(f"Saved to {CACHE_PATH}", flush=True)
