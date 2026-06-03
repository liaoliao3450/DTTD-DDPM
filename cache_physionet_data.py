import os
import sys
import time
import numpy as np
import mne

DATA_PATH = 'E:/data/PhysioNetMI'
CACHE_PATH = 'paper_results/physionet_mi/physionet_mi_cache.npz'
MI4C_RUNS = [4, 8, 12]
TIME_STEPS = 640
LABEL_MAP = {(4, 1): 0, (4, 2): 1, (8, 1): 2, (8, 2): 3,
             (12, 1): 0, (12, 2): 1}

all_segments = []
all_labels = []
loaded_subjects = 0

for sid in range(1, 110):
    subj_dir = os.path.join(DATA_PATH, f"S{sid:03d}")
    if not os.path.isdir(subj_dir):
        continue

    subj_segments = []
    subj_labels = []
    for run in MI4C_RUNS:
        edf_path = os.path.join(subj_dir, f"S{sid:03d}R{run:02d}.edf")
        if not os.path.exists(edf_path):
            continue
        try:
            t0 = time.time()
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, exclude="bads")
            raw.pick(eeg_picks)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
            sfreq = raw.info['sfreq']
            data_arr = raw.get_data().astype(np.float32)
            events = mne.find_events(raw, stim_channel='auto', verbose=False)
            for ev in events:
                if ev[2] not in (1, 2):
                    continue
                label = LABEL_MAP.get((run, ev[2]))
                if label is None:
                    continue
                start = ev[0]
                stop = start + int(4.0 * sfreq)
                if stop > data_arr.shape[1]:
                    continue
                seg = data_arr[:, start:stop]
                if seg.shape[1] > TIME_STEPS:
                    seg = seg[:, :TIME_STEPS]
                elif seg.shape[1] < TIME_STEPS:
                    seg = np.pad(seg, ((0, 0), (0, TIME_STEPS - seg.shape[1])), mode='edge')
                subj_segments.append(seg)
                subj_labels.append(label)
            elapsed = time.time() - t0
            print(f"  Subject {sid}, Run {run}: {elapsed:.1f}s, {len(subj_segments)} segments so far", flush=True)
        except Exception as e:
            print(f"  Subject {sid}, Run {run}: ERROR - {e}", flush=True)
            continue

    if subj_segments:
        all_segments.extend(subj_segments)
        all_labels.extend(subj_labels)
        loaded_subjects += 1
        print(f"Subject {sid} done. Total: {len(all_segments)} segments from {loaded_subjects} subjects", flush=True)

    if len(all_segments) >= 300:
        print(f"Reached 300+ segments, stopping", flush=True)
        break

all_data = np.array(all_segments, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)

print(f"\nTotal: {len(all_data)} segments, shape: {all_data.shape}, labels shape: {all_labels.shape}", flush=True)

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
np.savez(CACHE_PATH, data=all_data, labels=all_labels)
print(f"Saved to {CACHE_PATH}", flush=True)
