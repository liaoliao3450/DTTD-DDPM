"""快速验证PhysioNet MI分类准确率"""
import numpy as np
import torch
import sys
sys.path.insert(0, '.')

from data.physionet_mi import INPUT_CHANNEL_INDICES_16
from experiments.physionet_mi_classification import EEGNetClassifier, train_classifier, evaluate_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 加载缓存数据
cached = np.load('paper_results/physionet_mi/physionet_mi_all_subjects.npz', allow_pickle=True)
sdm = dict(cached['subject_data_map'].item())

# Cross-subject测试（2折快速验证）
print('\n=== Cross-subject 测试（2折快速） ===')
all_ids = list(sdm.keys())
np.random.seed(42)
np.random.shuffle(all_ids)
train_ids = all_ids[:55]
test_ids = all_ids[55:]

train_data_list, train_labels_list = [], []
for sid in train_ids:
    dd, ll = sdm[sid]
    train_data_list.append(dd)
    train_labels_list.append(ll)
train_data_64 = np.concatenate(train_data_list, axis=0).astype(np.float32)
train_labels = np.concatenate(train_labels_list, axis=0)

test_data_list, test_labels_list = [], []
for sid in test_ids:
    dd, ll = sdm[sid]
    test_data_list.append(dd)
    test_labels_list.append(ll)
test_data_64 = np.concatenate(test_data_list, axis=0).astype(np.float32)
test_labels = np.concatenate(test_labels_list, axis=0)

print(f'训练集: {len(train_data_64)}, 测试集: {len(test_data_64)}')

# 64ch cross-subject
clf_cs, m_cs, s_cs = train_classifier(train_data_64, train_labels, 64, device, epochs=300)
result_cs = evaluate_classifier(clf_cs, test_data_64, test_labels, device, m_cs, s_cs)
print(f'Cross-subject 64ch: acc={result_cs["accuracy"]:.4f}, kappa={result_cs["kappa"]:.4f}')

# 16ch cross-subject
train_data_16 = train_data_64[:, INPUT_CHANNEL_INDICES_16, :]
test_data_16 = test_data_64[:, INPUT_CHANNEL_INDICES_16, :]
clf_cs16, m_cs16, s_cs16 = train_classifier(train_data_16, train_labels, 16, device, epochs=300)
result_cs16 = evaluate_classifier(clf_cs16, test_data_16, test_labels, device, m_cs16, s_cs16)
print(f'Cross-subject 16ch: acc={result_cs16["accuracy"]:.4f}, kappa={result_cs16["kappa"]:.4f}')
