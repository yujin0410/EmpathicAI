import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import defaultdict
import numpy as np
import os
import re
from argparse import Namespace
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from model_4modal import GraphSmile4Modal

IEMOCAP_ROOT_PATH = "../IEMOCAP/"
PREPROCESSED_FEATURE_PATH = "iemocap_multi_features.pkl"
LANDMARK_FEATURE_DIR = "../landmark_features"
EMOTION_LABELS = ['neu', 'hap', 'sad', 'ang', 'sur', 'fea', 'dis', 'fru', 'exc', 'oth']

def load_iemocap_metadata():
    print(f"IEMOCAP {IEMOCAP_ROOT_PATH}")
    emotion_map = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6, 'fru': 7, 'exc': 8, 'oth': 9}
    all_utterances = []
    pattern = re.compile(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t(\w+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]$')
    for i in range(1, 6):
        session_name = f"Session{i}"
        emo_eval_dir = os.path.join(IEMOCAP_ROOT_PATH, session_name, 'dialog/EmoEvaluation')
        if not os.path.isdir(emo_eval_dir): continue
        for filename in os.listdir(emo_eval_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(emo_eval_dir, filename)
                with open(filepath, 'r', encoding='latin-1') as f:
                    for line in f:
                        match = pattern.match(line.strip())
                        if match:
                            utterance_id, emotion = match.group(3), match.group(4)
                            if emotion not in emotion_map: continue
                            all_utterances.append({'id': utterance_id, 'emotion_label': emotion_map[emotion]})
    print(f"총 {len(all_utterances)}개의 발화 메타데이터를 로드했습니다.")
    return all_utterances

class IEMOCAP_4Modal_Dataset(Dataset):
    def __init__(self, preprocessed_feature_path, landmark_dir, mode='train'):
        self.landmark_dir = landmark_dir
        self.mode = mode
        print(f"[{self.mode.upper()}] 전처리된 3모달 특징 파일 로딩: {preprocessed_feature_path}")
        with open(preprocessed_feature_path, 'rb') as f: data_list = pk.load(f)
        self.feature_text = data_list[3]
        self.feature_audio = data_list[4]
        self.feature_visual = data_list[5]
        train_vids, test_vids, video_ids_dict = data_list[6], data_list[7], data_list[0]
        train_uids = {uid for vid in train_vids for uid in video_ids_dict.get(vid, [])}
        test_uids = {uid for vid in test_vids for uid in video_ids_dict.get(vid, [])}
        self.target_ids = train_uids if mode == 'train' else test_uids
        all_utterances = load_iemocap_metadata()
        self.samples = self.create_dyadic_pairs(all_utterances)
        if not self.samples: print(f"[{self.mode.upper()}] 생성된 대화 쌍이 없습니다.")

    def create_dyadic_pairs(self, all_utterances):
        dyadic_pairs = []
        filtered_utterances = [u for u in all_utterances if u['id'] in self.target_ids]
        utterance_dict = {u['id']: u for u in filtered_utterances}
        sorted_ids = sorted(utterance_dict.keys())
        for i in range(len(sorted_ids) - 1):
            current_id, next_id = sorted_ids[i], sorted_ids[i+1]
            if '_'.join(current_id.split('_')[:-1]) != '_'.join(next_id.split('_')[:-1]): continue
            if current_id.split('_')[-1].startswith('F') and utterance_dict.get(next_id, {}).get('id', ' ').split('_')[-1].startswith('M'):
                dyadic_pairs.append((utterance_dict[current_id], utterance_dict[next_id]))
        print(f"총 {len(dyadic_pairs)}개의 (F->M) {self.mode} 대화 쌍을 찾았습니다.")
        return dyadic_pairs

    def __getitem__(self, index):
        f_utterance_data, m_utterance_data = self.samples[index]
        f_features = {'text': self.load_feature(f_utterance_data['id'], 'text'),
                      'audio': self.load_feature(f_utterance_data['id'], 'audio'),
                      'face': self.load_feature(f_utterance_data['id'], 'face'),
                      'landmark': self.load_feature(f_utterance_data['id'], 'landmark')}
        m_emotion_label = m_utterance_data['emotion_label']
        m_landmark_feature = self.load_feature(m_utterance_data['id'], 'landmark')
        return f_features, (torch.tensor(m_emotion_label, dtype=torch.long), m_landmark_feature)

    def __len__(self): return len(self.samples)

    def load_feature(self, utterance_id, modality):
        try:
            if modality == 'text': feature = self.feature_text[utterance_id]
            elif modality == 'audio': feature = self.feature_audio[utterance_id]
            elif modality == 'face': feature = self.feature_visual[utterance_id]
            elif modality == 'landmark':
                session_folder = utterance_id.split('_')[0]
                video_folder = '_'.join(utterance_id.split('_')[:-1])
                feature_path = os.path.join(self.landmark_dir, session_folder, video_folder, f"{utterance_id}.npy")
                feature = np.load(feature_path)
            else: return torch.zeros(1)
            return torch.from_numpy(feature).float()
        except (KeyError, FileNotFoundError):
            dims = {'text': 1024, 'audio': 342, 'face': 1582, 'landmark': 1434}
            return torch.zeros(dims.get(modality, 1))

def evaluate_model(model, test_loader, device, criterion_emotion, criterion_landmark):
    model.eval()
    all_emo_preds, all_emo_labels = [], []
    all_lm_preds, all_lm_labels = [], []
    total_emo_loss, total_lm_loss = 0.0, 0.0
    with torch.no_grad():
        for f_features, (m_emo_labels, m_lm_features) in test_loader:
            text_feat, audio_feat = f_features['text'].to(device), f_features['audio'].to(device)
            face_feat, landmark_feat = f_features['face'].to(device), f_features['landmark'].to(device)
            m_emo_labels_dev, m_lm_features_dev = m_emo_labels.to(device), m_lm_features.to(device)
            
            emo_preds_logits, lm_preds = model(text_feat, audio_feat, face_feat, landmark_feat)
            
            total_emo_loss += criterion_emotion(emo_preds_logits, m_emo_labels_dev).item()
            total_lm_loss += criterion_landmark(lm_preds, m_lm_features_dev).item()
            
            _, emo_preds = torch.max(emo_preds_logits, 1)
            all_emo_preds.extend(emo_preds.cpu().numpy())
            all_emo_labels.extend(m_emo_labels.cpu().numpy())
            all_lm_preds.extend(lm_preds.cpu().numpy())
            all_lm_labels.extend(m_lm_features.cpu().numpy())

    print("\n--- [감정 분류 성능] ---")
    accuracy = accuracy_score(all_emo_labels, all_emo_preds)
    unique_labels = sorted(list(set(all_emo_labels)))
    target_names = [EMOTION_LABELS[i] for i in unique_labels if i < len(EMOTION_LABELS)]
    report = classification_report(all_emo_labels, all_emo_preds, labels=unique_labels, target_names=target_names, zero_division=0)
    print(f"테스트 분류 Loss (CrossEntropy): {total_emo_loss / len(test_loader):.4f}")
    print(f"테스트 정확도: {accuracy*100:.2f}%")
    print("\n[상세]")
    print(report)
    cm = confusion_matrix(all_emo_labels, all_emo_preds, labels=unique_labels)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Emotion Classification Confusion Matrix'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.savefig('emotion_confusion_matrix.png')
    
    mse = mean_squared_error(np.array(all_lm_labels), np.array(all_lm_preds))
    print(f"MSE from criterion: {total_lm_loss / len(test_loader):.4f}")
    print(f"MSE (sklearn): {mse:.4f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        train_dataset = IEMOCAP_4Modal_Dataset(PREPROCESSED_FEATURE_PATH, LANDMARK_FEATURE_DIR, mode='train')
        test_dataset = IEMOCAP_4Modal_Dataset(PREPROCESSED_FEATURE_PATH, LANDMARK_FEATURE_DIR, mode='test')
    except FileNotFoundError:
        exit()

    if len(train_dataset) > 0 and len(test_dataset) > 0:
        train_labels = [s[1][0].item() for s in train_dataset]
        class_counts = np.bincount(train_labels, minlength=len(EMOTION_LABELS))
        class_weights = 1. / (class_counts + 1e-6)
        sample_weights = np.array([class_weights[label] for label in train_labels])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        try:
            fake_args = Namespace(hidden_dim=512, drop=0.3)
            num_emotion_classes = 10
            embedding_dims = [1024, 342, 1582, 1434]
            model = GraphSmile4Modal(args=fake_args, embedding_dims=embedding_dims, n_classes_emo=num_emotion_classes)
            model.to(device)
            print("ok.")
        except Exception as e:
            print(f"model fail: {e}")
            exit()
            
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion_emotion = nn.CrossEntropyLoss()
        criterion_landmark = nn.MSELoss()
        num_epochs = 50
        landmark_loss_weight = 0.5

        model.train()
        for epoch in range(num_epochs):
            total_loss_epoch, total_emo_loss_epoch, total_lm_loss_epoch = 0.0, 0.0, 0.0
            for i, (f_features, (m_emo_labels, m_lm_features)) in enumerate(train_loader):
                text_feat, audio_feat = f_features['text'].to(device), f_features['audio'].to(device)
                face_feat, landmark_feat = f_features['face'].to(device), f_features['landmark'].to(device)
                m_emo_labels_dev, m_lm_features_dev = m_emo_labels.to(device), m_lm_features.to(device)
                optimizer.zero_grad()
                emo_preds_logits, lm_preds = model(text_feat, audio_feat, face_feat, landmark_feat)
                loss_emo = criterion_emotion(emo_preds_logits, m_emo_labels_dev)
                loss_lm = criterion_landmark(lm_preds, m_lm_features_dev)
                total_loss = loss_emo + landmark_loss_weight * loss_lm
                total_loss.backward()
                optimizer.step()
                total_loss_epoch += total_loss.item()
                total_emo_loss_epoch += loss_emo.item()
                total_lm_loss_epoch += loss_lm.item()
            
            avg_loss = total_loss_epoch / len(train_loader)
            avg_emo_loss = total_emo_loss_epoch / len(train_loader)
            avg_lm_loss = total_lm_loss_epoch / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Combined Loss: {avg_loss:.4f} (Emo: {avg_emo_loss:.4f}, LM: {avg_lm_loss:.4f})")
            
        print("--- perfect---")
        evaluate_model(model, test_loader, device, criterion_emotion, criterion_landmark)
    else:
        print("fail.")
