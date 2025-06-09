import cv2
import mediapipe as mp
import os
import numpy as np
import glob
import re
from collections import defaultdict

ROOT_DIRECTORY = "../IEMOCAP"
OUTPUT_LANDMARK_DIR = "landmark_features"
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
mp_face_mesh = mp.solutions.face_mesh

def parse_emo_evaluation_file(filepath):
    utterance_times = {}
    pattern = re.compile(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\w+)\t\w+\t\[.*\]$')
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                start_time = float(match.group(1)) * 1000
                end_time = float(match.group(2)) * 1000
                utterance_id = match.group(3)
                utterance_times[utterance_id] = {'start': start_time, 'end': end_time}
    return utterance_times

def process_and_extract_landmarks(video_path, utterance_times, output_dir):
    landmarks_per_utterance_F = defaultdict(list)
    landmarks_per_utterance_M = defaultdict(list)
    face_detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=2,
            refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_utterance_id = None
            for uid, times in utterance_times.items():
                if times['start'] <= current_time_ms <= times['end']:
                    current_utterance_id = uid
                    break
            if not current_utterance_id:
                continue
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_detector_net.setInput(blob)
            detections = face_detector_net.forward()
            detected_faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    detected_faces.append({'box': (startX, startY, endX, endY)})
            if not detected_faces:
                continue
            landmarks_f, landmarks_m = None, None
            face_crops_for_mp = []
            for face in detected_faces:
                (startX, startY, endX, endY) = face['box']
                face_crop = frame[startY:endY, startX:endX]
                if face_crop.size == 0: continue
                rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                results_mp = face_mesh.process(rgb_face_crop)
                if results_mp.multi_face_landmarks:
                    landmarks = results_mp.multi_face_landmarks[0]
                    landmark_vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                    avg_x = np.mean([lm.x for lm in landmarks.landmark]) * (endX - startX) + startX
                    face_crops_for_mp.append({'landmarks': landmark_vector, 'avg_x': avg_x})
            if len(face_crops_for_mp) == 2:
                if face_crops_for_mp[0]['avg_x'] < face_crops_for_mp[1]['avg_x']:
                    landmarks_f = face_crops_for_mp[0]['landmarks']
                    landmarks_m = face_crops_for_mp[1]['landmarks']
                else:
                    landmarks_f = face_crops_for_mp[1]['landmarks']
                    landmarks_m = face_crops_for_mp[0]['landmarks']
            elif len(face_crops_for_mp) == 1:
                if current_utterance_id.split('_')[-1].startswith('F'):
                    landmarks_f = face_crops_for_mp[0]['landmarks']
                elif current_utterance_id.split('_')[-1].startswith('M'):
                    landmarks_m = face_crops_for_mp[0]['landmarks']

            if landmarks_f is not None:
                landmarks_per_utterance_F[current_utterance_id].append(landmarks_f)
            if landmarks_m is not None:
                landmarks_per_utterance_M[current_utterance_id].append(landmarks_m)
        cap.release()
    for uid, vectors in landmarks_per_utterance_F.items():
        if vectors:
            avg_vector = np.mean(np.array(vectors), axis=0)
            np.save(os.path.join(output_dir, f"{uid}.npy"), avg_vector)
    for uid, vectors in landmarks_per_utterance_M.items():
        if vectors:
            avg_vector = np.mean(np.array(vectors), axis=0)
            np.save(os.path.join(output_dir, f"{uid}.npy"), avg_vector)
    print(f"{len(landmarks_per_utterance_F) + len(landmarks_per_utterance_M)}")


def main():
    if not all(os.path.exists(p) for p in [PROTOTXT_PATH, MODEL_PATH]):
        return

    os.makedirs(OUTPUT_LANDMARK_DIR, exist_ok=True)
    emo_files_pattern = os.path.join(ROOT_DIRECTORY, "Session*", "dialog", "EmoEvaluation", "*.txt")
    emo_files = glob.glob(emo_files_pattern)
    for emo_file in emo_files:
        utterance_times = parse_emo_evaluation_file(emo_file)
        session_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(emo_file))))
        video_filename = os.path.splitext(os.path.basename(emo_file))[0] + ".avi"
        video_path = os.path.join(ROOT_DIRECTORY, session_name, "dialog", "avi", "DivX", video_filename)

        if os.path.exists(video_path):
            process_and_extract_landmarks(video_path, utterance_times, OUTPUT_LANDMARK_DIR)
        else:
            print(f"{video_path}")

if __name__ == '__main__':
    main()
