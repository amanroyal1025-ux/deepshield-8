import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    margin = int(0.15 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    return frame[y1:y2, x1:x2]

def extract_frames(video_path, num_frames=20):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(total // num_frames, 1)
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(frames) < num_frames:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def preprocess(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normed  = rgb.astype("float32") / 255.0
    return np.expand_dims(normed, axis=0)

def analyze_artifacts(frame):
    face = detect_face(frame)
    if face is None or face.size == 0:
        return {}
    face_r = cv2.resize(face, (128, 128))
    gray   = cv2.cvtColor(face_r, cv2.COLOR_BGR2GRAY)
    lap_var      = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_score   = min(lap_var / 1000.0, 1.0)
    b, g, r      = cv2.split(face_r.astype(float))
    color_score  = min(np.std([b.mean(), g.mean(), r.mean()]) / 30.0, 1.0)
    border = np.concatenate([gray[:10,:].ravel(), gray[-10:,:].ravel()])
    center = gray[30:90, 30:90].ravel()
    blur_score   = 1.0 - min(abs(border.std() - center.std()) / 50.0, 1.0)
    return {
        "edge_noise_score":          round(float(edge_score),  3),
        "color_inconsistency_score": round(float(color_score), 3),
        "boundary_blur_score":       round(float(blur_score),  3),
    }

def predict_video(video_path, model, num_frames=20):
    frames      = extract_frames(video_path, num_frames)
    predictions = []
    all_artifacts = []
    faces_found = 0
    for frame in frames:
        face = detect_face(frame)
        if face is None:
            continue
        faces_found += 1
        prob_real = float(model.predict(preprocess(face), verbose=0)[0][0])
        predictions.append(prob_real)
        all_artifacts.append(analyze_artifacts(frame))
    if not predictions:
        return {
            "verdict": "UNKNOWN", "confidence": 0.0,
            "fake_probability": 0.5, "real_probability": 0.5,
            "frames_analyzed": len(frames), "frames_with_face": 0,
            "artifacts": {}, "error": "No faces detected"
        }
    avg_real = np.mean(predictions)
    avg_fake = 1.0 - avg_real
    verdict  = "FAKE" if avg_fake > 0.5 else "REAL"
    conf     = avg_fake if verdict == "FAKE" else avg_real
    agg = {}
    for key in ["edge_noise_score","color_inconsistency_score","boundary_blur_score"]:
        vals = [a[key] for a in all_artifacts if key in a]
        if vals:
            agg[key] = round(float(np.mean(vals)), 3)
    return {
        "verdict":          verdict,
        "confidence":       round(float(conf),     4),
        "fake_probability": round(float(avg_fake), 4),
        "real_probability": round(float(avg_real), 4),
        "frames_analyzed":  len(frames),
        "frames_with_face": faces_found,
        "artifacts":        agg,
    }
