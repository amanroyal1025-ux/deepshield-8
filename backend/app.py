import os, uuid, tempfile, logging
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils import predict_video

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

ALLOWED = {"mp4","avi","mov","mkv","webm"}
MODEL_PATH = "./deepshield.h5"

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded")
else:
    logging.warning("⚠️ No model found — running in demo mode")

def allowed(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

def demo_result():
    import random
    isFake = random.random() > 0.5
    fp = round(random.uniform(0.65,0.95) if isFake else random.uniform(0.05,0.35), 4)
    return {
        "verdict":          "FAKE" if isFake else "REAL",
        "confidence":       fp if isFake else round(1-fp, 4),
        "fake_probability": fp,
        "real_probability": round(1-fp, 4),
        "frames_analyzed":  20,
        "frames_with_face": 17,
        "artifacts": {
            "edge_noise_score":          round(random.uniform(0.3,0.8), 3),
            "color_inconsistency_score": round(random.uniform(0.2,0.7), 3),
            "boundary_blur_score":       round(random.uniform(0.4,0.9), 3),
        },
        "demo_mode": True
    }

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model_loaded": model is not None})

@app.route("/api/demo", methods=["GET"])
def demo():
    return jsonify(demo_result())

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error":"No video provided"}), 400
    f = request.files["video"]
    if not allowed(f.filename):
        return jsonify({"error":"Unsupported format"}), 400
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{secure_filename(f.filename)}")
    try:
        f.save(tmp)
        result = predict_video(tmp, model) if model else demo_result()
        result["filename"] = f.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
