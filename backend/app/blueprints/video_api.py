"""
Video API blueprint.

Endpoints:
  POST /api/video/upload  -> upload video, enqueue Celery job
  GET  /api/video/status/<video_id>  -> processing status
  GET  /api/video/result/<video_id>  -> full result with frame heatmaps and video score
"""

import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.extensions import limiter
from app.models import Video, Prediction, FrameResult

video_api = Blueprint("video_api", __name__)


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config.get("ALLOWED_EXTENSIONS", {"mp4", "avi", "mov", "mkv"})


@video_api.route("/upload", methods=["POST"])
@limiter.limit("5 per minute")
def upload_video():
    """
    Accept video file, save to storage, enqueue Celery job.
    Rate-limited; enforces max file size.
    """

    if "video" not in request.files and "file" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files.get("video") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No video file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {current_app.config['ALLOWED_EXTENSIONS']}"}), 400

    # File size check (streaming)
    file.seek(0, 2)
    size_bytes = file.tell()
    file.seek(0)
    max_bytes = current_app.config["MAX_VIDEO_SIZE_MB"] * 1024 * 1024
    if size_bytes > max_bytes:
        return jsonify({
            "error": f"File too large. Max {current_app.config['MAX_VIDEO_SIZE_MB']}MB",
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }), 413

    video_id = str(uuid.uuid4())
    safe_name = secure_filename(file.filename) or "video.mp4"
    ext = safe_name.rsplit(".", 1)[-1] if "." in safe_name else "mp4"
    storage_name = f"{video_id}.{ext}"
    storage_path = Path(current_app.config["UPLOAD_DIR"]) / storage_name
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(str(storage_path))

    # Create DB record
    from flask import g
    session = g.db_session
    video = Video(
        id=video_id,
        filename=safe_name,
        storage_path=str(storage_path),
        file_size_bytes=size_bytes,
        status="pending",
    )
    session.add(video)
    session.commit()

    # Enqueue Celery job
    try:
        from tasks.process_video import process_video
        process_video.delay(video_id)
    except Exception as e:
        video.status = "error"
        video.error_message = str(e)
        session.commit()
        return jsonify({"error": f"Failed to enqueue job: {e}"}), 500

    return jsonify({
        "video_id": video_id,
        "status": "queued",
        "message": "Video uploaded; processing started.",
    }), 202


@video_api.route("/status/<video_id>", methods=["GET"])
def video_status(video_id: str):
    """Return processing status and partial results."""
    from flask import g
    session = g.db_session
    video = session.query(Video).filter_by(id=video_id).first()
    if not video:
        return jsonify({"error": "Video not found"}), 404

    resp = {
        "video_id": video_id,
        "status": video.status,
        "filename": video.filename,
        "created_at": video.created_at.isoformat() if video.created_at else None,
    }
    if video.error_message:
        resp["error_message"] = video.error_message
    if video.status == "done":
        pred = session.query(Prediction).filter_by(video_id=video_id).first()
        if pred:
            resp["video_score"] = pred.video_score
            resp["prediction"] = pred.prediction
    return jsonify(resp)


@video_api.route("/result/<video_id>", methods=["GET"])
def video_result(video_id: str):
    """Return full result: frame scores, heatmap URLs, video score."""
    from flask import g
    session = g.db_session
    video = session.query(Video).filter_by(id=video_id).first()
    if not video:
        return jsonify({"error": "Video not found"}), 404

    if video.status != "done":
        return jsonify({
            "video_id": video_id,
            "status": video.status,
            "message": "Processing not complete. Check /status/<video_id> first.",
        }), 202

    pred = session.query(Prediction).filter_by(video_id=video_id).first()
    if not pred:
        return jsonify({"error": "No prediction found"}), 404

    frames = session.query(FrameResult).filter_by(video_id=video_id).order_by(FrameResult.frame_index).all()

    return jsonify({
        "video_id": video_id,
        "status": "done",
        "filename": video.filename,
        "video_score": pred.video_score,
        "prediction": pred.prediction,
        "num_frames": pred.num_frames,
        "processing_time_sec": pred.processing_time_sec,
        "frame_scores": [{"frame_index": f.frame_index, "score": f.frame_score, "heatmap_url": f.heatmap_url} for f in frames],
    })


@video_api.route("/heatmaps/<video_id>/<filename>", methods=["GET"])
def serve_heatmap(video_id: str, filename: str):
    """Serve generated saliency heatmap images."""
    from flask import send_from_directory
    heatmap_dir = Path(current_app.root_path).parent / "heatmaps" / video_id
    if not heatmap_dir.exists():
        return jsonify({"error": "Heatmap not found"}), 404
    return send_from_directory(str(heatmap_dir), filename)
