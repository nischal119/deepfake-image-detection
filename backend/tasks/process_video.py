"""
Celery task: process uploaded video through frame extraction + model inference.

Writes results to PostgreSQL (Video, Prediction, FrameResult).
Generates gradient-saliency heatmaps for each frame.
"""

import os
import sys
import time
import uuid
from pathlib import Path

# Ensure backend is in PYTHONPATH for all sub-processes
_backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _backend_root + os.pathsep + os.environ.get("PYTHONPATH", "")

if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

# Also add the project root for model inference
_repo_root = os.path.dirname(_backend_root)
_video_project = os.path.join(_repo_root, "projects", "deepfake-detector-video")
if _video_project not in sys.path:
    sys.path.insert(0, _video_project)

# Heatmaps directory lives alongside uploads
HEATMAP_BASE_DIR = os.path.join(_backend_root, "heatmaps")

from celery_app import celery


@celery.task(bind=True, max_retries=3)
def process_video(self, video_id: str):
    """
    Extract frames, run inference with saliency heatmaps, save results to DB.
    """
    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/deepfake_video")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(database_url, pool_pre_ping=True)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    try:
        from app.models import Video, Prediction, FrameResult

        video = session.query(Video).filter_by(id=video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        video.status = "processing"
        session.commit()

        storage_path = Path(video.storage_path)
        if not storage_path.exists():
            video.status = "error"
            video.error_message = f"Video file not found: {storage_path}"
            session.commit()
            return {"status": "error", "error": video.error_message}

        # Prepare heatmap output directory for this video
        heatmap_dir = os.path.join(HEATMAP_BASE_DIR, video_id)

        # Run inference with heatmap generation
        from src.deploy.infer_video import predict_video

        t0 = time.perf_counter()
        result = predict_video(
            video_path=storage_path,
            num_frames=16,
            heatmap_dir=heatmap_dir,
        )
        elapsed = time.perf_counter() - t0

        pred_id = str(uuid.uuid4())
        prediction = Prediction(
            id=pred_id,
            video_id=video_id,
            video_score=result["video_score"],
            prediction=result["prediction"],
            num_frames=result["num_frames"],
            processing_time_sec=round(elapsed, 2),
        )
        session.add(prediction)

        # Frame results with real heatmap URLs
        heatmap_paths = result.get("heatmap_paths", {})
        for i, score in enumerate(result["frame_scores"]):
            # Use real heatmap if generated, otherwise empty
            if i in heatmap_paths:
                heatmap_url = f"/api/video/heatmaps/{video_id}/frame_{i}.png"
            else:
                heatmap_url = ""

            fr = FrameResult(
                id=str(uuid.uuid4()),
                video_id=video_id,
                prediction_id=pred_id,
                frame_index=i,
                frame_score=round(score, 4),
                heatmap_url=heatmap_url,
            )
            session.add(fr)

        video.status = "done"
        session.commit()
        return {"status": "done", "video_id": video_id, "video_score": result["video_score"]}

    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL ERROR processing video {video_id}: {error_msg}", file=sys.stderr)

        # Mark as error in DB using raw psycopg2
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE videos SET status = 'error', error_message = %s, updated_at = NOW() WHERE id = %s",
                    (error_msg, video_id)
                )
            conn.commit()
            conn.close()
            print(f"Successfully marked {video_id} as error via raw SQL")
        except Exception as sql_e:
            print(f"FAILED raw SQL error update: {sql_e}", file=sys.stderr)

        # Don't retry on fatal errors
        _fatal_errors = [
            "ModuleNotFoundError", "ImportError", "NameError",
            "AttributeError", "FileNotFoundError", "RuntimeError",
            "Cannot open video", "No frames in video",
        ]
        if any(err in error_msg for err in _fatal_errors):
            print(f"FATAL ERROR for video {video_id}: {error_msg}. Not retrying.")
            return {"status": "error", "error": error_msg}

        raise self.retry(exc=e)
    finally:
        session.close()
