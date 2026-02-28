"""
Celery task: process uploaded video through frame extraction + model inference.

Writes results to PostgreSQL (Video, Prediction, FrameResult).
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

print(f"DEBUG: Worker sys.path: {sys.path[:3]}")
print(f"DEBUG: Worker PYTHONPATH: {os.environ.get('PYTHONPATH')}")

from celery_app import celery

@celery.task(bind=True, max_retries=3)
def process_video(self, video_id: str):
    """
    Extract frames, run inference, save results to DB.

    1. Load Video from DB
    2. Call predict_video (from deploy/infer_video)
    3. Create Prediction and FrameResult records
    4. Heatmap URLs are placeholders; extend later with saliency maps.
    """
    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/deepfake_video")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(database_url, pool_pre_ping=True)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    video = None

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

        # Run inference (projects/deepfake-detector-video/src/deploy/infer_video)
        from src.deploy.infer_video import predict_video

        t0 = time.perf_counter()
        result = predict_video(video_path=storage_path, num_frames=16)
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

        # Frame results (heatmap_url as placeholder; extend later with saliency)
        for i, score in enumerate(result["frame_scores"]):
            fr = FrameResult(
                id=str(uuid.uuid4()),
                video_id=video_id,
                prediction_id=pred_id,
                frame_index=i,
                frame_score=round(score, 4),
                heatmap_url=f"/api/video/heatmaps/{video_id}/frame_{i}.png",
            )
            session.add(fr)

        video.status = "done"
        session.commit()
        return {"status": "done", "video_id": video_id, "video_score": result["video_score"]}

    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL ERROR processing video {video_id}: {error_msg}", file=sys.stderr)
        
        # Mark as error in DB using raw psycopg2 if needed
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

        # Don't retry on import or code errors
        _fatal_errors = ["ModuleNotFoundError", "ImportError", "NameError", "AttributeError"]
        if any(err in error_msg for err in _fatal_errors):
            return {"status": "error", "error": error_msg}
            
        raise self.retry(exc=e)
    finally:
        session.close()
