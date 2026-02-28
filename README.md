## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd "DeepFake Detector"
```

### 2. Backend Setup

```bash
cd frontend
pnpm install
```

### 3. Database Setup

```bash
#  generate Prisma client
pnpm prisma:generate

# run migrations
pnpm prisma:migrate

# Optional: Open Prisma Studio to view database
pnpm db:studio
```

### 4. Python Environment Setup

```bash
#navigate to root
cd ..

# virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Environment Configuration

Create `.env` file in the `frontend` directory. For the video detection flow (Flask backend):

```
FLASK_VIDEO_API_URL=http://localhost:5000
```

## Usage

### Starting the Development Server

```bash
cd frontend
pnpm dev
```

The application will be available at `http://localhost:3000`

## Technology Stack

### Frontend

- Framework : Next.js 14 with App Router
- Language : TypeScript
- UI Components : Radix UI, shadcn/ui
- Styling : Tailwind CSS 4
- Data Fetching : SWR (stale-while-revalidate)
- Form Validation : React Hook Form with Zod
- Visualization : Recharts

### Backend

- Runtime : Node.js
- API : Next.js API Routes (RESTful)
- ORM : Prisma
- Database : SQLite (development), PostgreSQL-ready
- File Processing : Native Node.js streams
- Background Jobs : Custom polling worker

### Machine Learning

- Framework : PyTorch 1.9+
- Model Architecture : Vision Transformer (ViT)
- Base Model : google/vit-base-patch16-224
- Fine-tuning : Custom checkpoint trained on deepfake datasets
- Library : HuggingFace Transformers 4.20+
- Explainability : Gradient-based saliency mapping
- Image Processing : Pillow, NumPy

## Prerequisites

### System Requirements

- Node.js 18+ and pnpm
- Python 3.8+ with pip
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for model checkpoints

### Development Tools

- Git for version control
- SQLite3 (included with most systems)
- Text editor or IDE (VS Code recommended)

### Uploading an Image

1. Navigate to the detection page
2. Click the upload area or drag and drop an image
3. Supported formats: JPEG, PNG, WebP
4. Maximum file size: 10MB (configurable)

### Viewing Results

After upload, the system will:

1. Queue the analysis job
2. Process the image through the ML model
3. Generate explainability visualizations
4. Display results with confidence score
5. Show heatmap overlay and detected artifacts

### Accessing History

Navigate to the history page to view:

- All previous detection jobs
- Timestamps and file names
- Quick result summaries
- Links to detailed reports

## API Documentation

### Endpoints

#### POST /api/predict

Create new detection job

Request :

```
Content-Type: multipart/form-data

file: <File>
type: "image" | "video"
```

Response :

```json
{
  "jobId": "cmg6mj5om0000xvvmno187fow",
  "status": "queued",
  "estimatedSeconds": 12
}
```

#### GET /api/jobs/[jobId]

Get job status and progress

Response :

```json
{
  "id": "cmg6mj5om0000xvvmno187fow",
  "status": "analyzing",
  "progress": 0.45,
  "step": "Analyzing",
  "createdAt": "2025-10-12T10:30:00Z"
}
```

#### GET /api/jobs/[jobId]/result

Get detection results (available when status is "complete")

Response :

```json
{
  "score": 0.87,
  "verdict": "likely_fake",
  "probs": {
    "real": 0.13,
    "fake": 0.87
  },
  "heatmap": "/heatmaps/cmg6mj5om0000xvvmno187fow.png",
  "artifacts": [
    {
      "name": "High frequency energy",
      "severity": "medium"
    }
  ],
  "metadata": {...},
  "timeline": [...]
}
```

#### GET /api/history

List all detection jobs

Query Parameters :

- `limit`: Number of results (default: 50)
- `offset`: Pagination offset (default: 0)

Response :

```json
{
  "jobs": [...],
  "total": 142,
  "hasMore": true
}
```

#### GET /api/model

Get model information

Response :

```json
{
  "name": "ViT DeepFake Detector",
  "version": "1.0",
  "architecture": "google/vit-base-patch16-224",
  "checkpoint": "checkpoint-14282",
  "accuracy": 0.94
}
```

## Database Schema

### DetectionJob

Stores information about analysis jobs

- `id`: Unique identifier (CUID)
- `createdAt`: Job creation timestamp
- `updatedAt`: Last update timestamp
- `status`: Job status (queued, analyzing, postprocessing, complete, error)
- `progress`: Completion percentage (0-1)
- `step`: Current processing step description
- `type`: Media type (image, video)
- `fileName`: Original file name
- `filePath`: Stored file path
- `errorMessage`: Error details if failed

### DetectionResult

Stores detection analysis results

- `id`: Unique identifier (CUID)
- `createdAt`: Result creation timestamp
- `score`: Deepfake probability (0-1)
- `verdict`: Classification (likely_fake, likely_real, inconclusive)
- `inputType`: Media type
- `inputWidth`: Image width in pixels
- `inputHeight`: Image height in pixels
- `durationSec`: Processing duration
- `heatmap`: Saliency heatmap URL
- `artifacts`: JSON array of detected artifacts
- `metadata`: JSON object with additional information
- `timeline`: JSON array for temporal analysis
- `reportUrl`: Full report URL
- `jobId`: Foreign key to DetectionJob

### JobEvent

Tracks job processing events

- `id`: Unique identifier (CUID)
- `createdAt`: Event timestamp
- `type`: Event type (progress, step, warning, complete, error)
- `message`: Event description
- `progress`: Progress value at time of event
- `jobId`: Foreign key to DetectionJob

## Model Information

### Vision Transformer Architecture

The detection system uses a fine-tuned Vision Transformer (ViT) model:

- Base Model : google/vit-base-patch16-224
- Input Size : 224x224 pixels
- Patch Size : 16x16 pixels
- Architecture : 12 transformer layers, 768 hidden dimensions
- Parameters : Approximately 86 million
- Fine-tuning Dataset : Deepfake vs Real image detection dataset

### Training Details

- Framework : PyTorch with HuggingFace Transformers
- Optimizer : AdamW
- Learning Rate : Custom schedule with warmup
- Augmentation : Random flips, color jitter, rotation
- Class Balancing : Imbalanced-learn sampling
- Checkpoint : Saved at step 14282 (best validation performance)

### Inference Features

- Temperature Scaling : Calibrates confidence scores (default: 1.5)
- Threshold-based Verdicts :
- Likely Fake: score > 0.7
- Likely Real: score < 0.3
- Inconclusive: 0.3 ≤ score ≤ 0.7
- Explainability : Gradient-based saliency for interpretability

### Code Style

- Frontend : ESLint with Next.js config
- Python : PEP 8 style guide
- TypeScript : Strict mode enabled
- Formatting : Prettier for frontend, Black for Python

## License

This project is proprietary software. All rights reserved.

## Contributors

Bibek Katwal

## Acknowledgments

- Vision Transformer implementation based on HuggingFace Transformers
- UI components from Radix UI and shadcn/ui
- Base model: google/vit-base-patch16-224
- Fine-tuning dataset: Deepfake vs Real image detection

## Support

For issues, questions, or contributions, please contact the development team or open an issue in the project repository.

---

Note : This system is designed for educational and research purposes. Detection results should be verified by human experts for critical applications.
