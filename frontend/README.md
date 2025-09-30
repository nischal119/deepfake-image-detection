# DeepFake Detector

A production-ready Next.js application for detecting deepfakes in images and videos using AI-powered analysis.

## Features

- 🎯 Upload images (JPG, PNG) or videos (MP4) for deepfake detection
- 📊 Real-time analysis progress with streaming updates
- 🔍 Detailed results with confidence scores, heatmaps, and artifact detection
- 📈 Per-frame timeline analysis for videos
- 📜 Detection history with search and filtering
- 🤖 Model performance metrics and health monitoring
- ⚙️ Customizable settings and webhook integrations

## Environment Variables

### Required

- `API_BASE_URL` - Backend API base URL (server-side only)
- `API_TOKEN` - Backend API authentication token (server-side only, **never** use NEXT*PUBLIC* prefix)

### Optional

- `NEXT_PUBLIC_API_BASE_URL` - Public API base URL for client-side requests (if different from server)
- `NEXT_PUBLIC_ENABLE_WEBCAM` - Enable webcam capture feature
- `NEXT_PUBLIC_ENABLE_AUTH` - Enable authentication features

### Database (Prisma + SQLite)

- `DATABASE_URL` - e.g. `file:./dev.db`

## Database Setup

1. Install deps
   - `pnpm add -D prisma && pnpm add @prisma/client`
2. Generate client
   - `pnpm prisma:generate`
3. Run migrations (creates `prisma/dev.db`)
   - `pnpm prisma:migrate`
4. Open DB studio (optional)
   - `pnpm db:studio`

## Security Notes

⚠️ **Important**: Never expose API tokens to the client side.

- Use `API_TOKEN` (without NEXT*PUBLIC* prefix) for server-side authentication
- All authenticated API calls should go through Next.js API routes or Server Actions
- The `NEXT_PUBLIC_` prefix makes variables available to the browser, so only use it for non-sensitive data

## Setup

1. Set your environment variables in Project Settings (gear icon in top right)
2. Add `API_BASE_URL` and `API_TOKEN` for server-side API authentication
3. Deploy or run locally

## Architecture

- **Client-side**: React components make requests to Next.js API routes
- **API Routes**: Server-side endpoints that authenticate with backend API using secure tokens
- **Backend API**: Your deepfake detection model service

### Backend within Next.js

This project includes server routes that implement a minimal backend:

- `POST /api/predict` – create a detection job and enqueue analysis
- `GET /api/jobs/:jobId` – job status
- `GET /api/jobs/:jobId/events` – SSE progress stream
- `GET /api/jobs/:jobId/result` – final results
- `GET /api/history` – paginated history
- `DELETE /api/history/:id` – delete item
- `GET /api/model` – model info/health
- `GET /api/reports/:id` – download analysis report

This architecture keeps sensitive credentials secure on the server while providing a seamless client experience.
