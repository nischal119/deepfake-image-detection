-- Prisma Postgres migration for Detection models

CREATE TYPE "JobStatus" AS ENUM ('uploading', 'queued', 'analyzing', 'postprocessing', 'complete', 'error');
CREATE TYPE "EventType" AS ENUM ('progress', 'step', 'warning', 'complete', 'error');
CREATE TYPE "MediaType" AS ENUM ('image', 'video');
CREATE TYPE "Verdict" AS ENUM ('likely_real', 'inconclusive', 'likely_fake');

CREATE TABLE "DetectionJob" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "status" "JobStatus" NOT NULL DEFAULT 'queued',
    "progress" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "step" TEXT NOT NULL DEFAULT '',
    "type" "MediaType" NOT NULL,
    "fileName" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "errorMessage" TEXT
);

CREATE TABLE "JobEvent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "type" "EventType" NOT NULL,
    "message" TEXT NOT NULL,
    "progress" DOUBLE PRECISION,
    "jobId" TEXT NOT NULL,
    CONSTRAINT "JobEvent_jobId_fkey"
        FOREIGN KEY ("jobId") REFERENCES "DetectionJob" ("id")
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);

CREATE TABLE "DetectionResult" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "score" DOUBLE PRECISION NOT NULL,
    "verdict" "Verdict" NOT NULL,
    "inputType" "MediaType" NOT NULL,
    "inputWidth" INTEGER NOT NULL,
    "inputHeight" INTEGER NOT NULL,
    "durationSec" INTEGER NOT NULL,
    "heatmap" TEXT,
    "artifacts" JSONB,
    "metadata" JSONB,
    "timeline" JSONB,
    "reportUrl" TEXT,
    "jobId" TEXT NOT NULL,
    CONSTRAINT "DetectionResult_jobId_fkey"
        FOREIGN KEY ("jobId") REFERENCES "DetectionJob" ("id")
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);

CREATE UNIQUE INDEX "DetectionResult_jobId_key" ON "DetectionResult"("jobId");
