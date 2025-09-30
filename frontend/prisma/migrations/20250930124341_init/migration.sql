-- CreateTable
CREATE TABLE "DetectionJob" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'queued',
    "progress" REAL NOT NULL DEFAULT 0,
    "step" TEXT NOT NULL DEFAULT '',
    "type" TEXT NOT NULL,
    "fileName" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "errorMessage" TEXT
);

-- CreateTable
CREATE TABLE "JobEvent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "type" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "progress" REAL,
    "jobId" TEXT NOT NULL,
    CONSTRAINT "JobEvent_jobId_fkey" FOREIGN KEY ("jobId") REFERENCES "DetectionJob" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "DetectionResult" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "score" REAL NOT NULL,
    "verdict" TEXT NOT NULL,
    "inputType" TEXT NOT NULL,
    "inputWidth" INTEGER NOT NULL,
    "inputHeight" INTEGER NOT NULL,
    "durationSec" INTEGER NOT NULL,
    "heatmap" TEXT,
    "artifacts" JSONB,
    "metadata" JSONB,
    "timeline" JSONB,
    "reportUrl" TEXT,
    "jobId" TEXT NOT NULL,
    CONSTRAINT "DetectionResult_jobId_fkey" FOREIGN KEY ("jobId") REFERENCES "DetectionJob" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "DetectionResult_jobId_key" ON "DetectionResult"("jobId");
