export type DetectionStatus =
  | "idle"
  | "uploading"
  | "queued"
  | "analyzing"
  | "postprocessing"
  | "complete"
  | "error";

export type Verdict = "likely_real" | "inconclusive" | "likely_fake";

export interface HistoryItem {
  id: string;
  fileName: string;
  type: "image" | "video";
  score: number;
  verdict: Verdict;
  createdAt: string;
  thumbnailUrl: string;
}

export interface JobStatus {
  jobId: string;
  status: DetectionStatus;
  progress: number;
  step: string;
  etaSeconds: number;
}

export interface DetectionResult {
  id: string;
  createdAt: string;
  input: {
    fileName: string;
    type: "image" | "video";
    dimensions: [number, number];
    durationSec: number;
    previewUrl?: string;
  };
  score: number;
  verdict: Verdict;
  explanations: {
    heatmap: string;
    artifacts: Array<{
      name: string;
      severity: "low" | "medium" | "high";
      frame?: number;
    }>;
    metadata: {
      exif?: Record<string, unknown>;
      codec?: string;
      bitrate?: number;
    };
  };
  timeline?: Array<{
    t: number;
    score: number;
  }>;
  reportUrl: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  checkpoint: string;
  device: string;
  latency: {
    p50: number;
    p90: number;
  };
  metrics: {
    val: number;
    test: number;
    f1: number;
  };
  lastUpdated: string;
  health: "healthy" | "degraded" | "down";
}
