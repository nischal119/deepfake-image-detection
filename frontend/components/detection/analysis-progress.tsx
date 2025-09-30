"use client";

import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Loader2 } from "lucide-react";
import type { DetectionStatus } from "@/lib/types";
import useSWR from "swr";

interface AnalysisProgressProps {
  status: DetectionStatus;
  jobId: string | null;
}

const statusMessages: Record<
  DetectionStatus,
  { label: string; description: string; progress: number }
> = {
  idle: { label: "Ready", description: "Waiting to start", progress: 0 },
  uploading: {
    label: "Uploading",
    description: "Transferring file to server...",
    progress: 15,
  },
  queued: {
    label: "Queued",
    description: "Waiting for analysis to begin...",
    progress: 25,
  },
  analyzing: {
    label: "Analyzing",
    description: "Running frequency analysis and detecting facial artifacts...",
    progress: 65,
  },
  postprocessing: {
    label: "Finalizing",
    description: "Generating heatmaps and compiling results...",
    progress: 90,
  },
  complete: {
    label: "Complete",
    description: "Analysis finished",
    progress: 100,
  },
  error: { label: "Error", description: "Something went wrong", progress: 0 },
};

export function AnalysisProgress({ status, jobId }: AnalysisProgressProps) {
  const statusInfo = statusMessages[status];
  const { data } = useSWR(
    jobId && status !== "complete" ? `/api/jobs/${jobId}` : null,
    (url) => fetch(url).then((r) => r.json()),
    { refreshInterval: 1200 }
  );
  const errorStep: string | undefined =
    data?.status === "error" ? data?.step : undefined;

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h2 className="mb-1 text-xl font-semibold">Analysis in Progress</h2>
          {jobId && (
            <p className="text-xs text-muted-foreground font-mono">
              Job ID: {jobId}
            </p>
          )}
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <div className="flex-1">
              <p className="font-medium">{statusInfo.label}</p>
              <p className="text-sm text-muted-foreground">
                {status === "error" && errorStep
                  ? errorStep
                  : statusInfo.description}
              </p>
            </div>
          </div>

          <Progress value={statusInfo.progress} className="h-2" />

          <div className="flex justify-between text-sm text-muted-foreground">
            <span>{statusInfo.progress}% complete</span>
            {status === "analyzing" && <span>ETA: ~6s</span>}
          </div>
        </div>

        <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Our AI model is examining your media for signs of manipulation,
            including facial inconsistencies, temporal artifacts, and
            compression anomalies.
          </p>
        </div>
      </div>
    </Card>
  );
}
