"use client";

import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Loader2 } from "lucide-react";
import useSWR from "swr";
import type { VideoStatusResponse } from "@/lib/video-api-client";

interface VideoAnalysisProgressProps {
  videoId: string | null;
}

const statusToProgress: Record<string, number> = {
  pending: 15,
  processing: 65,
  done: 100,
  error: 0,
};

export function VideoAnalysisProgress({ videoId }: VideoAnalysisProgressProps) {
  const { data } = useSWR<VideoStatusResponse>(
    videoId ? `/api/video/status/${videoId}` : null,
    (url) => fetch(url).then((r) => r.json()),
    { refreshInterval: 1500 }
  );

  const status = data?.status ?? "pending";
  const progress = statusToProgress[status] ?? 25;

  const statusMessages: Record<string, { label: string; description: string }> = {
    pending: {
      label: "Queued",
      description: "Waiting for analysis to begin...",
    },
    processing: {
      label: "Analyzing",
      description: "Extracting frames and running deepfake detection...",
    },
    done: {
      label: "Complete",
      description: "Analysis finished",
    },
    error: {
      label: "Error",
      description: data?.error_message || "Something went wrong",
    },
  };

  const info = statusMessages[status] ?? statusMessages.pending;

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h2 className="mb-1 text-xl font-semibold">Analysis in Progress</h2>
          {videoId && (
            <p className="text-xs font-mono text-muted-foreground">
              Video ID: {videoId}
            </p>
          )}
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <div className="flex-1">
              <p className="font-medium">{info.label}</p>
              <p className="text-sm text-muted-foreground">{info.description}</p>
            </div>
          </div>

          <Progress value={progress} className="h-2" />

          <div className="flex justify-between text-sm text-muted-foreground">
            <span>{progress}% complete</span>
            {status === "processing" && <span>ETA: ~10–30s</span>}
          </div>
        </div>

        <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
          <p className="text-sm leading-relaxed text-muted-foreground">
            Our model examines extracted frames for signs of manipulation,
            including facial inconsistencies and temporal artifacts.
          </p>
        </div>
      </div>
    </Card>
  );
}
