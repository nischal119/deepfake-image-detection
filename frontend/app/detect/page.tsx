"use client";

import { useEffect, useRef, useState } from "react";
import { UploadCard } from "@/components/upload/upload-card";
import { AnalysisProgress } from "@/components/detection/analysis-progress";
import { ResultPanel } from "@/components/detection/result-panel";
import type { DetectionStatus, DetectionResult } from "@/lib/types";

export default function DetectPage() {
  const [status, setStatus] = useState<DetectionStatus>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const handleFileSelect = (file: File | null) => {
    setUploadedFile(file);
    setStatus("idle");
    setResult(null);
  };

  const handleStartAnalysis = async (file: File) => {
    setStatus("uploading");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("type", file.type.startsWith("video") ? "video" : "image");

      const res = await fetch("/api/predict", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = (await res.json()) as { jobId: string };
      setJobId(data.jobId);
      setStatus("queued");

      // Subscribe to SSE for progress
      eventSourceRef.current?.close();
      const es = new EventSource(`/api/jobs/${data.jobId}/events`);
      eventSourceRef.current = es;
      es.addEventListener("progress", (e) => {
        try {
          const payload = JSON.parse((e as MessageEvent).data) as {
            progress: number;
            step: string;
          };
          const step = payload.step?.toLowerCase() || "queued";
          if (step.includes("analyzing")) setStatus("analyzing");
          else if (step.includes("post")) setStatus("postprocessing");
          else setStatus("queued");
        } catch {}
      });
      es.addEventListener("complete", async () => {
        es.close();
        // Final server sync to avoid UI race conditions
        try {
          const s = await fetch(`/api/jobs/${data.jobId}`);
          const statusData = (await s.json()) as { status: string };
          if (statusData.status !== "complete") {
            // small retry window if result not yet persisted
            await new Promise((r) => setTimeout(r, 800));
          }
        } catch {}
        try {
          const r = await fetch(`/api/jobs/${data.jobId}/result`);
          if (r.status === 202) {
            // result not ready yet; short retry
            await new Promise((r2) => setTimeout(r2, 800));
          }
          const r2 = await fetch(`/api/jobs/${data.jobId}/result`);
          if (!r2.ok) throw new Error("Result fetch failed");
          const resultData = (await r2.json()) as DetectionResult;
          setResult(resultData);
          setStatus("complete");
        } catch (e) {
          setStatus("error");
        }
      });
      es.addEventListener("error", async () => {
        es.close();
        // fallback: check server status once before showing error
        try {
          const s = await fetch(`/api/jobs/${data.jobId}`);
          const statusData = (await s.json()) as { status: string };
          if (statusData.status === "complete") {
            const r = await fetch(`/api/jobs/${data.jobId}/result`);
            if (r.ok) {
              const resultData = (await r.json()) as DetectionResult;
              setResult(resultData);
              setStatus("complete");
              return;
            }
          }
        } catch {}
        setStatus("error");
      });
    } catch (error) {
      setStatus("error");
      console.error("Analysis failed:", error);
    }
  };

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const handleNewUpload = () => {
    setStatus("idle");
    setJobId(null);
    setResult(null);
    setUploadedFile(null);
  };

  return (
    <div className="min-h-screen">
      <main className="container py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">
            Detect Deepfakes
          </h1>
          <p className="text-muted-foreground">
            Upload an image or video to analyze for manipulations
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            <UploadCard
              onFileSelect={handleFileSelect}
              onStartAnalysis={handleStartAnalysis}
              disabled={status !== "idle" && status !== "complete"}
              currentFile={uploadedFile}
            />
          </div>

          <div>
            {status !== "idle" && status !== "complete" && (
              <AnalysisProgress status={status} jobId={jobId} />
            )}

            {status === "complete" && result && (
              <ResultPanel result={result} onNewUpload={handleNewUpload} />
            )}

            {status === "idle" && (
              <div className="flex h-full min-h-[400px] items-center justify-center rounded-xl border border-dashed border-border/50 bg-muted/20">
                <p className="text-sm text-muted-foreground">
                  Results will appear here after analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
