"use client";

import { useEffect, useRef, useState } from "react";
import { VideoUploadCard } from "@/components/video/upload-card";
import { VideoAnalysisProgress } from "@/components/video/analysis-progress";
import { VideoResultPanel } from "@/components/video/result-panel";
import { videoApiClient } from "@/lib/video-api-client";
import { useToast } from "@/hooks/use-toast";

type VideoStatus = "idle" | "uploading" | "processing" | "done" | "error";

export default function VideoUploadPage() {
  const [status, setStatus] = useState<VideoStatus>("idle");
  const [videoId, setVideoId] = useState<string | null>(null);
  const [result, setResult] = useState<Awaited<ReturnType<typeof videoApiClient.getResult>> | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  const handleFileSelect = (file: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setUploadedFile(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
    setStatus("idle");
    setResult(null);
    setVideoId(null);
  };

  const handleStartAnalysis = async (file: File) => {
    setStatus("uploading");
    try {
      const data = await videoApiClient.uploadVideo(file);
      setVideoId(data.video_id);
      setStatus("processing");
    } catch (err: unknown) {
      setStatus("error");
      const msg = err instanceof Error ? err.message : "Upload failed";
      toast({
        title: "Upload failed",
        description: msg,
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    if (status !== "processing" || !videoId) return;

    const checkStatus = async () => {
      try {
        const statusData = await videoApiClient.getStatus(videoId);
        if (statusData.status === "done") {
          const resultData = await videoApiClient.getResult(videoId);
          setResult(resultData);
          setStatus("done");
        } else if (statusData.status === "error") {
          setStatus("error");
          toast({
            title: "Analysis failed",
            description: statusData.error_message || "Unknown error",
            variant: "destructive",
          });
        }
      } catch {
        // Keep polling on network errors
      }
    };

    pollRef.current = setInterval(checkStatus, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [status, videoId, toast]);

  const handleNewUpload = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setStatus("idle");
    setVideoId(null);
    setResult(null);
    setUploadedFile(null);
    setPreviewUrl(null);
  };

  return (
    <div className="min-h-screen">
      <main className="container py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">
            Video Deepfake Detection
          </h1>
          <p className="text-muted-foreground">
            Upload a video to analyze for manipulation across frames
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            <VideoUploadCard
              onFileSelect={handleFileSelect}
              onStartAnalysis={handleStartAnalysis}
              disabled={status !== "idle" && status !== "done"}
              currentFile={uploadedFile}
            />
          </div>

          <div>
            {status !== "idle" && status !== "done" && (
              <VideoAnalysisProgress videoId={videoId} />
            )}

            {status === "done" && result && (
              <VideoResultPanel
                result={result}
                previewUrl={previewUrl}
                onNewUpload={handleNewUpload}
              />
            )}

            {status === "idle" && (
              <div className="flex min-h-[400px] w-full items-center justify-center rounded-xl border border-dashed border-border/50 bg-muted/20">
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
