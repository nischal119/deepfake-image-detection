"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { formatFileSize } from "@/lib/format";
import { formatScore } from "@/lib/format";
import { videoApiClient } from "@/lib/video-api-client";
import type { DetectionResult, Verdict } from "@/lib/types";
import type { VideoResultResponse } from "@/lib/video-api-client";
import {
  UploadCloud,
  X,
  ImageIcon,
  VideoIcon,
  FileWarning,
  Loader2,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  Upload,
  FileDown,
  BarChart3,
  Eye,
  Info,
  AlertTriangle,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/* ───────────────────── Types ───────────────────── */

type MediaType = "image" | "video";
type UnifiedStatus = "idle" | "uploading" | "processing" | "complete" | "error";

const MAX_FILE_SIZE = 100 * 1024 * 1024;
const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "video/mp4",
  "video/avi",
  "video/quicktime",
  "video/x-matroska",
  "video/webm",
];

function scoreToVerdict(score: number): Verdict {
  if (score < 0.3) return "likely_real";
  if (score < 0.6) return "inconclusive";
  return "likely_fake";
}

const verdictConfig: Record<
  Verdict,
  { label: string; icon: typeof ShieldCheck; color: string; bg: string; glow: string; desc: string }
> = {
  likely_real: {
    label: "Likely Authentic",
    icon: ShieldCheck,
    color: "#34d399",
    bg: "rgba(52, 211, 153, 0.12)",
    glow: "0 0 40px rgba(52, 211, 153, 0.25)",
    desc: "No significant signs of manipulation were detected",
  },
  inconclusive: {
    label: "Inconclusive",
    icon: ShieldQuestion,
    color: "#fbbf24",
    bg: "rgba(251, 191, 36, 0.12)",
    glow: "0 0 40px rgba(251, 191, 36, 0.25)",
    desc: "Some anomalies detected — manual review recommended",
  },
  likely_fake: {
    label: "Likely Manipulated",
    icon: ShieldAlert,
    color: "#f87171",
    bg: "rgba(248, 113, 113, 0.12)",
    glow: "0 0 40px rgba(248, 113, 113, 0.25)",
    desc: "High probability of synthetic manipulation detected",
  },
};

/* ───────────────────── Main Page ───────────────────── */

export default function DetectPage() {
  const [status, setStatus] = useState<UnifiedStatus>("idle");
  const [mediaType, setMediaType] = useState<MediaType | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const { toast } = useToast();

  // Image flow
  const [jobId, setJobId] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<DetectionResult | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Video flow
  const [videoId, setVideoId] = useState<string | null>(null);
  const [videoResult, setVideoResult] = useState<VideoResultResponse | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  /* ── File handling ── */

  const handleFileSelect = useCallback(
    (file: File | null) => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setUploadedFile(file);
      setPreviewUrl(file ? URL.createObjectURL(file) : null);
      setMediaType(file ? (file.type.startsWith("video/") ? "video" : "image") : null);
      setStatus("idle");
      setImageResult(null);
      setVideoResult(null);
      setJobId(null);
      setVideoId(null);
    },
    [previewUrl]
  );

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Unsupported format. Use JPG, PNG, MP4, AVI, MOV, MKV, or WebM.";
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large. Maximum is ${formatFileSize(MAX_FILE_SIZE)}.`;
    }
    return null;
  };

  const handleDrop = useCallback(
    (file: File) => {
      const err = validateFile(file);
      if (err) {
        toast({ title: "Invalid file", description: err, variant: "destructive" });
        return;
      }
      handleFileSelect(file);
    },
    [handleFileSelect, toast]
  );

  /* ── Analysis ── */

  const handleStartAnalysis = useCallback(
    async (file: File) => {
      const isVideo = file.type.startsWith("video/");
      setMediaType(isVideo ? "video" : "image");
      setStatus("uploading");

      try {
        if (isVideo) {
          // Video flow → Flask backend
          const data = await videoApiClient.uploadVideo(file);
          setVideoId(data.video_id);
          setStatus("processing");
        } else {
          // Image flow → Next.js backend
          const form = new FormData();
          form.append("file", file);
          form.append("type", "image");
          const res = await fetch("/api/predict", { method: "POST", body: form });
          if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
          const data = (await res.json()) as { jobId: string };
          setJobId(data.jobId);
          setStatus("processing");

          // SSE for progress
          eventSourceRef.current?.close();
          const es = new EventSource(`/api/jobs/${data.jobId}/events`);
          eventSourceRef.current = es;
          es.addEventListener("complete", async () => {
            es.close();
            await new Promise((r) => setTimeout(r, 600));
            const r2 = await fetch(`/api/jobs/${data.jobId}/result`);
            if (!r2.ok) throw new Error("Result fetch failed");
            setImageResult((await r2.json()) as DetectionResult);
            setStatus("complete");
          });
          es.addEventListener("error", async () => {
            es.close();
            try {
              const s = await fetch(`/api/jobs/${data.jobId}`);
              const sd = (await s.json()) as { status: string };
              if (sd.status === "complete") {
                const r = await fetch(`/api/jobs/${data.jobId}/result`);
                if (r.ok) {
                  setImageResult((await r.json()) as DetectionResult);
                  setStatus("complete");
                  return;
                }
              }
            } catch {}
            setStatus("error");
          });
        }
      } catch (error) {
        setStatus("error");
        toast({
          title: "Analysis failed",
          description: error instanceof Error ? error.message : "Unknown error",
          variant: "destructive",
        });
      }
    },
    [toast]
  );

  /* ── Video polling ── */

  useEffect(() => {
    if (status !== "processing" || mediaType !== "video" || !videoId) return;
    const check = async () => {
      try {
        const sd = await videoApiClient.getStatus(videoId);
        if (sd.status === "done") {
          const rd = await videoApiClient.getResult(videoId);
          setVideoResult(rd);
          setStatus("complete");
        } else if (sd.status === "error") {
          setStatus("error");
          toast({
            title: "Analysis failed",
            description: sd.error_message || "Unknown error",
            variant: "destructive",
          });
        }
      } catch {}
    };
    pollRef.current = setInterval(check, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [status, mediaType, videoId, toast]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleNewUpload = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setStatus("idle");
    setMediaType(null);
    setUploadedFile(null);
    setPreviewUrl(null);
    setImageResult(null);
    setVideoResult(null);
    setJobId(null);
    setVideoId(null);
  };

  /* ── Derived values ── */

  const score = imageResult?.score ?? videoResult?.video_score ?? null;
  const verdict = score !== null ? scoreToVerdict(score) : null;

  /* ── Render ── */

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background orbs */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <div
          className="absolute top-[-15%] right-[-5%] h-[500px] w-[500px] rounded-full opacity-15 blur-[100px]"
          style={{ background: "oklch(0.62 0.19 280)", animation: "floatOrb 20s ease-in-out infinite" }}
        />
        <div
          className="absolute bottom-[-10%] left-[-5%] h-[400px] w-[400px] rounded-full opacity-10 blur-[90px]"
          style={{ background: "oklch(0.72 0.12 195)", animation: "floatOrb 25s ease-in-out infinite reverse" }}
        />
      </div>
      <div
        className="pointer-events-none fixed inset-0 z-0 opacity-[0.02]"
        style={{
          backgroundImage: "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <main className="container relative z-10 py-8 md:py-12">
        {/* Header */}
        <div className="mb-10 text-center animate-in fade-in slide-in-from-bottom-3 duration-700">
          <h1 className="mb-3 text-3xl font-bold tracking-tight md:text-4xl">
            DeepFake Detection
          </h1>
          <p className="mx-auto max-w-xl text-muted-foreground">
            Upload any image or video — our AI automatically detects the media type and analyzes it
            for signs of manipulation.
          </p>
        </div>

        {/* ── Upload / Progress / Result ── */}
        {status === "idle" && (
          <UploadSection
            onFileSelect={handleFileSelect}
            onDrop={handleDrop}
            onStartAnalysis={handleStartAnalysis}
            currentFile={uploadedFile}
            previewUrl={previewUrl}
            mediaType={mediaType}
          />
        )}

        {(status === "uploading" || status === "processing") && (
          <ProgressSection status={status} mediaType={mediaType} jobId={jobId} videoId={videoId} />
        )}

        {status === "error" && (
          <div className="mx-auto max-w-lg text-center space-y-4">
            <Card className="p-8 border-red-500/30 bg-red-500/5">
              <AlertTriangle className="mx-auto mb-4 h-12 w-12 text-red-400" />
              <h2 className="mb-2 text-xl font-semibold">Analysis Failed</h2>
              <p className="text-muted-foreground mb-6">
                Something went wrong during processing. Please try again.
              </p>
              <Button onClick={handleNewUpload}>Try Again</Button>
            </Card>
          </div>
        )}

        {status === "complete" && score !== null && verdict !== null && (
          <ResultSection
            score={score}
            verdict={verdict}
            mediaType={mediaType!}
            imageResult={imageResult}
            videoResult={videoResult}
            previewUrl={previewUrl}
            onNewUpload={handleNewUpload}
          />
        )}
      </main>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Upload Section
   ═══════════════════════════════════════════════════════════════ */

function UploadSection({
  onFileSelect,
  onDrop,
  onStartAnalysis,
  currentFile,
  previewUrl,
  mediaType,
}: {
  onFileSelect: (f: File | null) => void;
  onDrop: (f: File) => void;
  onStartAnalysis: (f: File) => void;
  currentFile: File | null;
  previewUrl: string | null;
  mediaType: MediaType | null;
}) {
  const [isDragging, setIsDragging] = useState(false);

  return (
    <div className="mx-auto max-w-2xl">
      <Card className="p-6 md:p-8">
        <div className="mb-6">
          <h2 className="mb-1 text-xl font-semibold">Upload Media</h2>
          <p className="text-sm text-muted-foreground">
            Drag & drop or select an image or video file
          </p>
        </div>

        {!currentFile ? (
          <div
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const f = e.dataTransfer.files[0];
              if (f) onDrop(f);
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
            onPaste={(e) => {
              for (let i = 0; i < e.clipboardData.items.length; i++) {
                const item = e.clipboardData.items[i];
                if (item.type.startsWith("image")) {
                  const f = item.getAsFile();
                  if (f) onDrop(f);
                  break;
                }
              }
            }}
            className={`relative flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-200 ${
              isDragging
                ? "border-primary bg-primary/5 scale-[1.01]"
                : "border-border/50 bg-muted/20 hover:border-primary/50 hover:bg-muted/30"
            }`}
            role="button"
            tabIndex={0}
          >
            <input
              type="file"
              accept={ALLOWED_TYPES.join(",")}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onDrop(f);
              }}
              className="absolute inset-0 cursor-pointer opacity-0"
              aria-label="File input"
            />
            <div
              className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl"
              style={{ backgroundColor: "oklch(0.62 0.19 280 / 0.12)" }}
            >
              <UploadCloud className="h-8 w-8" style={{ color: "oklch(0.62 0.19 280)" }} />
            </div>
            <h3 className="mb-1 text-lg font-medium">Drop your file here</h3>
            <p className="mb-5 text-sm text-muted-foreground">
              Images (JPG, PNG) or Videos (MP4, AVI, MOV, MKV, WebM) up to 100MB
            </p>
            <Button type="button" variant="secondary">
              Select File
            </Button>
          </div>
        ) : (
          <div className="space-y-5">
            {/* File info */}
            <Card className="flex items-center gap-4 border-border/50 bg-muted/30 p-4">
              <div
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl"
                style={{ backgroundColor: mediaType === "video" ? "oklch(0.72 0.12 195 / 0.15)" : "oklch(0.62 0.19 280 / 0.15)" }}
              >
                {mediaType === "video" ? (
                  <VideoIcon className="h-6 w-6" style={{ color: "oklch(0.72 0.12 195)" }} />
                ) : (
                  <ImageIcon className="h-6 w-6" style={{ color: "oklch(0.62 0.19 280)" }} />
                )}
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="truncate font-medium">{currentFile.name}</p>
                <div className="flex items-center gap-2">
                  <p className="text-sm text-muted-foreground">{formatFileSize(currentFile.size)}</p>
                  <Badge variant="outline" className="text-xs capitalize">{mediaType}</Badge>
                </div>
              </div>
              <Button size="icon" variant="ghost" onClick={() => onFileSelect(null)} aria-label="Remove">
                <X className="h-4 w-4" />
              </Button>
            </Card>

            {/* Preview */}
            {previewUrl && (
              <div className="max-w-md mx-auto overflow-hidden rounded-xl border border-border/50">
                {mediaType === "video" ? (
                  // eslint-disable-next-line jsx-a11y/media-has-caption
                  <video src={previewUrl} controls className="h-auto w-full" />
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt="Preview" className="h-auto w-full" />
                )}
              </div>
            )}

            <Button onClick={() => onStartAnalysis(currentFile)} className="w-full" size="lg">
              <Eye className="mr-2 h-5 w-5" />
              Analyze {mediaType === "video" ? "Video" : "Image"}
            </Button>
          </div>
        )}

        <div className="mt-6 rounded-lg border border-border/50 bg-muted/20 p-4">
          <div className="flex gap-3">
            <FileWarning className="h-5 w-5 shrink-0 text-muted-foreground" />
            <div className="space-y-0.5 text-sm">
              <p className="font-medium">Auto-detection</p>
              <p className="text-muted-foreground">
                The system automatically detects whether you uploaded an image or video and routes
                to the appropriate analysis pipeline.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Progress Section
   ═══════════════════════════════════════════════════════════════ */

function ProgressSection({
  status,
  mediaType,
  jobId,
  videoId,
}: {
  status: UnifiedStatus;
  mediaType: MediaType | null;
  jobId: string | null;
  videoId: string | null;
}) {
  const pct = status === "uploading" ? 20 : 60;
  const label = status === "uploading" ? "Uploading…" : "Analyzing…";
  const desc =
    status === "uploading"
      ? "Transferring your file to the server"
      : mediaType === "video"
        ? "Extracting frames, running deepfake detection & generating heatmaps…"
        : "Running frequency analysis and detecting facial artifacts…";

  return (
    <div className="mx-auto max-w-lg">
      <Card className="p-8">
        <div className="space-y-6">
          <div className="text-center">
            <Loader2 className="mx-auto mb-4 h-10 w-10 animate-spin text-primary" />
            <h2 className="mb-1 text-xl font-semibold">{label}</h2>
            <p className="text-sm text-muted-foreground">{desc}</p>
          </div>

          <Progress value={pct} className="h-2" />

          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{pct}%</span>
            {status === "processing" && <span>ETA: ~{mediaType === "video" ? "15-30" : "6"}s</span>}
          </div>

          {(jobId || videoId) && (
            <p className="text-center text-xs font-mono text-muted-foreground">
              ID: {jobId || videoId}
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Result Section
   ═══════════════════════════════════════════════════════════════ */

function ResultSection({
  score,
  verdict,
  mediaType,
  imageResult,
  videoResult,
  previewUrl,
  onNewUpload,
}: {
  score: number;
  verdict: Verdict;
  mediaType: MediaType;
  imageResult: DetectionResult | null;
  videoResult: VideoResultResponse | null;
  previewUrl: string | null;
  onNewUpload: () => void;
}) {
  const vc = verdictConfig[verdict];
  const VerdictIcon = vc.icon;
  const filename = imageResult?.input?.fileName ?? videoResult?.filename ?? "Media";

  const timelineData =
    videoResult?.frame_scores?.map((f) => ({ t: f.frame_index, score: f.score })) ?? [];

  const frameHeatmaps =
    videoResult?.frame_scores?.filter((f) => f.heatmap_url && f.heatmap_url.length > 0) ?? [];

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* ── Score Card ── */}
      <Card className="overflow-hidden">
        <div className="relative p-8 md:p-10" style={{ background: vc.bg }}>
          <div
            className="absolute inset-0 opacity-40"
            style={{
              background: `radial-gradient(circle at 50% 0%, ${vc.color}22, transparent 70%)`,
            }}
          />
          <div className="relative flex flex-col items-center text-center">
            {/* Animated ring */}
            <div
              className="relative mb-6 flex h-36 w-36 items-center justify-center rounded-full border-4"
              style={{
                borderColor: vc.color,
                boxShadow: vc.glow,
              }}
            >
              <div className="text-center">
                <div className="text-4xl font-bold" style={{ color: vc.color }}>
                  {(score * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground mt-0.5">Fake Score</div>
              </div>
            </div>

            <div className="flex items-center gap-2 mb-2">
              <VerdictIcon className="h-6 w-6" style={{ color: vc.color }} />
              <span className="text-2xl font-bold" style={{ color: vc.color }}>
                {vc.label}
              </span>
            </div>
            <p className="text-sm text-muted-foreground max-w-md">{vc.desc}</p>
            <p className="mt-3 text-xs text-muted-foreground">{filename}</p>
          </div>
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-border/40 border-t border-border/40">
          <QuickStat label="Media Type" value={mediaType === "video" ? "Video" : "Image"} />
          <QuickStat
            label="Prediction"
            value={(videoResult?.prediction ?? (score >= 0.35 ? "Fake" : "Real"))}
          />
          {videoResult && <QuickStat label="Frames" value={String(videoResult.num_frames)} />}
          {videoResult?.processing_time_sec != null && (
            <QuickStat label="Time" value={`${videoResult.processing_time_sec}s`} />
          )}
          {imageResult && (
            <>
              <QuickStat
                label="Dimensions"
                value={`${imageResult.input.dimensions[0]}×${imageResult.input.dimensions[1]}`}
              />
              <QuickStat label="Score" value={formatScore(imageResult.score)} />
            </>
          )}
        </div>
      </Card>

      {/* ── Evidence Tabs ── */}
      <Card className="p-6">
        <Tabs defaultValue={frameHeatmaps.length > 0 || (imageResult?.explanations?.heatmap) ? "heatmaps" : "details"} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="heatmaps" className="gap-1.5">
              <Eye className="h-3.5 w-3.5" />
              Heatmaps
            </TabsTrigger>
            {mediaType === "video" && (
              <TabsTrigger value="timeline" className="gap-1.5">
                <BarChart3 className="h-3.5 w-3.5" />
                Timeline
              </TabsTrigger>
            )}
            <TabsTrigger value="details" className="gap-1.5">
              <Info className="h-3.5 w-3.5" />
              Details
            </TabsTrigger>
          </TabsList>

          {/* Heatmaps tab */}
          <TabsContent value="heatmaps" className="mt-4 space-y-4">
            {/* Video heatmaps */}
            {mediaType === "video" && frameHeatmaps.length > 0 && (
              <>
                <div>
                  <h3 className="mb-1 font-semibold">Gradient Saliency Heatmaps</h3>
                  <p className="text-sm text-muted-foreground">
                    Pixel-level attention showing which regions influenced the fake detection score.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                  {frameHeatmaps
                    .sort((a, b) => b.score - a.score)
                    .map((f) => (
                      <Card key={f.frame_index} className="overflow-hidden border-border/50 bg-muted/10 group">
                        <div className="relative aspect-video w-full overflow-hidden">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={f.heatmap_url}
                            alt={`Heatmap frame ${f.frame_index}`}
                            className="h-full w-full object-cover transition-transform group-hover:scale-105"
                          />
                        </div>
                        <div className="flex items-center justify-between px-3 py-2">
                          <span className="text-xs text-muted-foreground">Frame {f.frame_index}</span>
                          <Badge
                            variant="outline"
                            className="text-xs"
                            style={{
                              borderColor: f.score >= 0.6 ? "#f87171" : f.score >= 0.3 ? "#fbbf24" : "#34d399",
                              color: f.score >= 0.6 ? "#f87171" : f.score >= 0.3 ? "#fbbf24" : "#34d399",
                            }}
                          >
                            {(f.score * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </Card>
                    ))}
                </div>
              </>
            )}

            {/* Video — no heatmaps */}
            {mediaType === "video" && frameHeatmaps.length === 0 && (
              <div className="flex flex-col items-center py-12 text-center">
                <Eye className="mb-3 h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">
                  No saliency heatmaps were generated for this video.
                </p>
              </div>
            )}

            {/* Image heatmap */}
            {mediaType === "image" && imageResult?.explanations?.heatmap && (
              <>
                <div>
                  <h3 className="mb-1 font-semibold">Attention Heatmap</h3>
                  <p className="text-sm text-muted-foreground">
                    Model attention regions correlated with manipulation likelihood.
                  </p>
                </div>
                <div className="relative max-w-2xl overflow-hidden rounded-xl border border-border/50">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl || ""}
                    alt="Original"
                    className="w-full h-auto"
                  />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageResult.explanations.heatmap}
                    alt="Heatmap"
                    className="pointer-events-none absolute inset-0 w-full h-full object-cover"
                    style={{ opacity: 0.6, mixBlendMode: "multiply" }}
                  />
                </div>
              </>
            )}

            {mediaType === "image" && !imageResult?.explanations?.heatmap && (
              <div className="flex flex-col items-center py-12 text-center">
                <Eye className="mb-3 h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">No heatmap data available.</p>
              </div>
            )}
          </TabsContent>

          {/* Timeline tab (video only) */}
          {mediaType === "video" && (
            <TabsContent value="timeline" className="mt-4 space-y-4">
              <div>
                <h3 className="mb-1 font-semibold">Per-Frame Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  Fake probability scores across sampled video frames.
                </p>
              </div>
              {timelineData.length > 0 ? (
                <Card className="border-border/50 bg-muted/10 p-4">
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={timelineData}>
                      <defs>
                        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#f87171" stopOpacity={0.7} />
                          <stop offset="100%" stopColor="#f87171" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="t"
                        label={{ value: "Frame Index", position: "insideBottom", offset: -5 }}
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        label={{ value: "Fake Prob.", angle: -90, position: "insideLeft" }}
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1a1a2e",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: "8px",
                          fontSize: 12,
                        }}
                        labelStyle={{ color: "#fff" }}
                        formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Score"]}
                      />
                      <Area type="monotone" dataKey="score" stroke="#f87171" fill="url(#sg)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>
              ) : (
                <p className="text-sm text-muted-foreground py-8 text-center">No frame data.</p>
              )}
            </TabsContent>
          )}

          {/* Details tab */}
          <TabsContent value="details" className="mt-4 space-y-4">
            <div>
              <h3 className="mb-1 font-semibold">Analysis Details</h3>
              <p className="text-sm text-muted-foreground">
                Technical metadata from the detection run
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4 rounded-xl border border-border/50 bg-muted/10 p-5">
              <DetailItem label="File Name" value={filename} />
              <DetailItem label="Media Type" value={mediaType} />
              <DetailItem label="Verdict" value={vc.label} color={vc.color} />
              <DetailItem label="Fake Score" value={`${(score * 100).toFixed(2)}%`} />
              {videoResult && (
                <>
                  <DetailItem label="Frames Analyzed" value={String(videoResult.num_frames)} />
                  <DetailItem label="Processing Time" value={`${videoResult.processing_time_sec ?? "N/A"}s`} />
                  <DetailItem label="Prediction" value={videoResult.prediction} />
                </>
              )}
              {imageResult && (
                <>
                  <DetailItem
                    label="Dimensions"
                    value={`${imageResult.input.dimensions[0]}×${imageResult.input.dimensions[1]}`}
                  />
                </>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </Card>

      {/* Actions */}
      <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
        <Button onClick={onNewUpload} size="lg" className="min-w-44">
          <Upload className="mr-2 h-4 w-4" />
          New Upload
        </Button>
      </div>
    </div>
  );
}

/* ── Tiny helpers ── */

function QuickStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-4 py-3 text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-semibold capitalize text-sm">{value}</p>
    </div>
  );
}

function DetailItem({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium capitalize" style={color ? { color } : undefined}>
        {value}
      </p>
    </div>
  );
}
