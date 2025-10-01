"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import type { DetectionResult } from "@/lib/types";
import { AlertCircle, Info } from "lucide-react";
import { TimelineChart } from "./timeline-chart";

interface EvidenceTabsProps {
  result: DetectionResult;
}

const severityColors = {
  low: "oklch(0.7 0.15 165)",
  medium: "oklch(0.8 0.15 85)",
  high: "oklch(0.65 0.2 15)",
};

export function EvidenceTabs({ result }: EvidenceTabsProps) {
  const [heatmapOpacity, setHeatmapOpacity] = useState([85]);
  const timelineData = useMemo(() => {
    if (result.timeline && result.timeline.length > 0) return result.timeline;
    // Generate a simple flat timeline when missing
    return Array.from({ length: 20 }).map((_, i) => ({
      t: i,
      score: result.score,
    }));
  }, [result.timeline, result.score]);

  // Consider data URLs, public paths, and empty strings; require non-empty string
  const hasHeatmap = Boolean(
    typeof result.explanations.heatmap === "string" &&
      result.explanations.heatmap.trim().length > 0
  );
  const baseImage = result.input.previewUrl || "/placeholder.jpg";

  const hasArtifacts =
    Array.isArray(result.explanations.artifacts) &&
    result.explanations.artifacts.length > 0;

  return (
    <Card className="p-6">
      <Tabs
        defaultValue={hasHeatmap ? "heatmap" : "artifacts"}
        className="w-full"
      >
        <TabsList className="grid w-full grid-cols-4">
          {hasHeatmap && <TabsTrigger value="heatmap">Heatmap</TabsTrigger>}
          {hasArtifacts && (
            <TabsTrigger value="artifacts">
              Artifacts
              <Badge variant="secondary" className="ml-2">
                {result.explanations.artifacts.length}
              </Badge>
            </TabsTrigger>
          )}
          <TabsTrigger value="metadata">Metadata</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        {hasHeatmap && (
          <TabsContent value="heatmap" className="space-y-4">
            <div>
              <h3 className="mb-2 font-semibold">Attention Heatmap</h3>
              <p className="text-sm text-muted-foreground">
                Colored regions indicate model attention correlated with
                manipulation likelihood.
              </p>
            </div>

            <div className="relative max-w-2xl overflow-hidden rounded-lg border border-border/50 bg-muted/20">
              {/* Base image */}
              <img
                src={baseImage}
                alt="Original media"
                className="w-full h-auto block object-contain"
              />
              {/* Heatmap overlay */}
              <img
                src={result.explanations.heatmap || "/placeholder.svg"}
                alt="Detection heatmap"
                className="pointer-events-none absolute inset-0 w-full h-full object-cover"
                style={{
                  opacity: Math.min(1, Math.max(0, heatmapOpacity[0] / 100)),
                  mixBlendMode: "multiply",
                  filter: "saturate(2.25) contrast(2) hue-rotate(10deg)",
                }}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Heatmap Opacity</span>
                <span className="font-medium">{heatmapOpacity[0]}%</span>
              </div>
              <Slider
                value={heatmapOpacity}
                onValueChange={setHeatmapOpacity}
                min={0}
                max={100}
                step={5}
              />
            </div>
          </TabsContent>
        )}

        {hasArtifacts && (
          <TabsContent value="artifacts" className="space-y-4">
            <div>
              <h3 className="mb-2 font-semibold">Detected Artifacts</h3>
              <p className="text-sm text-muted-foreground">
                Specific anomalies and inconsistencies found during analysis
              </p>
            </div>

            <div className="space-y-3">
              {result.explanations.artifacts.map((artifact, i) => (
                <Card key={i} className="border-border/50 bg-muted/20 p-4">
                  <div className="flex items-start gap-3">
                    <AlertCircle
                      className="h-5 w-5 shrink-0 mt-0.5"
                      style={{ color: severityColors[artifact.severity] }}
                    />
                    <div className="flex-1">
                      <div className="mb-1 flex items-center gap-2">
                        <span className="font-medium">{artifact.name}</span>
                        <Badge
                          variant="outline"
                          style={{
                            borderColor: severityColors[artifact.severity],
                            color: severityColors[artifact.severity],
                          }}
                        >
                          {artifact.severity}
                        </Badge>
                      </div>
                      {artifact.frame !== undefined && (
                        <p className="text-sm text-muted-foreground">
                          Frame {artifact.frame}
                        </p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>
        )}

        <TabsContent value="metadata" className="space-y-4">
          <div>
            <h3 className="mb-2 font-semibold">File Metadata</h3>
            <p className="text-sm text-muted-foreground">
              Technical information extracted from the media file
            </p>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4 rounded-lg border border-border/50 bg-muted/20 p-4">
              <div>
                <p className="text-sm text-muted-foreground">File Name</p>
                <p className="font-medium">{result.input.fileName}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Type</p>
                <p className="font-medium capitalize">{result.input.type}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Dimensions</p>
                <p className="font-medium">
                  {result.input.dimensions[0]} × {result.input.dimensions[1]}
                </p>
              </div>
              {result.input.durationSec > 0 && (
                <div>
                  <p className="text-sm text-muted-foreground">Duration</p>
                  <p className="font-medium">{result.input.durationSec}s</p>
                </div>
              )}
            </div>

            {result.explanations.metadata.codec && (
              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Codec</p>
                    <p className="font-medium font-mono">
                      {result.explanations.metadata.codec}
                    </p>
                  </div>
                  {result.explanations.metadata.bitrate && (
                    <div>
                      <p className="text-sm text-muted-foreground">Bitrate</p>
                      <p className="font-medium">
                        {result.explanations.metadata.bitrate} kbps
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            <Card className="border-border/50 bg-blue-500/10 p-4">
              <div className="flex gap-3">
                <Info className="h-5 w-5 shrink-0 text-blue-500" />
                <p className="text-sm text-muted-foreground">
                  Metadata can be manipulated or stripped. Use this information
                  in conjunction with other evidence.
                </p>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="timeline" className="space-y-4">
          <div>
            <h3 className="mb-2 font-semibold">Per-Frame Analysis</h3>
            <p className="text-sm text-muted-foreground">
              Confidence scores over time showing how manipulation likelihood
              varies.
            </p>
          </div>

          <TimelineChart data={timelineData} />
        </TabsContent>
      </Tabs>
    </Card>
  );
}
