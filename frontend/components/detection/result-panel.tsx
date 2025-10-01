"use client";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScoreGauge } from "@/components/results/score-gauge";
import { EvidenceTabs } from "@/components/results/evidence-tabs";
import { Upload } from "lucide-react";
import type { DetectionResult } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertTriangle } from "lucide-react";
import Image from "next/image";

interface ResultPanelProps {
  result: DetectionResult;
  onNewUpload: () => void;
}

export function ResultPanel({ result, onNewUpload }: ResultPanelProps) {
  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="mb-6">
          <h2 className="mb-1 text-xl font-semibold">Analysis Complete</h2>
          <p className="text-sm text-muted-foreground">
            {result.input.fileName}
          </p>
        </div>

        {result.input.type === "image" && result.input.previewUrl && (
          <div className="mb-6 max-w-sm overflow-hidden rounded-lg border border-border/50">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.input.previewUrl}
              alt="Uploaded preview"
              className="h-auto w-full"
            />
          </div>
        )}
        {result.input.type === "video" && result.input.previewUrl && (
          <div className="mb-6 max-w-md overflow-hidden rounded-lg border border-border/50">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              src={result.input.previewUrl}
              controls
              className="h-auto w-full"
            />
          </div>
        )}

        <ScoreGauge score={result.score} verdict={result.verdict} />

        {result.verdict === "likely_fake" && (
          <Alert variant="destructive" className="mt-6">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>High Risk Detected</AlertTitle>
            <AlertDescription>
              This media shows strong signs of manipulation. Review the evidence
              tabs below for detailed analysis.
            </AlertDescription>
          </Alert>
        )}

        {result.verdict === "inconclusive" && (
          <Alert className="mt-6 border-yellow-500/50 bg-yellow-500/10 text-yellow-500">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Inconclusive Results</AlertTitle>
            <AlertDescription>
              Some manipulations detected. Review the heatmap and artifacts for
              more details.
            </AlertDescription>
          </Alert>
        )}
      </Card>

      <EvidenceTabs result={result} />

      <Card className="p-6">
        <div className="flex flex-col gap-3 sm:flex-row">
          <Button onClick={onNewUpload} className="flex-1">
            <Upload className="mr-2 h-4 w-4" />
            New Upload
          </Button>
        </div>
      </Card>
    </div>
  );
}
