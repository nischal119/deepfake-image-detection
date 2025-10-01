"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, CheckCircle2, AlertCircle } from "lucide-react";
import type { ModelInfo } from "@/lib/types";
import { formatDate } from "@/lib/format";

const fallbackModelInfo: ModelInfo = {
  name: "deepfake-detector",
  version: "unknown",
  checkpoint: "unknown",
  device: "cpu",
  latency: { p50: 0, p90: 0 },
  metrics: { val: 0, test: 0, f1: 0 },
  lastUpdated: new Date().toISOString(),
  health: "healthy",
};

const healthConfig = {
  healthy: {
    label: "Healthy",
    color: "oklch(0.7 0.15 165)",
    bg: "oklch(0.7 0.15 165 / 0.15)",
    icon: CheckCircle2,
  },
  degraded: {
    label: "Degraded",
    color: "oklch(0.8 0.15 85)",
    bg: "oklch(0.8 0.15 85 / 0.15)",
    icon: AlertCircle,
  },
  down: {
    label: "Down",
    color: "oklch(0.65 0.2 15)",
    bg: "oklch(0.65 0.2 15 / 0.15)",
    icon: AlertCircle,
  },
};

export default function ModelPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo>(fallbackModelInfo);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchModel = async () => {
    try {
      const res = await fetch("/api/model", { cache: "no-store" });
      if (!res.ok) throw new Error("Bad response");
      const data = (await res.json()) as ModelInfo;
      setModelInfo(data);
    } catch (e) {
      // Keep fallback but mark degraded
      setModelInfo((prev) => ({ ...prev, health: "degraded" }));
    }
  };

  useEffect(() => {
    fetchModel();
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchModel();
    setIsRefreshing(false);
  };

  const healthInfo = healthConfig[modelInfo.health];
  const HealthIcon = healthInfo.icon;

  return (
    <div className="min-h-screen">
      <main className="container py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">
            Model Information
          </h1>
          <p className="text-muted-foreground">
            View performance metrics and health status of the detection model
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Health Status Card */}
          <Card className="p-6 lg:col-span-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className="flex h-14 w-14 items-center justify-center rounded-xl"
                  style={{ backgroundColor: healthInfo.bg }}
                >
                  <HealthIcon
                    className="h-7 w-7"
                    style={{ color: healthInfo.color }}
                  />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">{modelInfo.name}</h2>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge
                      style={{
                        backgroundColor: healthInfo.bg,
                        color: healthInfo.color,
                        borderColor: healthInfo.color,
                      }}
                    >
                      {healthInfo.label}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      Version {modelInfo.version}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw
                    className={`mr-2 h-4 w-4 ${
                      isRefreshing ? "animate-spin" : ""
                    }`}
                  />
                  Refresh
                </Button>
              </div>
            </div>
          </Card>

          {/* Model Details */}
          <Card className="p-6 lg:col-span-2">
            <h3 className="mb-4 text-lg font-semibold">Model Details</h3>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-1 text-sm text-muted-foreground">Checkpoint</p>
                <p className="font-mono font-medium">{modelInfo.checkpoint}</p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-1 text-sm text-muted-foreground">
                  Runtime Device
                </p>
                <p className="font-mono font-medium">{modelInfo.device}</p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-1 text-sm text-muted-foreground">
                  Last Updated
                </p>
                <p className="font-medium">
                  {formatDate(modelInfo.lastUpdated)}
                </p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-1 text-sm text-muted-foreground">
                  Model Version
                </p>
                <p className="font-medium">{modelInfo.version}</p>
              </div>
            </div>
          </Card>

          {/* Performance Metrics */}
          <Card className="p-6">
            <h3 className="mb-4 text-lg font-semibold">Performance</h3>
            <div className="space-y-4">
              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-2 text-sm text-muted-foreground">
                  Latency (p50)
                </p>
                <p className="text-2xl font-bold">{modelInfo.latency.p50}s</p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
                <p className="mb-2 text-sm text-muted-foreground">
                  Latency (p90)
                </p>
                <p className="text-2xl font-bold">{modelInfo.latency.p90}s</p>
              </div>
            </div>
          </Card>

          {/* Accuracy Metrics */}
          <Card className="p-6 lg:col-span-3">
            <h3 className="mb-4 text-lg font-semibold">Accuracy Metrics</h3>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border border-border/50 bg-muted/20 p-6 text-center">
                <p className="mb-2 text-sm text-muted-foreground">
                  Validation Accuracy
                </p>
                <p
                  className="text-3xl font-bold"
                  style={{ color: "oklch(0.62 0.19 280)" }}
                >
                  {(modelInfo.metrics.val * 100).toFixed(1)}%
                </p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-6 text-center">
                <p className="mb-2 text-sm text-muted-foreground">
                  Test Accuracy
                </p>
                <p
                  className="text-3xl font-bold"
                  style={{ color: "oklch(0.72 0.12 195)" }}
                >
                  {(modelInfo.metrics.test * 100).toFixed(1)}%
                </p>
              </div>

              <div className="rounded-lg border border-border/50 bg-muted/20 p-6 text-center">
                <p className="mb-2 text-sm text-muted-foreground">F1 Score</p>
                <p
                  className="text-3xl font-bold"
                  style={{ color: "oklch(0.75 0.15 350)" }}
                >
                  {(modelInfo.metrics.f1 * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
}
