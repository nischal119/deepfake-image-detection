import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    name: process.env.MODEL_NAME || "deepfake-detector",
    version: process.env.MODEL_VERSION || "unknown",
    checkpoint: process.env.MODEL_CHECKPOINT || "unknown",
    device: process.env.MODEL_DEVICE || "cpu",
    latency: {
      p50: Number(process.env.MODEL_P50 || 0),
      p90: Number(process.env.MODEL_P90 || 0),
    },
    metrics: {
      val: Number(process.env.MODEL_VAL || 0),
      test: Number(process.env.MODEL_TEST || 0),
      f1: Number(process.env.MODEL_F1 || 0),
    },
    lastUpdated: new Date().toISOString(),
    health:
      process.env.MODEL_HEALTH === "degraded"
        ? "degraded"
        : process.env.MODEL_HEALTH === "down"
        ? "down"
        : "healthy",
  });
}
