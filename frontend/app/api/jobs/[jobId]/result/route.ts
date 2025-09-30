import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET(
  _: Request,
  { params }: { params: { jobId: string } }
) {
  const job = await prisma.detectionJob.findUnique({
    where: { id: params.jobId },
    include: { result: true },
  });
  if (!job) return NextResponse.json({ message: "Not found" }, { status: 404 });
  if (!job.result)
    return NextResponse.json({ message: "Result not ready" }, { status: 202 });

  const r = job.result;
  return NextResponse.json({
    id: r.id,
    createdAt: r.createdAt.toISOString(),
    input: {
      fileName: job.fileName,
      type: r.inputType,
      dimensions: [r.inputWidth, r.inputHeight] as [number, number],
      durationSec: r.durationSec,
      previewUrl: job.filePath,
    },
    score: r.score,
    verdict: r.verdict,
    explanations: {
      heatmap: r.heatmap || "",
      artifacts: (r.artifacts as any) || [],
      metadata: (r.metadata as any) || {},
    },
    timeline: (r.timeline as any) || [],
    reportUrl: r.reportUrl || "",
  });
}
