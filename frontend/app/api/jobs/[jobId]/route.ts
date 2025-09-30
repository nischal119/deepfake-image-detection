import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET(
  _: Request,
  { params }: { params: { jobId: string } }
) {
  const job = await prisma.detectionJob.findUnique({
    where: { id: params.jobId },
  });
  if (!job) return NextResponse.json({ message: "Not found" }, { status: 404 });

  return NextResponse.json({
    jobId: job.id,
    status: job.status,
    progress: job.progress,
    step: job.step,
    etaSeconds: Math.max(0, Math.round(12 - job.progress * 12)),
  });
}
