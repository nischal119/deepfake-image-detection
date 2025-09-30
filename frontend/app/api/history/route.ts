import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const page = Number(searchParams.get("page") || 1);
  const pageSize = Number(searchParams.get("pageSize") || 20);

  const [items, total] = await Promise.all([
    prisma.detectionResult.findMany({
      orderBy: { createdAt: "desc" },
      skip: (page - 1) * pageSize,
      take: pageSize,
      include: { job: true },
    }),
    prisma.detectionResult.count(),
  ]);

  return NextResponse.json({
    items: items.map((r) => ({
      id: r.id,
      fileName: r.job.fileName,
      type: r.inputType,
      score: r.score,
      verdict: r.verdict,
      createdAt: r.createdAt.toISOString(),
      thumbnailUrl:
        r.inputType === "image" ? r.job.filePath : "/placeholder.jpg",
    })),
    page,
    pageSize,
    total,
  });
}
