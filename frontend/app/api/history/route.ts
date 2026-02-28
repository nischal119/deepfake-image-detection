import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

const FLASK_API = process.env.FLASK_VIDEO_API_URL || "http://localhost:5001";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const page = Number(searchParams.get("page") || 1);
  const pageSize = Number(searchParams.get("pageSize") || 20);

  // 1. Fetch images from Prisma
  const [dbItems, dbTotal] = await Promise.all([
    prisma.detectionResult.findMany({
      orderBy: { createdAt: "desc" },
      take: 100, // Fetch up to 100 for merging
      include: { job: true },
    }),
    prisma.detectionResult.count(),
  ]);

  const images = dbItems.map((r: any) => ({
    id: r.id,
    fileName: r.job.fileName,
    type: r.inputType,
    score: r.score,
    verdict: r.verdict,
    createdAt: r.createdAt.toISOString(),
  }));

  // 2. Fetch videos from Flask API
  let videos: any[] = [];
  let videoTotal = 0;
  try {
    const res = await fetch(`${FLASK_API}/api/video/history?page=1&pageSize=100`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      videoTotal = data.total || 0;
      
      // Keep only successful/done video detections
      const validVideos = (data.items || []).filter((v: any) => v.status === "done");
      
      videos = validVideos.map((v: any) => {
        let verdict = "inconclusive";
        const score = v.score ?? 0;
        if (score < 0.3) verdict = "likely_real";
        else if (score >= 0.6) verdict = "likely_fake";

        return {
          id: v.id,
          fileName: v.filename,
          type: "video",
          score: score,
          verdict: verdict,
          createdAt: v.created_at || new Date().toISOString(),
        };
      });
    }
  } catch (error) {
    console.error("Failed to fetch video history:", error);
  }

  // 3. Merge and sort descending
  const combined = [...images, ...videos].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  // 4. Paginate
  const start = (page - 1) * pageSize;
  const paginatedItems = combined.slice(start, start + pageSize);
  const total = dbTotal + videoTotal; // Note: approximation for simple UI

  return NextResponse.json({
    items: paginatedItems,
    page,
    pageSize,
    total,
  });
}
