import { type NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { ensureWorkerStarted } from "@/lib/worker";
import { writeFile, mkdir } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const type = (formData.get("type") as string | null)?.toLowerCase();

    if (!file || !type || !["image", "video"].includes(type)) {
      return NextResponse.json(
        { message: "Invalid file or type" },
        { status: 400 }
      );
    }

    const arrayBuffer = await file.arrayBuffer();
    const bytes = Buffer.from(arrayBuffer);

    const uploadsDir = path.join(process.cwd(), "public", "uploads");
    await mkdir(uploadsDir, { recursive: true });
    const fileId = randomUUID();
    const originalName = (file as any).name || `upload-${fileId}`;
    const storedName = `${fileId}-${originalName}`;
    const diskPath = path.join(uploadsDir, storedName);
    await writeFile(diskPath, bytes);
    const filePath = `/uploads/${storedName}`;

    const job = await prisma.detectionJob.create({
      data: {
        status: "queued",
        progress: 0,
        step: "Queued",
        type: type as any,
        fileName: originalName,
        filePath,
        events: {
          create: [{ type: "step", message: "Queued", progress: 0 }],
        },
      },
      select: { id: true },
    });

    ensureWorkerStarted();

    return NextResponse.json(
      { jobId: job.id, status: "queued", estimatedSeconds: 12 },
      { status: 202 }
    );
  } catch (error) {
    console.error("Predict API error:", error);
    return NextResponse.json(
      { message: "Internal server error" },
      { status: 500 }
    );
  }
}
