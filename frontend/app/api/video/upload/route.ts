import { NextRequest, NextResponse } from "next/server";

const FLASK_API = process.env.FLASK_VIDEO_API_URL || "http://localhost:5000";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const res = await fetch(`${FLASK_API}/api/video/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(
        { error: (data as { error?: string }).error || "Upload failed" },
        { status: res.status }
      );
    }
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    console.error("Video upload proxy error:", error);
    return NextResponse.json(
      { error: "Failed to connect to video service" },
      { status: 502 }
    );
  }
}
