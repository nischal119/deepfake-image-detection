import { NextRequest, NextResponse } from "next/server";

const FLASK_API = process.env.FLASK_VIDEO_API_URL || "http://localhost:5000";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ videoId: string }> }
) {
  try {
    const { videoId } = await params;
    const res = await fetch(`${FLASK_API}/api/video/result/${videoId}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(data, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error("Video result proxy error:", error);
    return NextResponse.json(
      { error: "Failed to connect to video service" },
      { status: 502 }
    );
  }
}
