import { NextRequest, NextResponse } from "next/server";

const FLASK_API = process.env.FLASK_VIDEO_API_URL || "http://localhost:5001";

export async function GET(
  _request: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const { videoId } = params;
    const res = await fetch(`${FLASK_API}/api/video/status/${videoId}`, {
      cache: "no-store",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(data, { status: res.status });
    }
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("Video status proxy error:", error);
    return NextResponse.json(
      { error: "Failed to connect to video service" },
      { status: 502 }
    );
  }
}
