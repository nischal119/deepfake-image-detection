/**
 * API client for Flask video detection endpoints.
 * Uses Next.js API routes as proxy to Flask backend.
 */

class VideoApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: unknown
  ) {
    super(message);
    this.name = "VideoApiError";
  }
}

async function fetchVideoApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new VideoApiError(
      response.status,
      (errorData as { error?: string }).error || `HTTP ${response.status}`,
      errorData
    );
  }

  return response.json();
}

export interface VideoUploadResponse {
  video_id: string;
  status: string;
  message: string;
}

export interface VideoStatusResponse {
  video_id: string;
  status: "pending" | "processing" | "done" | "error";
  filename: string;
  created_at: string | null;
  error_message?: string;
  video_score?: number;
  prediction?: string;
}

export interface FrameScore {
  frame_index: number;
  score: number;
  heatmap_url: string;
}

export interface VideoResultResponse {
  video_id: string;
  status: string;
  filename: string;
  video_score: number;
  prediction: string;
  num_frames: number;
  processing_time_sec?: number;
  frame_scores: FrameScore[];
}

export const videoApiClient = {
  async uploadVideo(file: File): Promise<VideoUploadResponse> {
    const formData = new FormData();
    formData.append("video", file);

    const res = await fetch("/api/video/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new VideoApiError(res.status, (err as { error?: string }).error || "Upload failed", err);
    }
    return res.json();
  },

  async getStatus(videoId: string): Promise<VideoStatusResponse> {
    return fetchVideoApi<VideoStatusResponse>(`/api/video/status/${videoId}`);
  },

  async getResult(videoId: string): Promise<VideoResultResponse> {
    return fetchVideoApi<VideoResultResponse>(`/api/video/result/${videoId}`);
  },
};

export { VideoApiError };
