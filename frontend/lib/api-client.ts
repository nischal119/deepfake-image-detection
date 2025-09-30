/**
 * Client-side API wrapper for making requests through Next.js API routes
 * This ensures sensitive tokens stay on the server side
 */

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: unknown,
  ) {
    super(message)
    this.name = "ApiError"
  }
}

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new ApiError(response.status, errorData.message || `HTTP ${response.status}`, errorData)
  }

  if (response.status === 204) {
    return {} as T
  }

  return response.json()
}

export const apiClient = {
  async uploadForPrediction(file: File, type: "image" | "video") {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("type", type)

    return fetchApi("/api/predict", {
      method: "POST",
      body: formData,
    })
  },

  async getJobStatus(jobId: string) {
    return fetchApi(`/api/jobs/${jobId}`, {
      method: "GET",
    })
  },

  async getJobResult(jobId: string) {
    return fetchApi(`/api/jobs/${jobId}/result`, {
      method: "GET",
    })
  },

  async getHistory(params?: Record<string, string>) {
    const queryString = params ? `?${new URLSearchParams(params).toString()}` : ""
    return fetchApi(`/api/history${queryString}`, {
      method: "GET",
    })
  },

  async deleteHistoryItem(id: string) {
    return fetchApi(`/api/history/${id}`, {
      method: "DELETE",
    })
  },

  async getModelInfo() {
    return fetchApi("/api/model", {
      method: "GET",
    })
  },
}

export { ApiError }
