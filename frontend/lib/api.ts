type FetchOptions = RequestInit & {
  timeout?: number
  retries?: number
}

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

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || ""

async function fetchWithRetry(url: string, options: FetchOptions = {}): Promise<Response> {
  const { timeout = 60000, retries = 2, headers = {}, ...fetchOptions } = options

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  const requestHeaders: HeadersInit = {
    ...headers,
  }

  let lastError: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers: requestHeaders,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new ApiError(response.status, errorData.message || `HTTP ${response.status}`, errorData)
      }

      return response
    } catch (error) {
      lastError = error as Error

      if (attempt < retries && error instanceof ApiError && error.status >= 500) {
        await new Promise((resolve) => setTimeout(resolve, 800 * (attempt + 1)))
        continue
      }

      break
    }
  }

  clearTimeout(timeoutId)
  throw lastError
}

export const api = {
  async get<T>(path: string, options?: FetchOptions): Promise<T> {
    const response = await fetchWithRetry(`${BASE_URL}${path}`, {
      ...options,
      method: "GET",
    })
    return response.json()
  },

  async post<T>(path: string, body?: unknown, options?: FetchOptions): Promise<T> {
    const isFormData = body instanceof FormData

    const response = await fetchWithRetry(`${BASE_URL}${path}`, {
      ...options,
      method: "POST",
      headers: {
        ...(isFormData ? {} : { "Content-Type": "application/json" }),
        ...options?.headers,
      },
      body: isFormData ? body : JSON.stringify(body),
    })
    return response.json()
  },

  async delete<T>(path: string, options?: FetchOptions): Promise<T> {
    const response = await fetchWithRetry(`${BASE_URL}${path}`, {
      ...options,
      method: "DELETE",
    })

    if (response.status === 204) {
      return {} as T
    }

    return response.json()
  },
}

export { ApiError }
