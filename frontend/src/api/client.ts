const API_BASE_URL = 'http://localhost:8000/api/v1';

interface FetchOptions extends RequestInit {
  data?: any;
}

export class ApiClient {
  private static async request<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
    const { data, headers, ...customConfig } = options;
    
    // Auto attach JWT token if it exists in local storage
    const token = localStorage.getItem('aurora_token');
    
    const isFormData = data instanceof FormData;
    console.log(`[ApiClient] Request to ${endpoint}`, { isFormData, method: options.method });

    const config: RequestInit = {
      ...customConfig,
      headers: {
        ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...headers,
      },
    };

    if (data) {
      config.body = isFormData ? data : JSON.stringify(data);
    }

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
      
      if (response.status === 401) {
        localStorage.removeItem('aurora_token');
        window.location.href = '/login';
        throw new Error('Session expired. Please log in again.');
      }

      if (response.status === 204) {
        return {} as T;
      }

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.detail || 'An error occurred with the API request');
      }
      
      return result as T;
    } catch (error) {
      console.error(`API Request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Generic REST Methods
  static get<T>(endpoint: string, customConfig = {}) {
    return this.request<T>(endpoint, { ...customConfig, method: 'GET' });
  }

  static post<T>(endpoint: string, body: any, customConfig = {}) {
    return this.request<T>(endpoint, { ...customConfig, data: body, method: 'POST' });
  }

  static put<T>(endpoint: string, body: any, customConfig = {}) {
    return this.request<T>(endpoint, { ...customConfig, data: body, method: 'PUT' });
  }

  static delete<T>(endpoint: string, customConfig = {}) {
    return this.request<T>(endpoint, { ...customConfig, method: 'DELETE' });
  }
}
