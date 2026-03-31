import { ApiClient } from './client';

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
    role?: string;
  };
}

export interface CurrentUser {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export const AuthService = {
  login: async (credentials: { username: string; password: string }) => {
    // Backend auth.py expects UserLogin schema (JSON) with 'email' and 'password'
    const result = await ApiClient.post<AuthResponse>('/auth/login', {
      email: credentials.username,
      password: credentials.password
    });

    localStorage.setItem('aurora_token', result.access_token);
    return result;
  },

  register: (data: any) => {
    // Hits /api/v1/auth/register
    return ApiClient.post<AuthResponse>('/auth/register', data);
  },

  getMe: () => {
    return ApiClient.get<CurrentUser>('/auth/me');
  },

  logout: () => {
    localStorage.removeItem('aurora_token');
  },

  isAuthenticated: () => {
    return !!localStorage.getItem('aurora_token');
  }
};
