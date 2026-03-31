import { ApiClient } from './client';

export interface Project {
  id: string;
  name: string;
  description: string | null;
  owner_id: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectsResponse {
  projects: Project[];
  total: number;
}

export const ProjectService = {
  getProjects: () => {
    return ApiClient.get<ProjectsResponse>('/projects');
  },
  
  createProject: (data: { name: string; description?: string }) => {
    return ApiClient.post<Project>('/projects', data);
  },

  getSystemHealth: () => {
    // The backend router includes /health
    return ApiClient.get<{ status: string; version: string }>('/health');
  },

  deleteProject: (projectId: string) => {
    return ApiClient.delete(`/projects/${projectId}`);
  }
};
