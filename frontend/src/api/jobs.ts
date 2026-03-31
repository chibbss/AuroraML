import { ApiClient } from './client';

export interface StartJobRequest {
  dataset_id: string;
  target_column: string;
  problem_type?: string;
  model_types?: string[];
  config?: any;
}

export interface JobProgress {
  completed: number;
  total: number;
  current?: string;
  best_score?: number;
  best_model?: string;
}

export interface JobMetricResult {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  r2_score?: number;
  rmse?: number;
  mae?: number;
  hyperparameters?: Record<string, any>;
  error?: string;
}

export interface JobResponse {
  id: string;
  project_id: string;
  dataset_id: string;
  job_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  target_column: string;
  problem_type: string;
  best_model_type: string | null;
  best_model_id: string | null;
  best_score: number | null;
  metrics: Record<string, any> | null;
  all_results: Record<string, JobMetricResult> | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  config: {
    progress?: JobProgress;
    data_cleaning?: any;
    feature_engineering?: any;
  } | null;
}

export const JobsService = {
  startTraining: (projectId: string, request: StartJobRequest) => {
    return ApiClient.post<JobResponse>(`/projects/${projectId}/jobs/train`, request);
  },
  
  getJob: (jobId: string) => {
    return ApiClient.get<JobResponse>(`/jobs/${jobId}`);
  },
  
  listProjectJobs: (projectId: string) => {
    return ApiClient.get<{jobs: JobResponse[], total: number}>(`/projects/${projectId}/jobs`);
  }
};
