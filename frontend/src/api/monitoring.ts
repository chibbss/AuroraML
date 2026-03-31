import { ApiClient } from './client';

export interface FeatureDrift {
  drift_score: number;
  drift_detected: boolean;
  metric_name: string;
}

export interface ModelDriftResponse {
  drift_share: number;
  number_of_columns: number;
  number_of_drifted_columns: number;
  dataset_drift: boolean;
  feature_drift: Record<string, FeatureDrift>;
  error?: string;
}

export interface LiveDriftFeature {
  feature: string;
  drift_score: number;
  drift_detected: boolean;
  metric: string;
  baseline: Record<string, any>;
  current: Record<string, any>;
}

export interface LiveDriftResponse {
  drift_share: number;
  number_of_columns: number;
  number_of_drifted_columns: number;
  dataset_drift: boolean;
  features: LiveDriftFeature[];
  source?: string;
  window_hours?: number;
  sample_rows?: number;
  message?: string;
  error?: string;
}

export interface ModelHealthResponse {
  status: 'healthy' | 'staging' | 'offline' | 'needs_attention' | 'idle';
  latency_avg: number;
  throughput: number;
  error_rate: number;
  uptime: string;
  source?: string;
  request_count_1h?: number;
}

export interface MonitoringSnapshot {
  window_start: string;
  window_end: string;
  resolution_minutes: number;
  source: string;
  request_count: number;
  row_count: number;
  latency_avg_ms: number;
  latency_p95_ms: number;
  error_rate: number;
  uptime_pct: number;
  drift_share: number;
  drift_detected: boolean;
  payload: Record<string, any>;
}

export interface MonitoringSnapshotResponse {
  model_id: string;
  snapshot: MonitoringSnapshot | null;
}

export interface ProjectMonitoringSummary {
  model_id: string;
  model_name: string;
  project_name?: string;
  stage: string;
  health: ModelHealthResponse;
}

export const MonitoringService = {
  getProjectSummary: (projectId: string) => {
    return ApiClient.get<ProjectMonitoringSummary[]>(`/monitoring/projects/${projectId}/summary`);
  },

  getGlobalSummary: () => {
    return ApiClient.get<ProjectMonitoringSummary[]>('/monitoring/summary');
  },
  
  getModelDrift: (modelId: string) => {
    return ApiClient.get<ModelDriftResponse>(`/monitoring/models/${modelId}/drift`);
  },
  
  getModelHealth: (modelId: string) => {
    return ApiClient.get<ModelHealthResponse>(`/monitoring/models/${modelId}/health`);
  },

  getModelLiveDrift: (modelId: string) => {
    return ApiClient.get<LiveDriftResponse>(`/monitoring/models/${modelId}/drift-live`);
  },

  getLatestSnapshot: (modelId: string) => {
    return ApiClient.get<MonitoringSnapshotResponse>(`/monitoring/models/${modelId}/snapshots/latest`);
  }
};
