import { ApiClient } from './client';

export interface ModelReportFeature {
  name: string;
  label: string;
  importance: number;
}

export interface ModelReportClassMetric {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelReport {
  summary?: {
    problem_type?: string;
    framework?: string;
    primary_score?: number;
    feature_count?: number;
    parameter_count?: number;
  };
  top_features?: ModelReportFeature[];
  per_class_metrics?: ModelReportClassMetric[];
  validation?: {
    type?: string;
    holdout_fraction?: number;
    stratified?: boolean;
    cv_folds?: number;
  };
  cross_validation?: {
    metric?: string;
    folds?: number;
    mean_score?: number;
    strategy?: string;
  };
  lineage?: {
    dataset_id?: string;
    dataset_checksum?: string;
    dataset_version?: number;
    dataset_rows?: number;
    dataset_columns?: number;
    target_column?: string;
    problem_type?: string;
    train_rows?: number;
    test_rows?: number;
  };
  class_labels?: string[];
  quality_signals?: Record<string, number | null>;
  deployment?: {
    is_deployed?: boolean;
    stage?: string | null;
    endpoint_url?: string | null;
  };
}

export interface MLModelResponse {
  id: string;
  job_id: string;
  project_id: string;
  name: string;
  model_type: string;
  framework: string;
  version: number;
  metrics: Record<string, any>;
  hyperparameters: Record<string, any>;
  feature_importance: Record<string, number>;
  is_deployed: boolean;
  deployment_stage: string | null;
  endpoint_url: string | null;
  report?: ModelReport;
  created_at: string;
  deployed_at: string | null;
}

export interface FeatureEffectPoint {
  x: number;
  y: number;
}

export interface FeatureEffect {
  feature: string;
  method: string;
  points: FeatureEffectPoint[];
}

export interface FeatureEffectsResponse {
  model_id: string;
  effects: FeatureEffect[];
}

export interface DriftBaselineFeature {
  feature: string;
  dtype: string;
  missing_pct: number;
  mean?: number | null;
  std?: number | null;
  p10?: number | null;
  p90?: number | null;
  top_value?: string | null;
  top_share?: number | null;
  unique_count?: number | null;
}

export interface DriftBaselineResponse {
  model_id: string;
  generated_at: string;
  sample_size: number;
  features: DriftBaselineFeature[];
}

export interface LocalExplanationContribution {
  feature: string;
  value?: string | null;
  baseline?: string | null;
  delta: number;
}

export interface LocalExplanationResponse {
  model_id: string;
  prediction: number;
  prediction_label?: string | null;
  contributions: LocalExplanationContribution[];
}

export interface DeployRequest {
  stage: 'staging' | 'production' | 'archived';
}

export const ModelsService = {
  getProjectModels: (projectId: string) => {
    return ApiClient.get<{models: MLModelResponse[], total: number}>(`/projects/${projectId}/models`);
  },
  
  getModel: (model_id: string) => {
    return ApiClient.get<MLModelResponse>(`/models/${model_id}`);
  },

  getFeatureEffects: (model_id: string) => {
    return ApiClient.get<FeatureEffectsResponse>(`/models/${model_id}/feature-effects`);
  },

  getDriftBaseline: (model_id: string) => {
    return ApiClient.get<DriftBaselineResponse>(`/models/${model_id}/drift-baseline`);
  },

  explainPrediction: (model_id: string, data: Record<string, any>) => {
    return ApiClient.post<LocalExplanationResponse>(`/models/${model_id}/explain`, { data });
  },
  
  deployModel: (model_id: string, stage: 'staging' | 'production') => {
    return ApiClient.post<MLModelResponse>(`/models/${model_id}/deploy`, { stage });
  },
  
  undeployModel: (model_id: string) => {
    return ApiClient.post<MLModelResponse>(`/models/${model_id}/undeploy`, {});
  },
  
  deleteModel: (model_id: string) => {
    return ApiClient.delete(`/models/${model_id}`);
  }
};
