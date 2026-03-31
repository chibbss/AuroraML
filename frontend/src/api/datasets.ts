import { ApiClient } from './client';

export interface DatasetProfile {
  shape: number[];
  dtypes: Record<string, string>;
  missing_values: Record<string, number>;
  missing_percentage: Record<string, number>;
  unique_counts: Record<string, number>;
  duplicated_rows: number;
  recommended_target: string | null;
  problem_type?: string;
  numeric_stats?: Record<string, any>;
  categorical_stats?: Record<string, any>;
  histograms?: Record<string, { bin: string; count: number }[]>;
  correlations?: Record<string, Record<string, number>>;
  sample_data?: any[];
  skewness?: Record<string, number>;
}

export interface DatasetCorrelatedPair {
  left: string;
  right: string;
  strength: number;
}

export interface DatasetFeatureRole {
  name: string;
  label: string;
  role: string;
  confidence: number;
  rationale: string;
}

export interface DatasetFeatureSpotlight {
  name: string;
  label: string;
  role: string;
  quality_score: number;
  note: string;
}

export interface DatasetFinding {
  title: string;
  detail: string;
  severity: string;
  feature?: string | null;
}

export interface DatasetTargetRelationship {
  feature: string;
  strength: number;
  direction: string;
}

export interface DatasetDistributionItem {
  label: string;
  share: number;
}

export interface DatasetTargetHealth {
  label: string;
  value: string;
  status: string;
}

export interface DatasetSegmentItem {
  title: string;
  feature: string;
  cohort: string;
  sample_size: number;
  share_of_rows: number;
  target_signal?: number | null;
  comparison: string;
  insight: string;
}

export interface DatasetResearchSection {
  title: string;
  body: string;
}

export interface DatasetAskResponse {
  answer: string;
  citations: string[];
  provider: string;
  grounded: boolean;
  warning?: string | null;
}

export interface DatasetReport {
  overview: {
    rows: number;
    columns: number;
    numeric_features: number;
    categorical_features: number;
    datetime_features: number;
    quality_score: number;
    modeling_readiness_score: number;
    dataset_type?: string;
    dataset_type_confidence?: number;
    dataset_type_signals?: string[];
  };
  quality: {
    score: number;
    missing_cell_ratio: number;
    duplicate_ratio: number;
    constant_features: string[];
    high_missing_features: string[];
    identifier_like_features: string[];
    high_cardinality_features: string[];
    skewed_features: string[];
    correlated_pairs: DatasetCorrelatedPair[];
    non_dataset_flags?: string[];
    dataset_type?: string;
    dataset_type_confidence?: number;
    dataset_type_signals?: string[];
  };
  modeling_readiness: {
    score: number;
    status: string;
    summary: string;
  };
  target_analysis: {
    recommended_target?: string | null;
    problem_type: string;
    rationale: string;
    target_health: DatasetTargetHealth[];
    distribution: DatasetDistributionItem[];
    top_relationships: DatasetTargetRelationship[];
    imbalance_ratio?: number | null;
  };
  segments: DatasetSegmentItem[];
  feature_roles: DatasetFeatureRole[];
  feature_spotlight: DatasetFeatureSpotlight[];
  findings: DatasetFinding[];
  recommendations: string[];
  analyst_brief: string[];
  research_report: DatasetResearchSection[];
}

export interface CleanConfig {
  target_column: string | null;
  remove_duplicates: boolean;
  drop_missing: boolean;
  missing_threshold: number;
  standardize_names: boolean;
}

export interface Dataset {
  id: string;
  project_id: string;
  project_name?: string;
  original_filename: string;
  file_size: number;
  file_type: string;
  num_rows: number;
  num_columns: number;
  column_names?: string[];
  summary_stats?: Record<string, any>;
  created_at: string;
}

export const DatasetService = {
  getDatasets: () => {
    return ApiClient.get<Dataset[]>('/datasets');
  },

  getDataset: (datasetId: string) => {
    return ApiClient.get<Dataset>(`/datasets/${datasetId}`);
  },

  getProjectDatasets: (projectId: string) => {
    return ApiClient.get<Dataset[]>(`/datasets/projects/${projectId}`);
  },

  uploadDataset: (projectId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return ApiClient.post<Dataset>(`/datasets/projects/${projectId}`, formData);
  },

  getProfile: (datasetId: string) => {
    return ApiClient.get<DatasetProfile>(`/datasets/${datasetId}/profile`);
  },

  getReport: (datasetId: string) => {
    return ApiClient.get<DatasetReport>(`/datasets/${datasetId}/report`);
  },

  askAurora: (datasetId: string, question: string) => {
    return ApiClient.post<DatasetAskResponse>(`/datasets/${datasetId}/ask`, { question });
  },
  
  cleanDataset: (datasetId: string, config: CleanConfig) => {
    return ApiClient.post<{ status: string; initial_rows: number; final_rows: number; dataset_id: string }>(
      `/datasets/${datasetId}/clean`, 
      config
    );
  },

  deleteDataset: (datasetId: string) => {
    return ApiClient.delete(`/datasets/${datasetId}`);
  }
};
