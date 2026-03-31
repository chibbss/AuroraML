import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ArrowLeft, UploadCloud, FileText, Zap, Trophy, ChevronRight, Trash2, ExternalLink } from 'lucide-react';
import { ModelsService, MLModelResponse } from '../api/models';
import { JobsService, JobResponse } from '../api/jobs';
import { DatasetService } from '../api/datasets';
import { Dropdown } from '../components/ui/Dropdown';
import './ProjectDetails.css';

export const ProjectDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [models, setModels] = useState<MLModelResponse[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [projectDatasets, setProjectDatasets] = useState<any[]>([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);
  const [latestJob, setLatestJob] = useState<JobResponse | null>(null);

  useEffect(() => {
    if (id) {
        fetchModels();
        fetchDatasets();
        fetchJobs();
    }
  }, [id]);

  const fetchJobs = async () => {
    try {
      const data = await JobsService.listProjectJobs(id!);
      if (data.jobs && data.jobs.length > 0) {
        // Sort by created_at desc
        const sorted = data.jobs.sort((a, b) => 
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        setLatestJob(sorted[0]);
      }
    } catch (err) {
      console.error("Failed to fetch jobs", err);
    }
  };

  const fetchDatasets = async () => {
    setIsLoadingDatasets(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/v1/datasets/projects/${id}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('aurora_token')}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setProjectDatasets(data);
      }
    } catch (err) {
      console.error("Failed to fetch project datasets", err);
    } finally {
      setIsLoadingDatasets(false);
    }
  };

  const fetchModels = async () => {
    setIsLoadingModels(true);
    try {
      const data = await ModelsService.getProjectModels(id!);
      setModels(data.models);
    } catch (err) {
      console.error("Failed to fetch models", err);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!window.confirm('Are you sure you want to delete this model?')) return;
    try {
        await ModelsService.deleteModel(modelId);
        fetchModels();
    } catch (err) {
        console.error("Failed to delete model", err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
        alert("Please select a file first.");
        return;
    }
    console.log("Starting upload for file:", file.name);
    setIsUploading(true);
    setUploadProgress(10);
    
    try {
      const interval = setInterval(() => {
        setUploadProgress(prev => (prev >= 90 ? 90 : prev + 10));
      }, 500);

      const response = await DatasetService.uploadDataset(id!, file);
      
      clearInterval(interval);
      setUploadProgress(100);
      console.log("Upload successful:", response);
      
      setTimeout(() => {
        setFile(null);
        setIsUploading(false);
        setUploadProgress(0);
        navigate(`/projects/${id}/datasets/${response.id}/profile`);
      }, 500);

    } catch (error: any) {
      console.error('Failed to upload dataset', error);
      alert(`Upload failed: ${error.message || 'Unknown error'}. Check console for details.`);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="project-details-container">
      <div className="details-header">
        <button className="back-btn" onClick={() => navigate('/projects')}>
          <ArrowLeft size={20} />
          <span>Back to Projects</span>
        </button>
        <div className="title-section">
          <h2>Project Dashboard</h2>
          <div className="project-id text-muted">ID: {id}</div>
        </div>
      </div>

      <div className="details-grid">
        {/* Dataset Card (Conditional) */}
        {isLoadingDatasets ? (
            <Card className="upload-card">
              <div className="p-8 text-center text-muted">Scanning for datasets...</div>
            </Card>
        ) : projectDatasets.length > 0 ? (
            <Card className="dataset-summary-card" glowTheme="cyan">
              <div className="dataset-summary-header">
                <div className="dataset-summary-copy">
                    <p className="dataset-summary-kicker">Dataset Workspace</p>
                    <h3 className="m-0">Project Dataset</h3>
                    <p className="text-muted text-sm dataset-summary-filename">{projectDatasets[0].original_filename}</p>
                </div>
                <div className="badge-outline success dataset-status-badge">
                    {latestJob?.status === 'running' ? 'Training...' : 
                     latestJob?.status === 'completed' ? 'AutoML Finished' : 'Ready'}
                </div>
              </div>

              <div className="dataset-meta-panel">
                <div className="dataset-meta-intro">
                  <span className="dataset-meta-title">Dataset Snapshot</span>
                  <span className="dataset-meta-note">Core structural properties of the currently active training file.</span>
                </div>
                <div className="mini-stats-grid">
                  <div className="mini-stat">
                      <span className="label">Rows</span>
                      <span className="value">{projectDatasets[0].num_rows?.toLocaleString() || '---'}</span>
                  </div>
                  <div className="mini-stat">
                      <span className="label">Columns</span>
                      <span className="value">{projectDatasets[0].num_columns || '---'}</span>
                  </div>
                  <div className="mini-stat">
                      <span className="label">Type</span>
                      <span className="value uppercase">{projectDatasets[0].file_type}</span>
                  </div>
                </div>
              </div>

              <div className="dataset-action-panel">
                <div className="dataset-action-copy">
                  <span className="dataset-meta-title">Actions</span>
                  <span className="dataset-meta-note">Train on this version or replace it with a fresh upload.</span>
                </div>
                <div className="dataset-action-group">
                  <Button 
                      variant="primary" 
                      className="w-full"
                      onClick={() => {
                          if (latestJob && (latestJob.status === 'running' || latestJob.status === 'completed')) {
                              navigate(`/projects/${id}/jobs/${latestJob.id}/dashboard`);
                          } else {
                              navigate(`/projects/${id}/datasets/${projectDatasets[0].id}/profile`);
                          }
                      }}
                      leftIcon={<Zap size={18} />}
                  >
                      {latestJob?.status === 'running' ? 'View Live Training' : 
                       latestJob?.status === 'completed' ? 'View Results' : 'Start Training Pipeline'}
                  </Button>
                  <Button 
                      variant="secondary" 
                      className="w-full"
                      onClick={() => setProjectDatasets([])} // Allow re-upload
                      leftIcon={<UploadCloud size={18} />}
                  >
                      Upload Fresh Version
                  </Button>
                </div>
              </div>
            </Card>
        ) : (
            <Card className="upload-card">
              <h3>Dataset Pipeline</h3>
              <p className="subtitle">Upload your training dataset (CSV format)</p>

              <div 
                className="upload-zone" 
                onClick={() => document.getElementById('file-upload')?.click()}
                onDragOver={onDragOver}
                onDrop={onDrop}
              >
                <input 
                  id="file-upload" 
                  type="file" 
                  accept=".csv" 
                  className="hidden-input" 
                  onChange={handleFileChange} 
                />
                {file ? (
                  <div className="file-selected">
                    <FileText size={48} className="file-icon" />
                    <div className="file-info">
                      <h4>{file.name}</h4>
                      <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <UploadCloud size={48} className="upload-icon" />
                    <h4>Click or drag CSV file to upload</h4>
                    <span>Supports .csv files up to 500MB</span>
                  </div>
                )}
              </div>

              {isUploading && (
                <div className="upload-progress-container">
                  <div className="progress-bar">
                    <div className="fill cyan" style={{ width: `${uploadProgress}%` }}></div>
                  </div>
                  <span className="text-muted">{uploadProgress}% Uploading...</span>
                </div>
              )}

              <Button 
                className="w-full mt-4" 
                disabled={!file || isUploading}
                onClick={handleUpload}
                isLoading={isUploading}
              >
                Upload to Secure Storage
              </Button>
            </Card>
        )}

        {/* Model Registry Card */}
        <Card className="training-card" glowTheme={models.length > 0 ? "purple" : "none"}>
          <div className="flex justify-between items-center mb-6">
            <h3>Model Registry</h3>
            {models.length > 0 && <span className="badge-purple">{models.length} Models</span>}
          </div>
          
          {isLoadingModels ? (
            <div className="p-4 text-center text-muted">Loading models...</div>
          ) : models.length > 0 ? (
            <div className="model-list">
              {models.map(model => (
                <div 
                  key={model.id} 
                  className="model-item-row"
                  onClick={() => navigate(`/projects/${id}/models/${model.id}/evaluation`)}
                >
                  <div className="model-icon-box">
                    <Trophy size={18} className={model.is_deployed ? "text-green" : "text-purple"} />
                  </div>
                  <div className="model-meta">
                    <div className="model-name-row">
                      <span className="font-medium">{model.name}</span>
                      {model.is_deployed && <span className="status-tag live">Live</span>}
                    </div>
                    <div className="model-subtext">
                      {model.model_type.replace('_', ' ')} • v{model.version}.0
                    </div>
                  </div>
                  <div className="model-score">
                    {(model.metrics?.f1_score || model.metrics?.r2_score || 0).toFixed(3)}
                    <ChevronRight size={16} className="ml-2 text-muted" />
                  </div>
                  <div className="model-actions" onClick={(e) => e.stopPropagation()}>
                    <Dropdown 
                      items={[
                        { label: 'View Metrics', icon: <ExternalLink size={14} />, onClick: () => navigate(`/projects/${id}/models/${model.id}/evaluation`) },
                        { label: 'Delete Model', icon: <Trash2 size={14} />, variant: 'danger', onClick: () => handleDeleteModel(model.id) }
                      ]} 
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="coming-soon-overlay">
              <Zap size={32} className="text-secondary mb-2" />
              <p>Upload a dataset and run the pipeline to see models here</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};
