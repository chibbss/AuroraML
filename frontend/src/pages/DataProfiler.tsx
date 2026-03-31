import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ArrowLeft, CheckCircle, AlertTriangle, Wand2, Database } from 'lucide-react';
import { DatasetService, DatasetProfile } from '../api/datasets';
import { JobsService } from '../api/jobs';
import './DataProfiler.css';

export const DataProfiler: React.FC = () => {
  const { id, datasetId } = useParams<{ id: string, datasetId: string }>();
  const navigate = useNavigate();

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCleaning, setIsCleaning] = useState(false);

  // Cleaning Config State
  const [config, setConfig] = useState<any>({
    target_column: null,
    problem_type: 'classification',
    remove_duplicates: true,
    drop_missing: true,
    missing_threshold: 0.5,
    standardize_names: true
  });

  useEffect(() => {
    if (id && datasetId) {
      fetchProfile();
    }
  }, [id, datasetId]);

  const fetchProfile = async () => {
    setIsLoading(true);
    try {
      const data = await DatasetService.getProfile(datasetId!);
        setProfile(data);
      if (data.recommended_target) {
        setConfig((prev: typeof config) => ({ 
            ...prev, 
            target_column: data.recommended_target,
            problem_type: data.problem_type || 'classification'
        }));
      }
    } catch (err) {
      console.error('Failed to fetch dataset profile', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClean = async () => {
    if (!datasetId || !id || !config.target_column) return;
    setIsCleaning(true);
    try {
      // 1. Run Data Cleaning
      const cleanResult = await DatasetService.cleanDataset(datasetId, config);
      
      // 2. Start ML Training Job
      const job = await JobsService.startTraining(id, {
        dataset_id: cleanResult.dataset_id,
        target_column: config.target_column,
        problem_type: config.problem_type,
        config: {
          data_cleaning: config,
        }
      });
      
      // 3. Navigate to Live Dashboard
      navigate(`/projects/${id}/jobs/${job.id}/dashboard`);
    } catch (err) {
      console.error('Failed to start pipeline', err);
      alert('Pipeline failed to start. Check console for details.');
    } finally {
      setIsCleaning(false);
    }
  };

  if (isLoading) {
    return (
      <div className="profiler-container flex-center">
        <div className="loading-spinner"></div>
        <p className="mt-4 text-muted">Profiling dataset... This may take a moment for large files.</p>
      </div>
    );
  }

  if (!profile) return <div className="profiler-container">Error loading profile.</div>;

  const totalRows = profile.shape[0] || 0;
  const totalCols = profile.shape[1] || 0;
  
  // High level health checks
  const hasDuplicates = profile.duplicated_rows > 0;
  const highMissingCols = Object.entries(profile.missing_percentage)
    .filter(([_, pct]) => pct > 20)
    .map(([col]) => col);

  return (
    <div className="profiler-container">
      <div className="profiler-header">
        <button className="back-btn" onClick={() => navigate(`/projects/${id}`)}>
          <ArrowLeft size={20} />
          <span>Back to Project</span>
        </button>
        <div className="title-section">
          <h2>Data Profiler & Cleaning Pipeline</h2>
          <p className="subtitle">Review data quality and configure preprocessing steps</p>
        </div>
      </div>

      <div className="profiler-grid">
        {/* Left Column: Data Quality & Profiling */}
        <div className="profiler-main">
          <div className="metrics-row mb-6">
            <Card className="metric-card" padding="sm" glowTheme="cyan">
              <span className="metric-label">Total Rows</span>
              <span className="metric-value">{totalRows.toLocaleString()}</span>
            </Card>
            <Card className="metric-card" padding="sm" glowTheme="purple">
              <span className="metric-label">Total Features</span>
              <span className="metric-value">{totalCols}</span>
            </Card>
            <Card className="metric-card" padding="sm" glowTheme={highMissingCols.length > 0 ? "purple" : "green"}>
              <span className="metric-label">Data Health</span>
              <span className="metric-value text-xl flex items-center gap-2">
                {highMissingCols.length > 0 ? <AlertTriangle className="text-orange" size={24} /> : <CheckCircle className="text-green" size={24} />}
                {highMissingCols.length > 0 ? 'Needs Cleaning' : 'Excellent'}
              </span>
            </Card>
          </div>

          <Card className="quality-card mb-6">
            <h3>Automated Insights</h3>
            <ul className="insights-list">
              <li>
                <Database size={18} className="text-cyan" />
                <span>Detected <strong>{Object.keys(profile.dtypes).length}</strong> columns. Target heuristic recommends predicting: <span className="highlight-badge">{profile.recommended_target || 'None found'}</span></span>
              </li>
              {hasDuplicates && (
                <li className="warning">
                  <AlertTriangle size={18} className="text-orange" />
                  <span>Found <strong>{profile.duplicated_rows.toLocaleString()}</strong> exactly duplicated rows.</span>
                </li>
              )}
              {highMissingCols.length > 0 && (
                <li className="warning">
                  <AlertTriangle size={18} className="text-orange" />
                  <span><strong>{highMissingCols.length}</strong> features have over 20% missing values ({highMissingCols.slice(0,3).join(', ')}{highMissingCols.length > 3 ? '...' : ''}).</span>
                </li>
              )}
              {highMissingCols.length === 0 && !hasDuplicates && (
                <li className="success">
                  <CheckCircle size={18} className="text-green" />
                  <span>Dataset is clean! No major missing values or duplicates detected.</span>
                </li>
              )}
            </ul>
          </Card>

          <Card className="features-card" padding="none">
            <div className="p-4 border-b border-glass">
              <h3 className="m-0">Feature Overview</h3>
            </div>
            <div className="table-wrapper max-h-400">
              <table className="aurora-table">
                <thead>
                  <tr>
                    <th>Feature Name</th>
                    <th>Type</th>
                    <th>Missing %</th>
                    <th>Unique</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(profile.dtypes).map(col => (
                    <tr key={col}>
                      <td className="font-medium">{col}</td>
                      <td><span className="type-badge">{profile.dtypes[col]}</span></td>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className={profile.missing_percentage[col] > 20 ? 'text-orange font-bold' : ''}>
                            {profile.missing_percentage[col]}%
                          </span>
                          <div className="mini-progress"><div className={`fill ${profile.missing_percentage[col] > 20 ? 'orange' : 'cyan'}`} style={{width: `${profile.missing_percentage[col]}%`}}></div></div>
                        </div>
                      </td>
                      <td className="text-muted">{profile.unique_counts[col]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Right Column: Configuration Panel */}
        <div className="profiler-sidebar">
          <Card className="config-card" glowTheme="cyan">
            <div className="config-header">
              <Wand2 size={24} className="text-cyan" />
              <h3>Cleaning Pipeline</h3>
            </div>
            <p className="text-muted text-sm mb-6">Configure how AuroraML should prepare this dataset before training.</p>

            <div className="form-section">
              <label className="section-label">1. Target Variable</label>
              <select 
                className="glass-select w-full" 
                value={config.target_column || ''} 
                onChange={e => setConfig({...config, target_column: e.target.value})}
              >
                <option value="" disabled>Select column to predict...</option>
                {Object.keys(profile.dtypes).map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            
            <div className="form-section">
              <label className="section-label">2. Problem Type</label>
              <select 
                className="glass-select w-full" 
                value={config.problem_type} 
                onChange={e => setConfig({...config, problem_type: e.target.value})}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>

            <div className="form-section">
              <label className="section-label">3. Handle Missing Values</label>
              <div className="toggle-row">
                <span>Drop rows/cols with missing data</span>
                <label className="switch">
                  <input type="checkbox" checked={config.drop_missing} onChange={e => setConfig({...config, drop_missing: e.target.checked})} />
                  <span className="slider round"></span>
                </label>
              </div>
              {config.drop_missing && (
                <div className="slider-container mt-3">
                  <div className="flex justify-between text-sm text-muted mb-1">
                    <span>Drop columns missing &gt;</span>
                    <span>{config.missing_threshold * 100}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="10" max="90" step="5" 
                    value={config.missing_threshold * 100} 
                    onChange={e => setConfig({...config, missing_threshold: parseInt(e.target.value)/100})}
                    className="w-full"
                  />
                </div>
              )}
            </div>

            <div className="form-section">
              <label className="section-label">3. Data Quality</label>
              <div className="toggle-row">
                <span>Remove exact duplicate rows</span>
                <label className="switch">
                  <input type="checkbox" checked={config.remove_duplicates} onChange={e => setConfig({...config, remove_duplicates: e.target.checked})} />
                  <span className="slider round"></span>
                </label>
              </div>
              <div className="toggle-row mt-3">
                <span>Standardize column names</span>
                <label className="switch">
                  <input type="checkbox" checked={config.standardize_names} onChange={e => setConfig({...config, standardize_names: e.target.checked})} />
                  <span className="slider round"></span>
                </label>
              </div>
            </div>

            <Button 
              className="w-full mt-6" 
              onClick={handleClean}
              isLoading={isCleaning}
              disabled={!config.target_column}
            >
              Start Pipeline & Train Model
            </Button>
            {!config.target_column && <p className="text-orange text-xs text-center mt-2">Please select a target variable first.</p>}
          </Card>
        </div>
      </div>
    </div>
  );
};
