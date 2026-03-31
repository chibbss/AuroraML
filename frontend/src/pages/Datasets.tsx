import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Search, Database, FileText, Filter, Trash2, ExternalLink } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { DatasetService, Dataset } from '../api/datasets';
import { Dropdown } from '../components/ui/Dropdown';
import './Datasets.css';

export const Datasets: React.FC = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    setIsLoading(true);
    try {
      const data = await DatasetService.getDatasets();
      setDatasets(data || []);
    } catch (err) {
      console.error("Failed to fetch datasets", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDataset = async (datasetId: string) => {
    if (!window.confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) return;
    try {
        await DatasetService.deleteDataset(datasetId);
        fetchDatasets();
    } catch (err) {
        console.error("Failed to delete dataset", err);
        alert("Failed to delete dataset. It might be in use by an active training job.");
    }
  };

  const formatSize = (bytes: number) => {
    // ... same ...
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const filteredDatasets = datasets.filter(ds => 
    ds.original_filename.toLowerCase().includes(searchTerm.toLowerCase()) || 
    ds.project_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const gradeFromScore = (score: number) => {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    if (score >= 50) return 'E';
    return 'F';
  };

  const getHealthBadge = (ds: Dataset) => {
    const qualityScore = ds.summary_stats?.quality_score;
    const datasetType = ds.summary_stats?.dataset_type;
    if (datasetType && datasetType !== 'training_dataset') {
      return { label: 'Non‑Dataset', tone: 'bad' };
    }
    if (typeof qualityScore === 'number') {
      const grade = gradeFromScore(qualityScore);
      const tone = grade === 'A' || grade === 'B' ? 'good' : grade === 'C' || grade === 'D' ? 'warning' : 'bad';
      return { label: `${grade} (${Math.round(qualityScore)})`, tone };
    }

    const columns = (ds.column_names || []).map((name) => name.toLowerCase());
    const keywordHit = columns.some((name) =>
      ["formula", "comment", "cell", "sheet", "calc", "note", "description"].some((k) => name.includes(k))
    );
    const rows = ds.num_rows || 0;
    if (keywordHit || rows < 30) {
      return { label: "Non‑Dataset", tone: "bad" };
    }
    if (rows < 200) {
      return { label: "Review", tone: "warning" };
    }
    return { label: "Healthy", tone: "good" };
  };

  return (
    <div className="datasets-container">
      <div className="page-header">
        <div>
          <h2>Global Datasets</h2>
          <p className="subtitle">Manage and monitor data quality across all your ML projects</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary" leftIcon={<Filter size={18} />}>Filter</Button>
          <Button leftIcon={<Database size={18} />}>Connect Database</Button>
        </div>
      </div>

      <div className="metrics-row">
        <Card className="metric-card" padding="sm" glowTheme="cyan">
          <span className="metric-label">Total Storage Used</span>
          <span className="metric-value">{formatSize(datasets.reduce((acc, d) => acc + (d.file_size || 0), 0))}</span>
        </Card>
        <Card className="metric-card" padding="sm" glowTheme="purple">
          <span className="metric-label">Total Datasets</span>
          <span className="metric-value">{datasets.length}</span>
        </Card>
        <Card className="metric-card" padding="sm" glowTheme="green">
          <span className="metric-label">Avg Data Quality</span>
          <span className="metric-value">
            {datasets.length === 0
              ? "—"
              : (() => {
                  const scored = datasets
                    .map((ds) => ds.summary_stats?.quality_score)
                    .filter((score): score is number => typeof score === 'number');
                  if (scored.length === 0) {
                    return `${Math.round(
                      (datasets.filter((ds) => getHealthBadge(ds).tone === "good").length / datasets.length) * 100
                    )}%`;
                  }
                  const avg = scored.reduce((acc, v) => acc + v, 0) / scored.length;
                  return `${Math.round(avg)}%`;
                })()}
          </span>
        </Card>
      </div>

      <Card className="datasets-table-card" padding="none">
        <div className="table-controls">
          <Input 
            placeholder="Search datasets by name or project..." 
            icon={<Search size={16} />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: '300px', marginBottom: 0 }}
          />
        </div>

        <div className="table-wrapper">
          <table className="aurora-table">
            <thead>
              <tr>
                <th>Dataset Name</th>
                <th>Associated Project</th>
                <th>Size / Rows</th>
                <th>Features</th>
                <th>Data Health</th>
                <th>Uploaded</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={7} className="text-center py-8">Loading datasets...</td></tr>
              ) : filteredDatasets.length === 0 ? (
                <tr><td colSpan={7} className="text-center py-8 text-muted">No datasets found matching your search.</td></tr>
              ) : (
                filteredDatasets.map(ds => (
                  <tr key={ds.id} className="clickable-row" onClick={() => navigate(`/datasets/${ds.id}/explorer`)}>
                    <td>
                      <div className="dataset-name-cell">
                        <div className="d-icon"><FileText size={16} /></div>
                        <div>
                          <div className="ds-name">{ds.original_filename}</div>
                          <div className="ds-id text-muted" style={{ fontSize: '0.7rem' }}>{ds.id}</div>
                        </div>
                      </div>
                    </td>
                    <td><span className="project-badge">{ds.project_name || 'Individual'}</span></td>
                    <td>
                      <div className="size-info">
                        <span className="ds-size">{formatSize(ds.file_size)}</span>
                        <span className="ds-rows text-muted">{ds.num_rows?.toLocaleString() || 0} rows</span>
                      </div>
                    </td>
                    <td>{ds.num_columns || 0} columns</td>
                    <td>
                      {(() => {
                        const health = getHealthBadge(ds);
                        return <span className={`health-badge ${health.tone}`}>{health.label}</span>;
                      })()}
                    </td>
                    <td className="text-muted">{new Date(ds.created_at).toLocaleDateString()}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      <Dropdown 
                        items={[
                          { label: 'Explore Data', icon: <ExternalLink size={14} />, onClick: () => navigate(`/datasets/${ds.id}/explorer`) },
                          { label: 'Delete Dataset', icon: <Trash2 size={14} />, variant: 'danger', onClick: () => handleDeleteDataset(ds.id) }
                        ]} 
                      />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};
