import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { AlertTriangle, RefreshCw, CheckCircle, Activity, Globe } from 'lucide-react';
import {
  MonitoringService,
  ProjectMonitoringSummary,
  LiveDriftResponse,
  MonitoringSnapshot,
} from '../api/monitoring';
import { ModelsService, DriftBaselineResponse } from '../api/models';
import './Monitoring.css';

export const Monitoring: React.FC = () => {
  const [summaries, setSummaries] = useState<ProjectMonitoringSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [baseline, setBaseline] = useState<DriftBaselineResponse | null>(null);
  const [baselineLoading, setBaselineLoading] = useState(false);
  const [liveDrift, setLiveDrift] = useState<LiveDriftResponse | null>(null);
  const [liveDriftLoading, setLiveDriftLoading] = useState(false);
  const [latestSnapshot, setLatestSnapshot] = useState<MonitoringSnapshot | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'health' | 'drift' | 'traffic'>('health');
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (summaries.length === 0) {
      setSelectedModelId(null);
      return;
    }

    const hasCurrentSelection = selectedModelId && summaries.some((summary) => summary.model_id === selectedModelId);
    if (hasCurrentSelection) return;

    const defaultModel = summaries.find((summary) => summary.health.status !== 'offline') || summaries[0];
    setSelectedModelId(defaultModel?.model_id ?? null);
  }, [summaries]);

  useEffect(() => {
    if (!selectedModelId) {
      setBaseline(null);
      setLiveDrift(null);
      setLatestSnapshot(null);
      return;
    }

    fetchBaseline(selectedModelId);
    fetchLiveDrift(selectedModelId);
    fetchLatestSnapshot(selectedModelId);
  }, [selectedModelId]);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      // Fetch monitoring summaries for ALL deployed models of the user
      const data = await MonitoringService.getGlobalSummary(); 
      setSummaries(data);
    } catch (err) {
      console.error("Failed to fetch monitoring data", err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchBaseline = async (modelId: string) => {
    setBaselineLoading(true);
    try {
      const data = await ModelsService.getDriftBaseline(modelId);
      setBaseline(data);
    } catch (err) {
      console.error('Failed to fetch drift baseline', err);
      setBaseline(null);
    } finally {
      setBaselineLoading(false);
    }
  };

  const fetchLiveDrift = async (modelId: string) => {
    setLiveDriftLoading(true);
    try {
      const data = await MonitoringService.getModelLiveDrift(modelId);
      setLiveDrift(data);
    } catch (err) {
      console.error('Failed to fetch live drift', err);
      setLiveDrift(null);
    } finally {
      setLiveDriftLoading(false);
    }
  };

  const fetchLatestSnapshot = async (modelId: string) => {
    setSnapshotLoading(true);
    try {
      const data = await MonitoringService.getLatestSnapshot(modelId);
      setLatestSnapshot(data.snapshot);
    } catch (err) {
      console.error('Failed to fetch monitoring snapshot', err);
      setLatestSnapshot(null);
    } finally {
      setSnapshotLoading(false);
    }
  };

  const activeModels = summaries.filter(s => s.health.status !== 'offline');
  const selectedModel = summaries.find((summary) => summary.model_id === selectedModelId) || activeModels[0] || summaries[0] || null;
  const avgLatency = activeModels.length > 0 
    ? activeModels.reduce((acc, s) => acc + s.health.latency_avg, 0) / activeModels.length 
    : 0;

  return (
    <div className="monitoring-container">
      <div className="page-header">
        <div>
          <h2>Model Monitoring Dashboard</h2>
          <p className="subtitle">Real-time health and statistical drift analysis</p>
        </div>
        <Button 
          variant="secondary" 
          onClick={fetchData} 
          isLoading={isLoading}
          leftIcon={<RefreshCw size={18} />}
        >
          Refresh Metrics
        </Button>
      </div>

      {summaries.length > 0 && (
        <div className="monitoring-focus-row">
          <div>
            <p className="monitoring-focus-label">Focused model</p>
            <div className="monitoring-focus-meta">
              <strong>{selectedModel?.model_name ?? 'Select a model'}</strong>
              {selectedModel?.project_name ? <span>{selectedModel.project_name}</span> : null}
            </div>
          </div>
          <select
            className="monitoring-select"
            value={selectedModelId ?? ''}
            onChange={(event) => setSelectedModelId(event.target.value || null)}
          >
            {summaries.map((summary) => (
              <option key={summary.model_id} value={summary.model_id}>
                {summary.model_name} · {summary.project_name ?? 'Project'}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="monitoring-tabs">
        {[
          { id: 'health', label: 'Service Health' },
          { id: 'drift', label: 'Drift' },
          { id: 'traffic', label: 'Traffic' },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'health' && (
        <>
          <div className="monitoring-grid">
            <Card className="alert-card" glowTheme="purple">
              <div className="alert-header">
                <AlertTriangle className="text-orange" size={24} />
                <h3>Data Drift Monitor</h3>
              </div>
              <p className="text-muted">
                {liveDrift?.message
                  ? liveDrift.message
                  : liveDrift?.dataset_drift
                  ? 'Recent prediction traffic shows measurable feature drift against the training baseline.'
                  : summaries.length > 0
                  ? 'Using materialized inference snapshots backed by prediction traffic.'
                  : 'No active models currently deployed to monitor.'}
              </p>
              <Button variant="secondary" className="mt-4" size="sm" disabled>
                {latestSnapshot?.source === 'prediction_events' ? 'Using Materialized Traffic Snapshots' : 'Awaiting Live Traffic'}
              </Button>
            </Card>

            <Card className="stats-card">
              <div className="stat-row">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-green" />
                  <span className="text-muted">Avg Latency</span>
                </div>
                <span className="stat-val text-green">{avgLatency.toFixed(1)}ms</span>
              </div>
              <div className="stat-row">
                <div className="flex items-center gap-2">
                  <Globe size={16} className="text-cyan" />
                  <span className="text-muted">Rows Last Hour</span>
                </div>
                <span className="stat-val text-cyan">
                  {summaries.reduce((acc, s) => acc + s.health.throughput, 0).toLocaleString()}
                </span>
              </div>
              <div className="stat-row">
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-purple" />
                  <span className="text-muted">Focused Window</span>
                </div>
                <span className="stat-val">
                  {latestSnapshot ? `${latestSnapshot.resolution_minutes}m` : `${activeModels.length} / ${summaries.length}`}
                </span>
              </div>
            </Card>
          </div>

          <h3>Deployed Models Health</h3>
          <div className="models-list">
            {summaries.length > 0 ? (
              summaries.map(s => (
                <Card
                  key={s.model_id}
                  className={`model-row mb-4 ${selectedModelId === s.model_id ? 'model-row-active' : ''}`}
                  onClick={() => setSelectedModelId(s.model_id)}
                >
                  <div className="m-info">
                    <div className="flex items-center gap-2">
                        <h4>{s.model_name}</h4>
                        <span className="badge-outline info">{s.project_name}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted">
                       <span className="capitalize">{s.stage}</span> • ID: {s.model_id.slice(0,8)}
                    </div>
                  </div>
                  <div className="m-metric">
                    <span className="label">Latency</span>
                    <span className="val">{s.health.latency_avg.toFixed(1)}ms</span>
                  </div>
                  <div className="m-metric">
                    <span className="label">Throughput</span>
                    <span className="val">{s.health.throughput} rows</span>
                  </div>
                  <div className={`m-status status-${s.health.status === 'healthy' ? 'healthy' : 'warning'}`}>
                    {s.health.status.replace('_', ' ')}
                  </div>
                </Card>
              ))
            ) : (
              <div className="p-8 text-center text-muted bg-glass rounded-xl border border-glass">
                 Deploy your first model to see real-time health metrics here.
              </div>
            )}
          </div>
        </>
      )}

      {activeTab === 'traffic' && (
        <Card className="chart-placeholder-card mb-8">
          <div className="chart-header">
            <h3>Inference Traffic Snapshot</h3>
            {selectedModel ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => fetchLatestSnapshot(selectedModel.model_id)}
                isLoading={snapshotLoading}
              >
                Refresh Snapshot
              </Button>
            ) : null}
          </div>
          {selectedModel == null ? (
            <p className="text-muted">Deploy a model and send predictions to populate live traffic metrics.</p>
          ) : snapshotLoading ? (
            <p className="text-muted">Refreshing materialized monitoring window...</p>
          ) : latestSnapshot ? (
            <>
              <div className="snapshot-summary">
                <div className="snapshot-metric">
                  <span className="label">Window</span>
                  <span className="val">
                    {new Date(latestSnapshot.window_start).toLocaleString()} to {new Date(latestSnapshot.window_end).toLocaleTimeString()}
                  </span>
                </div>
                <div className="snapshot-metric">
                  <span className="label">Requests</span>
                  <span className="val">{latestSnapshot.request_count}</span>
                </div>
                <div className="snapshot-metric">
                  <span className="label">Rows</span>
                  <span className="val">{latestSnapshot.row_count}</span>
                </div>
                <div className="snapshot-metric">
                  <span className="label">P95 Latency</span>
                  <span className="val">{latestSnapshot.latency_p95_ms.toFixed(1)}ms</span>
                </div>
                <div className="snapshot-metric">
                  <span className="label">Uptime</span>
                  <span className="val">{latestSnapshot.uptime_pct.toFixed(2)}%</span>
                </div>
                <div className="snapshot-metric">
                  <span className="label">Drift Share</span>
                  <span className="val">{(latestSnapshot.drift_share * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="snapshot-footnote">
                Rollup source: {latestSnapshot.source}. Resolution: {latestSnapshot.resolution_minutes} minutes.
              </div>
            </>
          ) : (
            <p className="text-muted">
              Aurora has not materialized a monitoring window for this model yet. Send live predictions or run a snapshot refresh.
            </p>
          )}
        </Card>
      )}

      {activeTab === 'drift' && (
        <>
          <Card className="baseline-card mb-8">
            <div className="chart-header">
              <h3>Training Drift Baseline</h3>
              <Button variant="secondary" size="sm" onClick={() => {
                if (selectedModel) fetchBaseline(selectedModel.model_id);
              }}>
                Refresh Baseline
              </Button>
            </div>
            {baselineLoading && <p className="text-muted">Loading baseline statistics...</p>}
            {!baselineLoading && !baseline && (
              <p className="text-muted">Deploy a model to generate baseline drift statistics.</p>
            )}
            {baseline && (
              <div className="baseline-grid">
                {baseline.features.slice(0, 6).map((feature) => (
                  <div key={feature.feature} className="baseline-item">
                    <div className="baseline-title">{feature.feature}</div>
                    <div className="baseline-meta">{feature.dtype}</div>
                    <div className="baseline-row">
                      <span className="label">Missing</span>
                      <span className="val">{feature.missing_pct.toFixed(2)}%</span>
                    </div>
                    {typeof feature.mean === 'number' ? (
                      <>
                        <div className="baseline-row">
                          <span className="label">Mean</span>
                          <span className="val">{feature.mean.toFixed(2)}</span>
                        </div>
                        <div className="baseline-row">
                          <span className="label">P10 / P90</span>
                          <span className="val">{feature.p10?.toFixed(2)} / {feature.p90?.toFixed(2)}</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="baseline-row">
                          <span className="label">Top</span>
                          <span className="val">{feature.top_value ?? '—'}</span>
                        </div>
                        <div className="baseline-row">
                          <span className="label">Top Share</span>
                          <span className="val">{typeof feature.top_share === 'number' ? `${(feature.top_share * 100).toFixed(1)}%` : '—'}</span>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card className="baseline-card mb-8">
            <div className="chart-header">
              <h3>Live Drift vs Baseline</h3>
              <Button variant="secondary" size="sm" onClick={() => {
                if (selectedModel) fetchLiveDrift(selectedModel.model_id);
              }}>
                Refresh Drift
              </Button>
            </div>
            {liveDriftLoading && <p className="text-muted">Computing drift scores...</p>}
            {!liveDriftLoading && !liveDrift && (
              <p className="text-muted">Deploy a model to compute live drift scores.</p>
            )}
            {!liveDriftLoading && liveDrift?.message && (
              <p className="text-muted">{liveDrift.message}</p>
            )}
            {liveDrift && (
              <div className="drift-grid">
                {liveDrift.features.slice(0, 6).map((feature) => (
                  <div key={feature.feature} className={`drift-item ${feature.drift_detected ? 'drifted' : ''}`}>
                    <div className="baseline-title">{feature.feature}</div>
                    <div className="baseline-meta">{feature.metric}</div>
                    <div className="baseline-row">
                      <span className="label">Drift Score</span>
                      <span className="val">{feature.drift_score.toFixed(3)}</span>
                    </div>
                    {feature.baseline.mean !== undefined ? (
                      <>
                        <div className="baseline-row">
                          <span className="label">Baseline Mean</span>
                          <span className="val">{Number(feature.baseline.mean).toFixed(2)}</span>
                        </div>
                        <div className="baseline-row">
                          <span className="label">Current Mean</span>
                          <span className="val">{Number(feature.current.mean).toFixed(2)}</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="baseline-row">
                          <span className="label">Top Value</span>
                          <span className="val">{feature.current.top_value ?? '—'}</span>
                        </div>
                        <div className="baseline-row">
                          <span className="label">Top Share</span>
                          <span className="val">{typeof feature.current.top_share === 'number' ? `${(feature.current.top_share * 100).toFixed(1)}%` : '—'}</span>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Card>
        </>
      )}

    </div>
  );
};
