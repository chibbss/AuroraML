import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { ModelsService, MLModelResponse, FeatureEffect, LocalExplanationResponse } from '../api/models';
import './EvaluationDashboard.css';

const formatMetric = (value: unknown, digits = 3, suffix = '') => {
  const numericValue = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numericValue)) return '—';
  return `${numericValue.toFixed(digits)}${suffix}`;
};

const formatPercent = (value: unknown) => {
  const numericValue = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numericValue)) return '—';
  return `${(numericValue * 100).toFixed(1)}%`;
};

const formatLabel = (value: string) =>
  value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

export const EvaluationDashboard: React.FC = () => {
  const { id, modelId } = useParams<{ id: string; modelId: string }>();
  const navigate = useNavigate();
  const [model, setModel] = useState<MLModelResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [featureEffects, setFeatureEffects] = useState<FeatureEffect[]>([]);
  const [effectsLoading, setEffectsLoading] = useState(false);
  const [explainInput, setExplainInput] = useState<string>('{}');
  const [explainResult, setExplainResult] = useState<LocalExplanationResponse | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  useEffect(() => {
    if (!modelId) {
      setIsLoading(false);
      setError('Missing model id.');
      return;
    }

    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await ModelsService.getModel(modelId);
        setModel(data);
      } catch (err) {
        console.error('Failed to fetch model', err);
        setError('Failed to load model evaluation.');
      } finally {
        setIsLoading(false);
      }
    };

    load();
  }, [modelId]);

  useEffect(() => {
    const loadEffects = async () => {
      if (!modelId) return;
      setEffectsLoading(true);
      try {
        const data = await ModelsService.getFeatureEffects(modelId);
        setFeatureEffects(data.effects || []);
      } catch (err) {
        console.error('Failed to fetch feature effects', err);
      } finally {
        setEffectsLoading(false);
      }
    };
    loadEffects();
  }, [modelId]);

  if (isLoading) {
    return <div className="evaluation-container"><div className="evaluation-state">Loading evaluation report...</div></div>;
  }

  if (error) {
    return <div className="evaluation-container"><div className="evaluation-state">{error}</div></div>;
  }

  if (!model) {
    return <div className="evaluation-container"><div className="evaluation-state">Model not found.</div></div>;
  }

  const metrics = model.metrics ?? {};
  const report = model.report ?? {};
  const importanceEntries = Array.isArray(report.top_features) && report.top_features.length > 0
    ? report.top_features.map((item) => ({
        label: item.label || item.name,
        value: typeof item.importance === 'number' ? item.importance : Number(item.importance) || 0,
      }))
    : Object.entries(model.feature_importance ?? {})
        .map(([name, value]) => ({
          label: name,
          value: typeof value === 'number' ? value : Number(value) || 0,
        }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10);
  const maxImportance = importanceEntries.length > 0 ? Math.max(...importanceEntries.map((item) => item.value), 0) : 0;
  const hyperparameters = Object.entries(model.hyperparameters ?? {});
  const perClassMetrics = Array.isArray(report.per_class_metrics) ? report.per_class_metrics : [];
  const endpointUrl = model.endpoint_url ? `${window.location.origin}${model.endpoint_url}` : '';
  const metricCards = [
    { label: 'Accuracy', value: formatPercent(metrics.accuracy) },
    { label: 'F1 Score', value: formatPercent(metrics.f1_score) },
    { label: 'ROC AUC', value: formatPercent(metrics.roc_auc) },
    { label: 'Precision', value: formatMetric(metrics.precision, 3) },
  ];
  const validationLabel = report.validation?.type
    ? formatLabel(report.validation.type)
    : 'Holdout';
  const summaryItems = [
    { label: 'Framework', value: model.framework || '—' },
    { label: 'Model Type', value: formatLabel(model.model_type || 'unknown') },
    { label: 'Deployment Stage', value: model.deployment_stage || 'Stored' },
    { label: 'Feature Count', value: formatMetric(report.summary?.feature_count ?? importanceEntries.length, 0) },
  ];
  const lineageItems = [
    { label: 'Target', value: report.lineage?.target_column || '—' },
    { label: 'Validation', value: validationLabel },
    {
      label: 'Train / Test Rows',
      value: report.lineage?.train_rows && report.lineage?.test_rows
        ? `${report.lineage.train_rows} / ${report.lineage.test_rows}`
        : '—',
    },
    {
      label: 'CV Mean',
      value: typeof report.cross_validation?.mean_score === 'number'
        ? formatMetric(report.cross_validation.mean_score, 3)
        : '—',
    },
    {
      label: 'Dataset Version',
      value: typeof report.lineage?.dataset_version === 'number'
        ? String(report.lineage.dataset_version)
        : '—',
    },
    {
      label: 'Class Labels',
      value: report.class_labels && report.class_labels.length > 0
        ? report.class_labels.join(', ')
        : '—',
    },
  ];

  const renderSparkline = (points: { x: number; y: number }[]) => {
    if (!points || points.length < 2) return null;
    const width = 180;
    const height = 60;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const scaleX = (x: number) => (maxX === minX ? width / 2 : ((x - minX) / (maxX - minX)) * width);
    const scaleY = (y: number) => (maxY === minY ? height / 2 : height - ((y - minY) / (maxY - minY)) * height);
    const path = points
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${scaleX(p.x)},${scaleY(p.y)}`)
      .join(' ');
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="feature-effect-sparkline">
        <path d={path} fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    );
  };

  const handleExplain = async () => {
    if (!modelId) return;
    setExplainError(null);
    setExplainLoading(true);
    try {
      const parsed = JSON.parse(explainInput || '{}');
      const data = await ModelsService.explainPrediction(modelId, parsed);
      setExplainResult(data);
    } catch (err: any) {
      setExplainError(err.message || 'Failed to explain prediction');
      setExplainResult(null);
    } finally {
      setExplainLoading(false);
    }
  };

  const handleCopyEndpoint = async () => {
    if (!endpointUrl || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(endpointUrl);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch (copyError) {
      console.error('Failed to copy endpoint URL', copyError);
    }
  };

  return (
    <div className="evaluation-container evaluation-safe-layout">
      <div className="evaluation-header">
        <div className="evaluation-header-row">
          <div className="evaluation-title-block">
            <p className="evaluation-kicker">Model evaluation</p>
            <h1 className="evaluation-title">{model.name}</h1>
            <p className="evaluation-subtitle">
              {model.framework} model
              {model.deployment_stage ? ` • ${model.deployment_stage}` : ''}
            </p>
          </div>
          <div className="evaluation-actions">
            <Button variant="secondary" onClick={() => navigate(`/projects/${id}`)}>
              Back to Project
            </Button>
          </div>
        </div>
      </div>

      <section className="evaluation-metric-strip" aria-label="Primary metrics">
        {metricCards.map((item) => (
          <Card key={item.label} className="evaluation-metric-card">
            <span className="summary-label">{item.label}</span>
            <p className="evaluation-metric-value">{item.value}</p>
          </Card>
        ))}
      </section>

      <div className="evaluation-report-grid">
        <Card className="evaluation-safe-card evaluation-analysis-card">
          <h3>Feature Importance</h3>
          <p className="evaluation-section-note">
            Ranked contribution of the most influential variables in the trained pipeline.
          </p>
          <div className="evaluation-feature-list">
            {importanceEntries.length > 0 ? (
              importanceEntries.map((item) => (
                <div key={item.label} className="evaluation-feature-row">
                  <span className="evaluation-feature-name">{item.label}</span>
                  <div className="evaluation-feature-track">
                    <div
                      className="evaluation-feature-fill"
                      style={{ width: `${maxImportance > 0 ? (item.value / maxImportance) * 100 : 0}%` }}
                    />
                  </div>
                  <span className="evaluation-feature-value">{formatMetric(item.value, 4)}</span>
                </div>
              ))
            ) : (
              <p className="text-muted">No feature importance available.</p>
            )}
          </div>
        </Card>

        <div className="evaluation-side-column">
          <Card className="evaluation-safe-card">
            <h3>Model Summary</h3>
            <div className="evaluation-summary-grid">
              {summaryItems.map((item) => (
                <div key={item.label} className="evaluation-summary-item">
                  <span className="summary-label">{item.label}</span>
                  <p className="summary-value">{item.value}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="evaluation-safe-card">
            <h3>Validation & Lineage</h3>
            <div className="evaluation-summary-grid">
              {lineageItems.map((item) => (
                <div key={item.label} className="evaluation-summary-item">
                  <span className="summary-label">{item.label}</span>
                  <p className="summary-value">{item.value}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="evaluation-safe-card">
            <h3>Deployment</h3>
            <div className="evaluation-summary-grid">
              <div className="evaluation-summary-item">
                <span className="summary-label">Status</span>
                <p className="summary-value">{model.is_deployed ? 'Deployed' : 'Stored'}</p>
              </div>
              <div className="evaluation-summary-item">
                <span className="summary-label">Endpoint</span>
                <div className="evaluation-endpoint-shell">
                  <p className="evaluation-endpoint">{endpointUrl || '—'}</p>
                  {endpointUrl && (
                    <Button
                      type="button"
                      variant="secondary"
                      className="evaluation-copy-button"
                      onClick={handleCopyEndpoint}
                    >
                      {copied ? 'Copied' : 'Copy'}
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <div className="evaluation-detail-grid">
        <Card className="evaluation-safe-card feature-effects-card">
          <h3>Feature Effects</h3>
          <p className="evaluation-section-note">
            Partial dependence curves for top numeric features (model-based). Use to interpret directionality and sensitivity.
          </p>
          {effectsLoading && <p className="text-muted">Computing feature effects...</p>}
          {!effectsLoading && featureEffects.length === 0 && (
            <p className="text-muted">Feature effects are unavailable for this model.</p>
          )}
          <div className="feature-effects-grid">
            {featureEffects.map((effect) => (
              <div key={effect.feature} className="feature-effect-card">
                <div className="feature-effect-title">{effect.feature}</div>
                <div className="feature-effect-meta">{effect.method === 'model' ? 'Model effect' : 'Data effect'}</div>
                <div className="feature-effect-chart">
                  {renderSparkline(effect.points)}
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="evaluation-safe-card local-explain-card">
          <h3>Local Explanation</h3>
          <p className="evaluation-section-note">
            Paste a single row as JSON to see which features pushed the prediction up or down.
          </p>
          <textarea
            className="explain-input"
            value={explainInput}
            onChange={(e) => setExplainInput(e.target.value)}
            rows={6}
            placeholder='{"tenure": 12, "MonthlyCharges": 80.5, "Contract": "Month-to-month"}'
          />
          <Button variant="secondary" onClick={handleExplain} isLoading={explainLoading} disabled={!explainInput.trim()}>
            Explain Prediction
          </Button>
          {explainError && <p className="text-muted">{explainError}</p>}
          {explainResult && (
            <div className="explain-results">
              <div className="explain-summary">
                <span className="summary-label">Prediction</span>
                <span className="summary-value">{explainResult.prediction.toFixed(3)}</span>
                {explainResult.prediction_label && (
                  <span className="summary-label">Label</span>
                )}
                {explainResult.prediction_label && (
                  <span className="summary-value">{explainResult.prediction_label}</span>
                )}
              </div>
              <div className="explain-list">
                {explainResult.contributions.map((item) => (
                  <div key={item.feature} className="explain-row">
                    <div className="explain-feature">{item.feature}</div>
                    <div className="explain-values">
                      <span className="text-muted">value</span>
                      <span className="mono">{item.value}</span>
                      <span className="text-muted">baseline</span>
                      <span className="mono">{item.baseline}</span>
                    </div>
                    <div className={`explain-delta ${item.delta >= 0 ? 'pos' : 'neg'}`}>
                      {item.delta >= 0 ? '+' : ''}{item.delta.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        {perClassMetrics.length > 0 && (
          <Card className="evaluation-safe-card">
            <h3>Per-Class Metrics</h3>
            <div className="per-class-table">
              <div className="pct-header">
                <span>Class</span>
                <span>P</span>
                <span>R</span>
                <span>F1</span>
                <span>Support</span>
              </div>
              {perClassMetrics.map((item) => (
                <div key={item.label} className="pct-row">
                  <span className="pct-label">{item.label}</span>
                  <span>{formatMetric(item.precision, 2)}</span>
                  <span>{formatMetric(item.recall, 2)}</span>
                  <span>{formatMetric(item.f1, 2)}</span>
                  <span className="pct-support">{formatMetric(item.support, 0)}</span>
                </div>
              ))}
            </div>
          </Card>
        )}

        <Card className="evaluation-safe-card">
          <h3>Hyperparameters</h3>
          <div className="evaluation-params-list">
            {hyperparameters.length > 0 ? (
              hyperparameters.map(([key, value]) => (
                <div key={key} className="evaluation-param-row">
                  <span className="evaluation-param-key">{formatLabel(key)}</span>
                  <span className="evaluation-param-value">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))
            ) : (
              <p className="text-muted">No hyperparameters recorded.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
