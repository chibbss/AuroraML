import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { 
  ArrowLeft, 
  BarChart3, 
  Database, 
  Layers, 
  Activity, 
  AlertCircle, 
  Table as TableIcon,
  BrainCircuit,
  ShieldAlert,
  Sparkles,
  Target
} from 'lucide-react';
import { DatasetService, DatasetProfile, Dataset, DatasetReport, DatasetAskResponse } from '../api/datasets';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell,
  PieChart,
  Pie
} from 'recharts';
import './DatasetExplorer.css';

export const DatasetExplorer: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'features' | 'segments' | 'research' | 'ask' | 'correlations' | 'sample'>('overview');
  const [auroraQuestion, setAuroraQuestion] = useState('What should I clean or review first before training a production model on this dataset?');
  const [auroraAnswer, setAuroraAnswer] = useState<DatasetAskResponse | null>(null);
  const [isAskingAurora, setIsAskingAurora] = useState(false);

  useEffect(() => {
    if (datasetId) {
      fetchData();
    }
  }, [datasetId]);

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      console.log("EDA Explorer: Loading dataset", datasetId);

      const [dsResult, profileResult, reportResult] = await Promise.all([
        DatasetService.getDataset(datasetId!),
        DatasetService.getProfile(datasetId!),
        DatasetService.getReport(datasetId!),
      ]);

      setDataset(dsResult);
      setProfile(profileResult);
      setReport(reportResult);

      console.log("EDA Explorer: Success", { metadata: dsResult, profile: profileResult, report: reportResult });
    } catch (err: any) {
      console.error("EDA Explorer: Error", err);
      setError(err.message || "Failed to load data details. Make sure the backend is reachable.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAskAurora = async () => {
    if (!datasetId || !auroraQuestion.trim()) return;
    setIsAskingAurora(true);
    try {
      const response = await DatasetService.askAurora(datasetId, auroraQuestion.trim());
      setAuroraAnswer(response);
    } catch (err: any) {
      console.error('Ask Aurora failed', err);
      setAuroraAnswer({
        answer: err.message || 'Aurora could not answer this question right now.',
        citations: [],
        provider: 'error',
        grounded: true,
      });
    } finally {
      setIsAskingAurora(false);
    }
  };

  if (isLoading) {
    return (
      <div className="explorer-loading" style={{ height: '80vh' }}>
        <div className="aurora-spinner"></div>
        <p className="mt-4 text-cyan animate-pulse">Running Deep Profiling & Correlation Analysis...</p>
      </div>
    );
  }

  if (error || !profile || !dataset || !report) {
    return (
        <div className="flex-center flex-column p-12 text-center" style={{ height: '60vh' }}>
            <AlertCircle size={48} className="text-orange mb-4" />
            <h2 className="mb-2">Analysis Failed</h2>
            <p className="text-muted max-w-400 mb-6">{error || "The dataset profile could not be generated."}</p>
            <Button onClick={fetchData} variant="secondary">Try Again</Button>
            <Button onClick={() => navigate('/datasets')} variant="ghost" className="mt-2">Back to Datasets</Button>
        </div>
    );
  }

  const COLORS = ['#00e4ff', '#7b61ff', '#00ffa3', '#ff923c', '#ff4d4d'];
  const QUALITY_COLORS: Record<string, string> = {
    high: '#ff923c',
    medium: '#ffd166',
    low: '#6ee7b7',
  };

  // Calculate type distribution for pie chart safely
  const typeCounts: Record<string, number> = {};
  if (profile?.dtypes) {
      Object.values(profile.dtypes).forEach(type => {
          const simplified = (type || '').includes('int') || (type || '').includes('float') ? 'Numeric' : 
                             (type || '').includes('bool') ? 'Boolean' : 'Categorical';
          typeCounts[simplified] = (typeCounts[simplified] || 0) + 1;
      });
  }
  const typeData = Object.entries(typeCounts).map(([name, value]) => ({ name, value }));

  // Safe integrity calculation
  const missingGrandTotal = Object.values(profile?.missing_percentage || {}).reduce((a, b) => a + (b || 0), 0);
  const dataIntegrity = (dataset?.num_columns || 0) > 0 
    ? Math.max(0, Math.round(100 - (missingGrandTotal / dataset!.num_columns)))
    : 100;

  const gradeFromScore = (score: number) => {
    if (score >= 90) return 'A';
    if (score >= 80) return 'B';
    if (score >= 70) return 'C';
    if (score >= 60) return 'D';
    if (score >= 50) return 'E';
    return 'F';
  };

  const qualityGrade = gradeFromScore(report.overview.quality_score);

  console.log("DatasetExplorer: Rendering with", { activeTab, typeData, dataIntegrity });

  try {
    return (
      <div className="explorer-container">
      <div className="explorer-header-floating">
        <button className="back-btn" onClick={() => navigate(-1)}>
          <ArrowLeft size={18} />
          <span>Back</span>
        </button>
        <div className="ds-identity">
          <Database size={20} className="text-cyan" />
          <h1>{dataset.original_filename}</h1>
          <span className="badge-purple">EDA MODE</span>
        </div>
        <div className="header-tabs">
          {[
            { id: 'overview', label: 'Overview', icon: <Activity size={16} /> },
            { id: 'features', label: 'Feature Analysis', icon: <BarChart3 size={16} /> },
            { id: 'segments', label: 'Segments', icon: <Layers size={16} /> },
            { id: 'research', label: 'Research Report', icon: <BrainCircuit size={16} /> },
            { id: 'ask', label: 'Ask Aurora', icon: <Sparkles size={16} /> },
            { id: 'correlations', label: 'Correlations', icon: <Layers size={16} /> },
            { id: 'sample', label: 'Data Sample', icon: <TableIcon size={16} /> }
          ].map(tab => (
            <button 
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id as any)}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="explorer-content">
        {activeTab === 'overview' && (
          <div className="tab-fade-in">
            <div className="hero-stats-grid">
              <Card className="hero-card" glowTheme="cyan">
                <div className="h-label">Quality Score</div>
                <div className="h-value">
                  {report.overview.quality_score.toFixed(1)}
                  <span className={`grade-badge grade-${qualityGrade}`}>{qualityGrade}</span>
                </div>
                <div className="h-sub">Dataset health grade</div>
              </Card>
              <Card className="hero-card" glowTheme="purple">
                <div className="h-label">Model Readiness</div>
                <div className="h-value">{report.modeling_readiness.score.toFixed(1)}</div>
                <div className="h-sub">{report.modeling_readiness.status.replace('_', ' ')}</div>
              </Card>
              <Card className="hero-card" glowTheme="green">
                <div className="h-label">Recommended Target</div>
                <div className="h-value explorer-mono-value">{report.target_analysis.recommended_target || 'Review'}</div>
                <div className="h-sub">{report.target_analysis.problem_type}</div>
              </Card>
              <Card className="hero-card" glowTheme="purple">
                <div className="h-label">Duplicates</div>
                <div className="h-value">{profile.duplicated_rows}</div>
                <div className="h-sub">{(report.quality.duplicate_ratio * 100).toFixed(2)}% of rows</div>
              </Card>
            </div>

            <div className="overview-main-grid intelligence-overview-grid">
              <Card className="chart-card intelligence-brief-card">
                <div className="section-headline">
                  <div className="section-heading">
                    <BrainCircuit size={18} className="text-cyan" />
                    <h3>Aurora Analyst Brief</h3>
                  </div>
                  <span className="report-badge">Structured Report</span>
                </div>
                <div className="brief-copy">
                  {report.analyst_brief.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              </Card>

              <Card className="chart-card">
                <div className="section-heading">
                  <BarChart3 size={18} className="text-cyan" />
                  <h3>Variable Type Distribution</h3>
                </div>
                <div className="h-300">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={typeData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {typeData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '8px' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="pie-legend">
                    {typeData.map((d, i) => (
                        <div key={d.name} className="legend-item">
                            <span className="dot" style={{ backgroundColor: COLORS[i % COLORS.length] }}></span>
                            <span className="label">{d.name}</span>
                            <span className="val">{d.value}</span>
                        </div>
                    ))}
                </div>
              </Card>
            </div>

            <div className="intelligence-grid">
              <Card className="intelligence-card">
                <div className="section-heading">
                  <ShieldAlert size={18} className="text-orange" />
                  <h3>Priority Findings</h3>
                </div>
                <div className="insight-list">
                  {report.findings.map((finding) => (
                    <div key={`${finding.title}-${finding.feature || ''}`} className="insight-item">
                      <div
                        className="severity-dot"
                        style={{ backgroundColor: QUALITY_COLORS[finding.severity] || '#94a3b8' }}
                      />
                      <div className="insight-copy">
                        <div className="insight-title-row">
                          <span className="alert-title">{finding.title}</span>
                          <span className={`severity-pill severity-${finding.severity}`}>{finding.severity}</span>
                        </div>
                        <div className="alert-desc">{finding.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              <Card className="intelligence-card">
                <div className="section-heading">
                  <Target size={18} className="text-green" />
                  <h3>Target Intelligence</h3>
                </div>
                <div className="target-intelligence">
                  <div className="target-summary-row">
                    <div>
                      <span className="summary-label">Target</span>
                      <p className="summary-value explorer-mono-value">{report.target_analysis.recommended_target || 'Not detected'}</p>
                    </div>
                    <div>
                      <span className="summary-label">Problem Type</span>
                      <p className="summary-value">{report.target_analysis.problem_type}</p>
                    </div>
                  </div>
                  <p className="insight-paragraph">{report.target_analysis.rationale}</p>
                  <div className="target-health-grid">
                    {report.target_analysis.target_health.map((item) => (
                      <div key={item.label} className="target-health-item">
                        <span className="summary-label">{item.label}</span>
                        <p className="summary-value">{item.value}</p>
                      </div>
                    ))}
                  </div>
                  {report.target_analysis.top_relationships.length > 0 && (
                    <div className="relationship-list">
                      <span className="summary-label">Top Numeric Relationships</span>
                      {report.target_analysis.top_relationships.map((item) => (
                        <div key={item.feature} className="relationship-row">
                          <span className="relationship-name">{item.feature}</span>
                          <span className="relationship-strength">{item.strength.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            </div>

            <div className="intelligence-grid intelligence-grid-secondary">
              <Card className="intelligence-card">
                <div className="section-heading">
                  <Sparkles size={18} className="text-purple" />
                  <h3>Recommended Next Steps</h3>
                </div>
                <div className="recommendation-list">
                  {report.recommendations.map((recommendation) => (
                    <div key={recommendation} className="recommendation-item">
                      <span className="recommendation-index">+</span>
                      <p>{recommendation}</p>
                    </div>
                  ))}
                </div>
              </Card>

              <Card className="intelligence-card">
                <div className="section-heading">
                  <Activity size={18} className="text-cyan" />
                  <h3>Feature Roles</h3>
                </div>
                <div className="feature-role-list">
                  {report.feature_roles.slice(0, 8).map((role) => (
                    <div key={role.name} className="feature-role-item">
                      <div>
                        <div className="feature-role-title">{role.label}</div>
                        <div className="feature-role-note">{role.rationale}</div>
                      </div>
                      <div className="feature-role-meta">
                        <span className="feature-role-pill">{role.role}</span>
                        <span className="feature-role-confidence">{Math.round(role.confidence * 100)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            <Card className="spotlight-card">
              <div className="section-heading">
                <Layers size={18} className="text-cyan" />
                <h3>Feature Spotlight</h3>
              </div>
              <div className="spotlight-grid">
                {report.feature_spotlight.map((item) => (
                  <div key={item.name} className="spotlight-item">
                    <div className="spotlight-head">
                      <span className="spotlight-name">{item.label}</span>
                      <span className="spotlight-score">{item.quality_score.toFixed(0)}</span>
                    </div>
                    <div className="spotlight-role">{item.role}</div>
                    <p className="spotlight-note">{item.note}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {activeTab === 'features' && (
          <div className="features-grid tab-fade-in">
            {Object.keys(profile.dtypes).map(col => {
              const isNumeric = profile.histograms && profile.histograms[col];
              const isCategorical = profile.categorical_stats && profile.categorical_stats[col];
              const numericStats = profile.numeric_stats?.[col];
              const categoricalStats = profile.categorical_stats?.[col];
              const missingPct = profile.missing_percentage?.[col] ?? 0;
              const uniqueCount = profile.unique_counts?.[col];
              const topCount = categoricalStats?.top_values?.[0]?.count ?? 0;
              const topShare = dataset.num_rows ? (topCount / dataset.num_rows) * 100 : 0;
              
              return (
                <Card key={col} className="feature-analysis-card">
                  <div className="feature-header">
                    <div className="f-title">
                      <h4>{col}</h4>
                      <span className="type-tag">{profile.dtypes[col]}</span>
                    </div>
                    {profile.missing_percentage[col] > 0 && (
                      <div className="f-missing">
                        {profile.missing_percentage[col]}% Missing
                      </div>
                    )}
                  </div>

                  <div className="feature-body">
                    <div className="f-viz">
                      {isNumeric ? (
                        <ResponsiveContainer width="100%" height={170}>
                          <BarChart data={profile.histograms![col]}>
                            <XAxis dataKey="bin" hide />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
                              labelStyle={{ color: '#00e4ff' }}
                            />
                            <Bar dataKey="count" fill="#00e4ff" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : isCategorical ? (
                        <ResponsiveContainer width="100%" height={170}>
                          <BarChart data={categoricalStats.top_values} layout="vertical">
                            <XAxis type="number" hide />
                            <YAxis dataKey="label" type="category" width={110} style={{ fontSize: '10px' }} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
                            />
                            <Bar dataKey="count" fill="#7b61ff" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="flex-center h-full text-muted text-xs">No distribution available</div>
                      )}
                    </div>

                    <div className="f-metrics">
                      <div className="f-metrics-header">Feature Summary</div>
                      <div className="f-stats">
                        <div className="f-stat"><span className="l">Missing</span><span className="v">{missingPct.toFixed(2)}%</span></div>
                        <div className="f-stat"><span className="l">Unique</span><span className="v">{uniqueCount ?? '—'}</span></div>
                        {isNumeric && numericStats ? (
                          <>
                            <div className="f-stat"><span className="l">Mean</span><span className="v">{numericStats.mean.toFixed(2)}</span></div>
                            <div className="f-stat"><span className="l">Median</span><span className="v">{numericStats['50%'].toFixed(2)}</span></div>
                            <div className="f-stat"><span className="l">Std Dev</span><span className="v">{numericStats.std.toFixed(2)}</span></div>
                            <div className="f-stat"><span className="l">Min</span><span className="v">{numericStats.min.toFixed(2)}</span></div>
                            <div className="f-stat"><span className="l">P25 / P75</span><span className="v">{numericStats['25%'].toFixed(2)} / {numericStats['75%'].toFixed(2)}</span></div>
                            <div className="f-stat"><span className="l">Max</span><span className="v">{numericStats.max.toFixed(2)}</span></div>
                            {profile.skewness?.[col] !== undefined && (
                              <div className="f-stat"><span className="l">Skew</span><span className="v">{profile.skewness[col].toFixed(2)}</span></div>
                            )}
                          </>
                        ) : isCategorical && categoricalStats ? (
                          <>
                            <div className="f-stat"><span className="l">Mode</span><span className="v">{categoricalStats.mode || 'N/A'}</span></div>
                            <div className="f-stat"><span className="l">Top Share</span><span className="v">{topShare.toFixed(1)}%</span></div>
                            <div className="f-stat"><span className="l">Cardinality</span><span className="v">{categoricalStats.unique}</span></div>
                          </>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {activeTab === 'segments' && (
          <div className="tab-fade-in">
            <div className="segments-grid">
              <Card className="intelligence-card">
                <div className="section-heading">
                  <Layers size={18} className="text-cyan" />
                  <h3>Cohort Discovery</h3>
                </div>
                <p className="subtitle mb-8">
                  AuroraML highlights the largest categorical cohorts that diverge from the overall dataset behavior.
                </p>
                <div className="segment-list">
                  {report.segments.length > 0 ? (
                    report.segments.map((segment) => (
                      <div key={`${segment.feature}-${segment.cohort}`} className="segment-item">
                        <div className="segment-item-head">
                          <div>
                            <div className="segment-title">{segment.title}</div>
                            <div className="segment-meta">
                              {segment.sample_size.toLocaleString()} rows • {(segment.share_of_rows * 100).toFixed(1)}% of dataset
                            </div>
                          </div>
                          <span className="segment-comparison">{segment.comparison}</span>
                        </div>
                        <p className="segment-insight">{segment.insight}</p>
                      </div>
                    ))
                  ) : (
                    <div className="p-12 text-center text-muted">No strong cohort-level segments were detected yet.</div>
                  )}
                </div>
              </Card>

              <Card className="intelligence-card">
                <div className="section-heading">
                  <Target size={18} className="text-green" />
                  <h3>Target Distribution Snapshot</h3>
                </div>
                <div className="distribution-list">
                  {report.target_analysis.distribution.length > 0 ? (
                    report.target_analysis.distribution.map((item) => (
                      <div key={item.label} className="distribution-row">
                        <span className="distribution-label">{item.label}</span>
                        <span className="distribution-value">
                          {report.target_analysis.problem_type === 'classification'
                            ? `${(item.share * 100).toFixed(1)}%`
                            : item.share.toFixed(3)}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="text-muted">Target distribution becomes available after a target is inferred or selected.</div>
                  )}
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'research' && (
          <div className="tab-fade-in">
            <Card className="research-report-card">
              <div className="section-heading mb-8">
                <BrainCircuit size={18} className="text-cyan" />
                <h3>Aurora Research Report</h3>
              </div>
              <div className="research-section-list">
                {report.research_report.map((section) => (
                  <section key={section.title} className="research-section">
                    <h4>{section.title}</h4>
                    <p>{section.body}</p>
                  </section>
                ))}
              </div>
            </Card>
          </div>
        )}

        {activeTab === 'ask' && (
          <div className="tab-fade-in">
            <div className="ask-aurora-grid">
              <Card className="ask-aurora-card">
                <div className="section-heading mb-8">
                  <Sparkles size={18} className="text-cyan" />
                  <h3>Ask Aurora</h3>
                </div>
                <p className="subtitle mb-8">
                  Ask grounded questions about the dataset. Aurora answers from the structured report, profile, and sample data.
                </p>
                <div className="ask-aurora-form">
                  <textarea
                    className="ask-aurora-input"
                    value={auroraQuestion}
                    onChange={(e) => setAuroraQuestion(e.target.value)}
                    placeholder="Ask about target choice, data quality, segment risks, or modeling readiness..."
                    rows={6}
                  />
                  <Button onClick={handleAskAurora} isLoading={isAskingAurora} disabled={!auroraQuestion.trim()}>
                    Ask Aurora
                  </Button>
                </div>
              </Card>

              <Card className="ask-aurora-card">
                <div className="section-heading mb-8">
                  <BrainCircuit size={18} className="text-cyan" />
                  <h3>Grounded Answer</h3>
                </div>
                {auroraAnswer ? (
                  <div className="aurora-answer">
                    <div className="aurora-answer-meta">
                      <span className="report-badge">{auroraAnswer.provider}</span>
                      <span className="text-muted">{auroraAnswer.grounded ? 'Grounded to dataset report' : 'Ungrounded'}</span>
                    </div>
                    <p>{auroraAnswer.answer}</p>
                    {auroraAnswer.warning && <div className="aurora-warning">{auroraAnswer.warning}</div>}
                    {auroraAnswer.citations.length > 0 && (
                      <div className="aurora-citations">
                        <span className="summary-label">Citations</span>
                        <div className="citation-list">
                          {auroraAnswer.citations.map((citation) => (
                            <span key={citation} className="citation-pill">{citation}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-muted">Ask a dataset question to generate a grounded answer.</div>
                )}
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'correlations' && (
          <div className="tab-fade-in">
            <Card className="correlations-card">
              <h3>Pearson Correlation Matrix</h3>
              <p className="subtitle mb-8">Relationships between numerical features (-1.0 to +1.0)</p>
              
              {!profile.correlations ? (
                  <div className="p-12 text-center text-muted">Insufficient numeric features for correlation analysis.</div>
              ) : (
                  <div className="corr-matrix-wrapper">
                    <div className="corr-matrix" style={{ 
                      display: 'grid', 
                      gridTemplateColumns: `auto repeat(${Object.keys(profile.correlations).length}, 1fr)` 
                    }}>
                        <div className="corr-header-empty"></div>
                        {Object.keys(profile.correlations).map(col => (
                            <div key={col} className="corr-header-label">{col}</div>
                        ))}

                        {Object.entries(profile.correlations).map(([rowKey, rowData]) => (
                            <React.Fragment key={rowKey}>
                                <div className="corr-row-label">{rowKey}</div>
                                {Object.values(rowData).map((val, idx) => {
                                    const opacity = Math.abs(val as number);
                                    const color = (val as number) > 0 ? `rgba(0, 228, 255, ${opacity})` : `rgba(255, 77, 77, ${opacity})`;
                                    return (
                                        <div 
                                            key={idx} 
                                            className="corr-cell" 
                                            style={{ backgroundColor: color }}
                                            title={`${rowKey} vs ${Object.keys(profile.correlations!)[idx]}: ${val}`}
                                        >
                                            {(val as number).toFixed(2)}
                                        </div>
                                    );
                                })}
                            </React.Fragment>
                        ))}
                    </div>
                  </div>
              )}
            </Card>
          </div>
        )}

        {activeTab === 'sample' && (
            <div className="tab-fade-in">
                <Card className="sample-card" padding="none">
                    <div className="p-6 border-b border-glass flex justify-between items-center">
                        <h3 className="m-0">Dataset Snapshot (Top 10 Rows)</h3>
                        <span className="text-muted text-sm italic">Showing raw values before cleaning</span>
                    </div>
                    <div className="table-wrapper scroll-x">
                        <table className="aurora-table">
                            <thead>
                                <tr>
                                    {Object.keys(profile.dtypes).map(col => <th key={col}>{col}</th>)}
                                </tr>
                            </thead>
                            <tbody>
                                {profile.sample_data?.map((row, i) => (
                                    <tr key={i}>
                                        {Object.keys(profile.dtypes).map(col => (
                                            <td key={col} className="text-xs truncate max-w-200">
                                                {row[col]?.toString() || <span className="text-orange opacity-50">NaN</span>}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            </div>
        )}
      </div>
    </div>
  );
  } catch (e: any) {
    console.error("DatasetExplorer: Render Crash!", e);
    return (
      <div className="p-20 text-center" style={{ height: '100vh', background: '#111', color: 'white' }}>
        <h1 className="text-orange mb-4">Component Crash!</h1>
        <p className="mb-4">{e.message}</p>
        <button onClick={() => window.location.reload()} className="p-2 bg-cyan text-black rounded">Reload Page</button>
      </div>
    );
  }
};
