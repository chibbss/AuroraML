import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Activity, Trophy, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { JobsService, JobResponse } from '../api/jobs';
import './TrainingDashboard.css';

export const TrainingDashboard: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const fetchJobStatus = async () => {
      try {
        const data = await JobsService.getJob(jobId);
        setJob(data);

        // If job completes or fails, stop polling
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      } catch (err) {
        console.error("Error polling job", err);
        setError("Failed to fetch job status.");
      }
    };

    // Initial fetch
    fetchJobStatus();

    // Poll every 2 seconds
    intervalRef.current = window.setInterval(fetchJobStatus, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId]);

  if (error) {
    return <div className="p-8 text-center text-orange">{error}</div>;
  }

  if (!job) {
    return (
      <div className="flex-center" style={{ height: '60vh' }}>
        <Loader className="animate-spin text-cyan mb-4" size={40} />
        <p className="text-muted">Initializing AutoML Pipeline...</p>
      </div>
    );
  }

  const isComplete = job.status === 'completed';
  const progress = job.config?.progress;
  const progressPercent = progress ? Math.round((progress.completed / progress.total) * 100) : 0;
  
  const allResults = job.all_results || {};
  const sortedModels = Object.entries(allResults).sort((a, b) => {
    const scoreA = (a[1].f1_score || a[1].r2_score) || 0;
    const scoreB = (b[1].f1_score || b[1].r2_score) || 0;
    return scoreB - scoreA; // Descending
  });

  return (
    <div className="training-container">
      <div className="training-header">
        <div className="title-row">
          <div className="flex items-center gap-3">
            <div className={`status-beacon ${job.status}`}></div>
            <h2>AutoML Training Pipeline</h2>
          </div>
          {isComplete && (
            <div className="badge success-badge">
              <CheckCircle size={16} /> Training Complete
            </div>
          )}
        </div>
        <p className="subtitle text-muted flex gap-6 mt-2">
          <span><strong>Target:</strong> {job.target_column}</span>
          <span><strong>Type:</strong> {job.problem_type || 'Auto-detect'}</span>
        </p>
      </div>

      {/* Progress Section */}
      <Card className="progress-card mb-6" glowTheme={isComplete ? "green" : "cyan"}>
        <div className="progress-info mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {!isComplete && <Activity size={20} className="text-cyan animate-pulse" />}
            <span className="font-semibold text-lg text-primary">
              {isComplete ? "Finalizing Leaderboard" : `Evaluating ${progress?.current || 'models'}...`}
            </span>
          </div>
          <span className="text-xl font-bold font-mono">
            {progressPercent}%
          </span>
        </div>
        
        <div className="progress-bar-container">
          <div 
            className={`progress-fill ${isComplete ? 'completed' : 'active'}`}
            style={{ width: `${isComplete ? 100 : progressPercent}%` }}
          />
        </div>
        
        {!isComplete && progress && (
          <p className="text-xs text-muted text-right mt-2">
            Completed {progress.completed} of {progress.total} algorithms
          </p>
        )}
      </Card>

      {/* Live Leaderboard */}
      <Card className="leaderboard-card mb-6">
        <div className="flex items-center gap-3 mb-6 p-4 border-b border-glass">
          <Trophy className="text-purple" size={24} />
          <h3 className="m-0 text-xl font-medium">Live Leaderboard</h3>
        </div>

        <div className="table-wrapper">
          <table className="aurora-table animated-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Algorithm</th>
                <th>Primary Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.length > 0 ? (
                sortedModels.map(([modelName, metrics], index) => {
                  const isTop = index === 0;
                  const score = (metrics.f1_score || metrics.r2_score || 0).toFixed(4);
                  const isError = !!metrics.error;

                  return (
                    <tr key={modelName} className={`row-appear ${isTop ? 'top-rank' : ''}`}>
                      <td>
                        {isTop ? <Trophy size={16} className="text-orange" /> : `#${index + 1}`}
                      </td>
                      <td className="font-medium capitalize">{modelName.replace('_', ' ')}</td>
                      <td>
                        <span className={`score-badge ${isTop ? 'gold' : ''}`}>
                           {isError ? 'N/A' : score}
                        </span>
                      </td>
                      <td>
                        {isError ? (
                          <span className="flex items-center gap-1 text-orange text-sm"><AlertCircle size={14} /> Failed</span>
                        ) : (
                          <span className="flex items-center gap-1 text-green text-sm"><CheckCircle size={14} /> Done</span>
                        )}
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr>
                  <td colSpan={4} className="text-center text-muted p-8">
                    Waiting for first model to complete evaluation...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {isComplete && (
         <div className="completion-action animate-slide-up bg-glass border border-cyan/30 rounded-xl p-8 text-center mt-8">
            <h2 className="text-2xl mb-2">Training Successful! 🎉</h2>
            <p className="text-muted mb-6">The best performing model was the <strong>{job.best_model_type?.replace('_', ' ')}</strong> with a score of {job.best_score?.toFixed(4)}.</p>
            {job.best_model_id && (
              <Button 
                size="lg" 
                onClick={() => navigate(`/projects/${job.project_id}/models/${job.best_model_id}/evaluation`)} 
                className="w-64 max-w-full"
              >
                View Evaluation & Deployment
              </Button>
            )}
         </div>
      )}
    </div>
  );
};
