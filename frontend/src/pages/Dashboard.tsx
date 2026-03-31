import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Plus, Database, Activity, Server, Zap } from 'lucide-react';
import { ProjectService, Project } from '../api/projects';
import { AuthService, CurrentUser } from '../api/auth';
import { ApiClient } from '../api/client';
import './Dashboard.css';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<any>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [apiHealth, setApiHealth] = useState<string>('Checking...');
  const [isLoading, setIsLoading] = useState(true);
  const [userName, setUserName] = useState<string>('Developer');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // Fetch user info for the greeting
        AuthService.getMe().then((me: CurrentUser) => {
          if (me && me.full_name) {
            setUserName(me.full_name.split(' ')[0]);
          }
        }).catch(console.error);

        // Fetch everything in parallel
        const [statsData, projectsRes, healthRes] = await Promise.all([
          ApiClient.get<any>('/dashboard/stats').catch(() => null),
          ProjectService.getProjects().catch(() => ({ projects: [], total: 0 })),
          ProjectService.getSystemHealth().catch(() => ({ status: 'offline', version: 'unknown' }))
        ]);
        
        if (statsData) setStats(statsData);
        setProjects(projectsRes.projects || []);
        setApiHealth(healthRes.status === 'healthy' ? 'Online' : 'Offline');
      } catch (err) {
        console.error("Dashboard data fetch failed", err);
        setApiHealth('Offline');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  return (
    <div className="dashboard-container">
      {/* Welcome Banner */}
      <Card className="welcome-banner" glowTheme="purple">
        <div className="banner-content">
          <h1 className="banner-title">Welcome back, {userName}! 🧪</h1>
          <p className="banner-subtitle">Your AutoML models are performing at their peak. You have {stats?.projects_count ?? projects.length} active projects.</p>
          <div className="banner-actions">
            <Button 
                leftIcon={<Plus size={18} />}
                onClick={() => navigate('/projects')}
            >
                New Project
            </Button>
            <Button 
                variant="secondary" 
                leftIcon={<Database size={18} />}
                onClick={() => navigate('/datasets')}
            >
                View Datasets
            </Button>
          </div>
        </div>
        <div className="banner-illustration">
          <div className="floating-orb green"></div>
          <div className="floating-orb blue"></div>
        </div>
      </Card>

      {/* Stats Grid */}
      <div className="stats-grid">
        <Card className="stat-card">
          <div className="stat-header">
            <span className="stat-title">Active Projects</span>
            <div className="stat-icon-wrapper purple">
              <Activity size={20} />
            </div>
          </div>
          <div className="stat-value">{isLoading ? '-' : (stats?.projects_count ?? projects.length)}</div>
          <div className="stat-trend positive">Real-time stats</div>
        </Card>

        <Card className="stat-card">
          <div className="stat-header">
            <span className="stat-title">Models Deployed</span>
            <div className="stat-icon-wrapper cyan">
              <Server size={20} />
            </div>
          </div>
          <div className="stat-value">{isLoading ? '-' : (stats?.models_count ?? 0)}</div>
          <div className="stat-trend neutral">Production</div>
        </Card>

        <Card className="stat-card">
          <div className="stat-header">
            <span className="stat-title">Training Jobs</span>
            <div className="stat-icon-wrapper green">
              <Zap size={20} />
            </div>
          </div>
          <div className="stat-value">{isLoading ? '-' : (stats?.jobs_count ?? 0)}</div>
          <div className="stat-trend">
            {stats?.running_jobs_count || 0} Running
          </div>
        </Card>
      </div>

      {/* Main Content Split */}
      <div className="dashboard-split">
        <Card className="recent-projects">
          <h3>Recent Projects</h3>
          <div className="project-list">
            {isLoading ? (
              <div className="project-item"><p>Loading projects...</p></div>
            ) : projects.length === 0 ? (
              <div className="project-item"><p>No projects created yet.</p></div>
            ) : (
              projects.slice(0, 3).map(p => (
                <div 
                    className="project-item clickable-row" 
                    key={p.id}
                    onClick={() => navigate(`/projects/${p.id}`)}
                >
                  <div className="project-info">
                    <h4>{p.name}</h4>
                    <span>{new Date(p.updated_at).toLocaleDateString()}</span>
                  </div>
                  <div className="project-status status-active">Active</div>
                </div>
              ))
            )}
          </div>
          <Button 
            variant="ghost" 
            style={{ width: '100%', marginTop: '16px' }}
            onClick={() => navigate('/projects')}
          >
            View All Projects
          </Button>
        </Card>

        <Card className="system-health">
          <h3>System Health</h3>
          <div className="health-metrics">
            <div className="metric">
              <div className="metric-header">
                <span>API Status</span>
                <span className={apiHealth === 'Online' ? 'text-green' : 'text-orange'}>{apiHealth}</span>
              </div>
              <div className="progress-bar"><div className={`fill ${apiHealth === 'Online' ? 'green' : 'orange'}`} style={{ width: '100%' }}></div></div>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>Database</span>
                <span className="text-green">SQLite Mode</span>
              </div>
              <div className="progress-bar"><div className="fill green" style={{ width: '100%' }}></div></div>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>Storage (Local)</span>
                <span className="text-cyan">{stats?.storage_usage_pct || 0}%</span>
              </div>
              <div className="progress-bar"><div className="fill cyan" style={{ width: `${stats?.storage_usage_pct || 0}%` }}></div></div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
