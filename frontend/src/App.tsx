import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { Projects } from './pages/Projects';
import { ProjectDetails } from './pages/ProjectDetails';
import { Datasets } from './pages/Datasets';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { DataProfiler } from './pages/DataProfiler';
import { TrainingDashboard } from './pages/TrainingDashboard';
import { EvaluationDashboard } from './pages/EvaluationDashboard';
import { Monitoring } from './pages/Monitoring';
import { Settings } from './pages/Settings';
import { Login } from './pages/Login';
import { AuthService } from './api/auth';

const AuthGuard: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuth = AuthService.isAuthenticated();
  if (!isAuth) return <Navigate to="/login" replace />;
  return <>{children}</>;
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route 
          path="/*" 
          element={
            <AuthGuard>
              <MainLayout>
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/projects" element={<Projects />} />
                  <Route path="/projects/:id" element={<ProjectDetails />} />
                  <Route path="/projects/:id/datasets/:datasetId/profile" element={<DataProfiler />} />
                  <Route path="/projects/:id/jobs/:jobId/dashboard" element={<TrainingDashboard />} />
                  <Route path="/projects/:id/models/:modelId/evaluation" element={<EvaluationDashboard />} />
                  <Route path="/datasets" element={<Datasets />} />
                  <Route path="/datasets/:datasetId/explorer" element={<DatasetExplorer />} />
                  <Route path="/monitoring" element={<Monitoring />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </MainLayout>
            </AuthGuard>
          } 
        />
      </Routes>
    </Router>
  );
}

export default App;
