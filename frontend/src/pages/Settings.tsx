import React from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { User, Key, Server, Database } from 'lucide-react';
import './Settings.css';

export const Settings: React.FC = () => {
  return (
    <div className="settings-container">
      <div className="page-header">
        <div>
          <h2>Platform Settings</h2>
          <p className="subtitle">Configure your AuroraML environment and integrations</p>
        </div>
        <Button>Save Changes</Button>
      </div>

      <div className="settings-grid">
        <div className="settings-nav">
          <div className="nav-item active"><User size={18} /> Profile</div>
          <div className="nav-item"><Key size={18} /> API Keys</div>
          <div className="nav-item"><Server size={18} /> Compute Resources</div>
          <div className="nav-item"><Database size={18} /> External Databases</div>
        </div>

        <div className="settings-content">
          <Card className="settings-card">
            <h3>Profile Information</h3>
            <p className="text-muted mb-6">Update your account details and preferences.</p>
            
            <div className="form-group">
              <Input label="Full Name" defaultValue="Emmanuel" />
              <Input label="Email Address" defaultValue="emmanuel@example.com" type="email" />
            </div>
            
            <div className="form-group">
              <Input label="Role" defaultValue="Administrator" disabled />
            </div>
          </Card>

          <Card className="settings-card mt-6">
            <h3>API Configuration</h3>
            <p className="text-muted mb-6">Manage endpoints for external integrations.</p>
            
            <Input label="FastAPI Backend URL" defaultValue="http://127.0.0.1:8000/api/v1" />
            <Input label="MinIO Storage Endpoint" defaultValue="http://127.0.0.1:9000" className="mt-4" />
          </Card>
        </div>
      </div>
    </div>
  );
};
