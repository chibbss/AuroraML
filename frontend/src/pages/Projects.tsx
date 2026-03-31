import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Plus, Search, FolderKanban, X, Trash2, ExternalLink } from 'lucide-react';
import { ProjectService, Project } from '../api/projects';
import { Dropdown } from '../components/ui/Dropdown';
import './Projects.css';

export const Projects: React.FC = () => {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    setIsLoading(true);
    try {
      const res = await ProjectService.getProjects();
      setProjects(res.projects || []);
    } catch (err) {
      console.error('Failed to fetch projects', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteProject = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this project? All associated datasets and models will be permanently removed.')) return;
    try {
      await ProjectService.deleteProject(id);
      fetchProjects();
    } catch (err) {
      console.error('Failed to delete project', err);
    }
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;

    setIsCreating(true);
    try {
      await ProjectService.createProject({
        name: newProjectName,
        description: newProjectDesc
      });
      setIsModalOpen(false);
      setNewProjectName('');
      setNewProjectDesc('');
      fetchProjects();
    } catch (err) {
      console.error('Failed to create project', err);
    } finally {
      setIsCreating(false);
    }
  };

  const filteredProjects = projects.filter(p => 
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    (p.description && p.description.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="projects-container">
      <div className="projects-header">
        <div>
          <h2>ML Projects</h2>
          <p className="subtitle">Manage your automated machine learning pipelines</p>
        </div>
        <Button 
          leftIcon={<Plus size={18} />} 
          onClick={() => setIsModalOpen(true)}
        >
          New Project
        </Button>
      </div>

      <Card className="projects-table-card" padding="none">
        <div className="table-controls">
          <Input 
            placeholder="Search projects by name..." 
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
                <th>Project Name</th>
                <th>Description</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={5} className="text-center py-8">Loading projects...</td>
                </tr>
              ) : filteredProjects.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-8 empty-state">
                    <FolderKanban size={48} className="empty-icon" />
                    <p>No projects found. Create one to get started.</p>
                  </td>
                </tr>
              ) : (
                filteredProjects.map(project => (
                  <tr key={project.id} className="clickable-row" onClick={() => navigate(`/projects/${project.id}`)}>
                    <td>
                      <div className="project-name-cell">
                        <div className="p-icon"><FolderKanban size={16} /></div>
                        {project.name}
                      </div>
                    </td>
                    <td className="text-muted">{project.description || 'No description'}</td>
                    <td>
                      <span className="status-badge status-active">Active</span>
                    </td>
                    <td className="text-muted">{new Date(project.created_at).toLocaleDateString()}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      <Dropdown 
                        items={[
                          { label: 'Open Project', icon: <ExternalLink size={14} />, onClick: () => navigate(`/projects/${project.id}`) },
                          { label: 'Delete Project', icon: <Trash2 size={14} />, variant: 'danger', onClick: () => handleDeleteProject(project.id) }
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

      {/* Create Project Modal */}
      {isModalOpen && (
        <div className="modal-overlay">
          <Card className="modal-content" glowTheme="cyan">
            <div className="modal-header">
              <h3>Create New ML Project</h3>
              <button className="close-btn" onClick={() => setIsModalOpen(false)}>
                <X size={20} />
              </button>
            </div>
            <form onSubmit={handleCreateProject} className="modal-body">
              <Input 
                label="Project Name" 
                placeholder="e.g. Demand Forecasting" 
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                required
              />
              <div className="input-wrapper">
                <label className="input-label">Description (Optional)</label>
                <textarea 
                  className="glass-input textarea" 
                  placeholder="Describe the goal of this ML project..."
                  rows={3}
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                />
              </div>
              <div className="modal-footer">
                <Button type="button" variant="ghost" onClick={() => setIsModalOpen(false)}>Cancel</Button>
                <Button type="submit" isLoading={isCreating}>Create Project</Button>
              </div>
            </form>
          </Card>
        </div>
      )}
    </div>
  );
};
