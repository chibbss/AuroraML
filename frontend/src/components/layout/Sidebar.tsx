import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, FolderKanban, Settings, Database, Activity, LogOut } from 'lucide-react';
import { AuthService } from '../../api/auth';
import classNames from 'classnames';
import './Sidebar.css';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/projects', label: 'Projects', icon: FolderKanban },
  { path: '/datasets', label: 'Datasets', icon: Database },
  { path: '/monitoring', label: 'Monitoring', icon: Activity },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const [user, setUser] = React.useState<any>(null);

  React.useEffect(() => {
    AuthService.getMe().then(setUser).catch(console.error);
  }, []);

  const handleLogout = () => {
    AuthService.logout();
    navigate('/login');
  };
  return (
    <aside className="glass-panel sidebar" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="sidebar-header">
        <h2 className="gradient-text" style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0 }}>AuroraML</h2>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => classNames('nav-link', { active: isActive })}
            >
              <Icon size={20} className="nav-icon" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
      
      <div className="sidebar-footer">
        <div className="user-info">
          <div className="avatar">{user?.full_name?.charAt(0) || 'D'}</div>
          <div className="details">
            <span className="name">{user?.full_name || 'Developer'}</span>
            <span className="role">{user?.role || 'User'}</span>
          </div>
        </div>
        <button className="logout-button-compact" onClick={handleLogout} title="Logout">
          <LogOut size={18} />
        </button>
      </div>
    </aside>
  );
};
