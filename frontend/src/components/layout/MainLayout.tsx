import React from 'react';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import './MainLayout.css';

export const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="layout-wrapper">
      <Sidebar />
      <main className="main-content">
        <Topbar />
        <div className="page-container">
          {children}
        </div>
      </main>
    </div>
  );
};
