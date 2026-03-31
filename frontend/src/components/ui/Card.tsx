import React from 'react';
import classNames from 'classnames';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  glowTheme?: 'none' | 'green' | 'purple' | 'cyan';
}

export const Card: React.FC<CardProps> = ({ 
  className, 
  children, 
  padding = 'md',
  glowTheme = 'none',
  ...props 
}) => {
  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const glowClass = glowTheme !== 'none' ? `glow-${glowTheme}` : '';

  return (
    <div 
      className={classNames('glass-panel', paddingClasses[padding], glowClass, className)} 
      style={{ padding: padding === 'md' ? '24px' : padding === 'sm' ? '16px' : padding === 'lg' ? '32px' : 0 }}
      {...props}
    >
      {children}
    </div>
  );
};
