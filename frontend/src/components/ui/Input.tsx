import React from 'react';
import classNames from 'classnames';
import './Input.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, icon, ...props }, ref) => {
    return (
      <div className="input-wrapper">
        {label && <label className="input-label">{label}</label>}
        <div className="input-container">
          {icon && <div className="input-icon">{icon}</div>}
          <input
            ref={ref}
            className={classNames('glass-input', { 'has-icon': icon, 'has-error': error }, className)}
            {...props}
          />
        </div>
        {error && <span className="input-error">{error}</span>}
      </div>
    );
  }
);
Input.displayName = 'Input';
