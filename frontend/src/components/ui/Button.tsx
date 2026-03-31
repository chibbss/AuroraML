import React from 'react';
import classNames from 'classnames';
import './Button.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, leftIcon, rightIcon, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={classNames('btn', `btn-${variant}`, `btn-${size}`, { 'btn-loading': isLoading }, className)}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading && <span className="spinner"></span>}
        {!isLoading && leftIcon && <span className="btn-icon left">{leftIcon}</span>}
        <span className="btn-text">{children}</span>
        {!isLoading && rightIcon && <span className="btn-icon right">{rightIcon}</span>}
      </button>
    );
  }
);
Button.displayName = 'Button';
