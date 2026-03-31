import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { AuthService } from '../api/auth';
import { Shield, Mail, Lock, User as UserIcon, ArrowRight } from 'lucide-react';
import './Login.css';

export const Login: React.FC = () => {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      if (isLogin) {
        await AuthService.login({ username: email, password });
      } else {
        await AuthService.register({ email, password, full_name: fullName });
        // After register, log them in
        await AuthService.login({ username: email, password });
      }
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-background">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <Card className="login-card" glowTheme={isLogin ? "cyan" : "purple"}>
        <div className="login-header">
          <div className="logo-icon">
            <Shield size={32} />
          </div>
          <h2 className="gradient-text">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
          <p className="text-muted">
            {isLogin 
              ? 'Enter your credentials to access AuroraML' 
              : 'Sign up to start building automated ML pipelines'}
          </p>
        </div>

        {error && <div className="error-alert">{error}</div>}

        <form onSubmit={handleSubmit} className="login-form">
          {!isLogin && (
            <Input 
              label="Full Name"
              placeholder="Emmanuel Ochiba"
              icon={<UserIcon size={18} />}
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          )}

          <Input 
            label="Email Address"
            type="email"
            placeholder="name@company.com"
            icon={<Mail size={18} />}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <Input 
            label="Password"
            type="password"
            placeholder="••••••••"
            icon={<Lock size={18} />}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <Button 
            type="submit" 
            className="w-full mt-4" 
            isLoading={isLoading}
            rightIcon={<ArrowRight size={18} />}
          >
            {isLogin ? 'Sign In' : 'Create Account'}
          </Button>
        </form>

        <div className="login-footer">
          <span>{isLogin ? "Don't have an account?" : "Already have an account?"}</span>
          <button 
            className="toggle-btn" 
            onClick={() => setIsLogin(!isLogin)}
            type="button"
          >
            {isLogin ? 'Create one now' : 'Sign in here'}
          </button>
        </div>
      </Card>
    </div>
  );
};
