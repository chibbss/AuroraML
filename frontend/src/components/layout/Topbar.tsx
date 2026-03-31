import React, { useEffect, useRef, useState } from 'react';
import { Bell, Search } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { NotificationsService, NotificationItem } from '../../api/notifications';
import './Topbar.css';

export const Topbar: React.FC = () => {
  const location = useLocation();
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [open, setOpen] = useState(false);
  const pollRef = useRef<number | null>(null);
  const streamRef = useRef<EventSource | null>(null);
  const [toast, setToast] = useState<NotificationItem | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  const titleMap: Record<string, string> = {
    '/dashboard': 'Dashboard Overview',
    '/projects': 'ML Projects',
    '/datasets': 'Datasets & Features',
    '/monitoring': 'Model Monitoring',
    '/settings': 'Platform Settings',
  };

  const getTitle = () => {
    const path = location.pathname;
    if (titleMap[path]) return titleMap[path];
    if (path.startsWith('/projects/')) return 'Project Details';
    return 'AuroraML';
  };

  const fetchNotifications = async () => {
    try {
      const data = await NotificationsService.list();
      setNotifications(data.notifications);
      setUnreadCount(data.unread_count);
    } catch (err) {
      console.error('Failed to fetch notifications', err);
    }
  };

  useEffect(() => {
    fetchNotifications();
    const token = localStorage.getItem('aurora_token');
    if (token) {
      const since = new Date().toISOString();
      const stream = new EventSource(`http://localhost:8000/api/v1/notifications/stream?token=${encodeURIComponent(token)}&since=${encodeURIComponent(since)}`);
      stream.addEventListener('notifications', (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent).data);
          if (payload.notifications?.length) {
            setNotifications((prev) => {
              const existingIds = new Set(prev.map((item) => item.id));
              const incoming = payload.notifications.filter((item: NotificationItem) => !existingIds.has(item.id));
              if (incoming.length > 0 && !open) {
                setToast(incoming[0]);
                if (toastTimerRef.current) {
                  window.clearTimeout(toastTimerRef.current);
                }
                toastTimerRef.current = window.setTimeout(() => setToast(null), 4500);
              }
              return [...incoming, ...prev];
            });
            setUnreadCount((prev) => (typeof payload.unread_count === 'number' ? payload.unread_count : prev));
          }
        } catch (err) {
          console.error('Notification stream parse failed', err);
        }
      });
      stream.onerror = () => {
        stream.close();
        streamRef.current = null;
        pollRef.current = window.setInterval(fetchNotifications, 30000);
      };
      streamRef.current = stream;
    } else {
      pollRef.current = window.setInterval(fetchNotifications, 30000);
    }
    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
      }
      if (streamRef.current) {
        streamRef.current.close();
      }
      if (toastTimerRef.current) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  const handleToggle = () => {
    const nextOpen = !open;
    setOpen(nextOpen);
  };

  const handleMarkAllRead = async () => {
    try {
      const data = await NotificationsService.markAllRead();
      setNotifications(data.notifications);
      setUnreadCount(data.unread_count);
    } catch (err) {
      console.error('Failed to mark notifications as read', err);
    }
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="page-title">{getTitle()}</h1>
      </div>
      
      <div className="topbar-right">
        <div className="search-box">
          <Search size={18} className="search-icon" />
          <input type="text" placeholder="Search projects or models..." />
        </div>
        
        <button className="icon-button glass-icon" onClick={handleToggle}>
          <Bell size={20} />
          {unreadCount > 0 && <span className="notification-dot"></span>}
        </button>
        {toast && !open && (
          <div className="notification-toast">
            <div className="notification-toast-title">{toast.title}</div>
            <div className="notification-toast-body">{toast.message}</div>
          </div>
        )}
        {open && (
          <div className="notification-panel">
            <div className="notification-header">
              <span>Notifications</span>
              {unreadCount > 0 && (
                <button className="notification-clear" onClick={handleMarkAllRead}>
                  Mark all read
                </button>
              )}
            </div>
            <div className="notification-list">
              {notifications.length === 0 ? (
                <div className="notification-empty">No notifications yet.</div>
              ) : (
                notifications.map((item) => (
                  <div key={item.id} className={`notification-item ${item.is_read ? '' : 'unread'}`}>
                    <div className="notification-title">{item.title}</div>
                    <div className="notification-body">{item.message}</div>
                    <div className="notification-time">
                      {new Date(item.created_at).toLocaleString()}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
