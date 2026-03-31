import { ApiClient } from './client';

export interface NotificationItem {
  id: string;
  user_id: string;
  project_id?: string | null;
  job_id?: string | null;
  notification_type: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
  read_at?: string | null;
}

export interface NotificationListResponse {
  notifications: NotificationItem[];
  unread_count: number;
}

export const NotificationsService = {
  list: () => ApiClient.get<NotificationListResponse>('/notifications'),
  markRead: (ids: string[]) => ApiClient.post<NotificationListResponse>('/notifications/mark-read', { ids }),
  markAllRead: () => ApiClient.post<NotificationListResponse>('/notifications/mark-all-read', {}),
};
