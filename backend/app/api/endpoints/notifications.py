"""
Notifications Endpoints — List and mark notifications as read.
"""

from datetime import datetime, timezone
import time
import json
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from app.core.security import decode_access_token
from app.models.user import User
from app.models.notification import Notification
from app.schemas.notification import (
    NotificationListResponse,
    NotificationResponse,
    NotificationMarkReadRequest,
)

router = APIRouter(tags=["Notifications"])


def _get_user_from_token(token: str, db: Session) -> User:
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return user


@router.get("/notifications", response_model=NotificationListResponse)
def list_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(30)
        .all()
    )
    unread_count = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.is_read.is_(False))
        .count()
    )
    return NotificationListResponse(
        notifications=[NotificationResponse.model_validate(n) for n in notifications],
        unread_count=unread_count,
    )


@router.post("/notifications/mark-read", response_model=NotificationListResponse)
def mark_notifications_read(
    payload: NotificationMarkReadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not payload.ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No notification ids provided")

    now = datetime.now(timezone.utc)
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.id.in_(payload.ids),
    ).update(
        {"is_read": True, "read_at": now},
        synchronize_session=False,
    )
    db.commit()
    return list_notifications(current_user=current_user, db=db)


@router.post("/notifications/mark-all-read", response_model=NotificationListResponse)
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read.is_(False),
    ).update({"is_read": True, "read_at": now}, synchronize_session=False)
    db.commit()
    return list_notifications(current_user=current_user, db=db)


@router.get("/notifications/stream")
def stream_notifications(
    token: str = Query(...),
    since: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Stream notifications as Server-Sent Events for real-time updates."""
    user = _get_user_from_token(token, db)
    last_seen = None
    if since:
        try:
            last_seen = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            last_seen = None

    def event_generator():
        nonlocal last_seen
        while True:
            query = db.query(Notification).filter(Notification.user_id == user.id)
            if last_seen:
                query = query.filter(Notification.created_at > last_seen)
            new_notifications = query.order_by(Notification.created_at.asc()).limit(20).all()
            if new_notifications:
                last_seen = new_notifications[-1].created_at
                unread_count = (
                    db.query(Notification)
                    .filter(Notification.user_id == user.id, Notification.is_read.is_(False))
                    .count()
                )
                payload = {
                    "notifications": [NotificationResponse.model_validate(n).model_dump() for n in new_notifications],
                    "unread_count": unread_count,
                }
                yield f"event: notifications\ndata: {json.dumps(payload)}\n\n"
            time.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
