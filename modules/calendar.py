import core
import datetime
import ulid
import asyncio
import json

structure = core.config.get_module_structure()
channels = []
for name, data in structure.items():
    if data.get("metadata", {}).get("type") == "channel":
        channels.append(name)


class Calendar(core.module.Module):
    """
    Read-only calendar access by default.

    Use list_events, get_next_event, and search_events to answer questions about
    the calendar.

    Only use add_event, edit_event, or delete_event when the user explicitly
    asks to create, add, update, edit, move, reschedule, delete, cancel, or
    remove a calendar event.

    Requests to check, inspect, list, compare, cross-reference, verify,
    summarize, audit, or find discrepancies are not permission to modify the
    calendar.

    If a discrepancy is found, report it only. Do not fix it automatically
    unless the user explicitly asks for that calendar change.
    """

    settings = {
        "insert_system_prompt": {
            "description": "Whether to add upcoming calendar events to the system prompt as read-only context.",
            "default": True
        },
        "range": {
            "type": "date",
            "description": "How many days ahead the AI can see calendar events in the system prompt.",
            "default": 7
        },
        "notifications": {
            "description": "Whether to receive notifications about upcoming events.",
            "default": True
        },
        "notification_channel": {
            "type": "select",
            "default": "telegram",
            "description": "Which channel to send calendar notifications to.",
            "options": {name: f"Send notifications via {name}" for name in channels}
        },
        "notification_window": {
            "description": "Amount of minutes in advance you should be notified. Set this to 0 to be notified at the event time.",
            "default": 30
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.events = core.storage.StorageList("calendar", "json")
        self._notification_tasks = {}

    async def on_ready(self):
        for event in self.events:
            if event.get("notify"):
                await self._schedule_notification(event)

    def _new_id(self):
        try:
            return str(ulid.ULID())
        except Exception:
            try:
                return str(ulid.new())
            except Exception:
                return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    def _now(self):
        return datetime.datetime.now().replace(microsecond=0)

    def _parse_date(self, value):
        return datetime.datetime.fromisoformat(value)

    def _format_event(self, event):
        return {
            "id": event.get("id"),
            "title": event.get("title"),
            "date": event.get("date"),
            "notify": event.get("notify", False),
            "notify_channel": event.get("notify_channel")
        }

    def _json_result(self, payload, success=True):
        return self.result(json.dumps(payload, indent=2), success=success)

    def _get_default_notification_channel(self):
        return self.config.get("notification_channel")

    def _get_default_notification_window(self):
        return int(self.config.get("notification_window", default=30))

    async def _get_event_by_id(self, id: str):
        if not id:
            return -1

        for index, event in enumerate(self.events):
            if str(event.get("id", "")).strip() == str(id).strip():
                return index

        return -1

    async def _get_events_between(self, start, end, query=None):
        matches = []
        query_text = query.lower().strip() if query else None

        for event in self.events:
            try:
                event_date = self._parse_date(event["date"])
            except Exception:
                continue

            if event_date < start or event_date > end:
                continue

            if query_text:
                title = str(event.get("title", "")).lower()
                event_id = str(event.get("id", "")).lower()

                if query_text not in title and query_text not in event_id:
                    continue

            matches.append(event)

        matches.sort(key=lambda item: item.get("date", ""))
        return matches

    async def _get_upcoming_events(self, days_after=None, query=None):
        if days_after is None:
            days_after = int(self.config.get("range", default=7))

        now = self._now()
        end = now + datetime.timedelta(days=int(days_after))
        return await self._get_events_between(now, end, query=query)

    async def _get_events_in_prompt_range(self):
        days_after = int(self.config.get("range", default=7))
        now = self._now()
        end = now + datetime.timedelta(days=days_after)
        return await self._get_events_between(now, end)

    async def on_system_prompt(self):
        if not self.config.get("insert_system_prompt", default=True):
            return None

        matches = await self._get_events_in_prompt_range()
        if not matches:
            return None

        output = [
            "Calendar events below are read-only context.",
            "Use this calendar context to answer questions only.",
            "Do not create, edit, delete, move, reschedule, reconcile, or correct calendar events unless the user explicitly asks for that calendar change in their latest message.",
            "Checking, comparing, cross-referencing, verifying, summarizing, auditing, or finding discrepancies is not permission to modify calendar data.",
            "If calendar data conflicts with another source, report the discrepancy only."
        ]

        for event in matches:
            output.append(f"{event.get('id')}: on {event['date']}: {event['title']}")

        return "\n".join(output)

    async def list_events(self, days_before: int = 0, days_after: int = 30, query: str = None):
        """
        Read-only tool. Lists calendar events.

        Use this for checking, inspecting, reviewing, auditing, comparing,
        cross-referencing, or answering questions about the calendar.

        This tool never modifies calendar data.
        """
        now = self._now()
        start = now - datetime.timedelta(days=int(days_before or 0))
        end = now + datetime.timedelta(days=int(days_after or 30))

        matches = await self._get_events_between(start, end, query=query)

        return self._json_result({
            "read_only": True,
            "count": len(matches),
            "events": [self._format_event(event) for event in matches]
        })

    async def get_next_event(self, query: str = None, days_after: int = 365):
        """
        Read-only tool. Returns the next upcoming calendar event.

        Use this for questions like:
        - What is my next assignment?
        - When is my next due date?
        - What is the next event on my calendar?

        This tool never modifies calendar data.
        """
        matches = await self._get_upcoming_events(days_after=days_after, query=query)

        if not matches:
            return self._json_result({
                "read_only": True,
                "found": False,
                "message": "No upcoming matching events found."
            })

        return self._json_result({
            "read_only": True,
            "found": True,
            "event": self._format_event(matches[0])
        })

    async def search_events(self, query: str, days_before: int = 30, days_after: int = 365):
        """
        Read-only tool. Searches calendar events by title or ID.

        Use this when the user asks to find calendar entries, compare them
        against another source, or check for discrepancies.

        This tool never modifies calendar data.
        """
        if not query:
            return self._json_result({
                "read_only": True,
                "found": False,
                "message": "Search query is required."
            }, success=False)

        now = self._now()
        start = now - datetime.timedelta(days=int(days_before or 0))
        end = now + datetime.timedelta(days=int(days_after or 365))

        matches = await self._get_events_between(start, end, query=query)

        return self._json_result({
            "read_only": True,
            "query": query,
            "count": len(matches),
            "events": [self._format_event(event) for event in matches]
        })

    async def _schedule_notification(self, event: dict):
        if not event.get("notify"):
            return False

        event_id = event.get("id")
        if not event_id:
            return False

        existing_task = self._notification_tasks.get(event_id)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        try:
            event_time = self._parse_date(event["date"])
        except Exception as e:
            core.log_error("[CALENDAR] failed to parse event date", e)
            return False

        now = self._now()
        window_minutes = int(event.get("notification_window", self._get_default_notification_window()))
        window_seconds = window_minutes * 60
        notify_time = event_time - datetime.timedelta(seconds=window_seconds)

        if event_time <= now:
            index = await self._get_event_by_id(event_id)
            if index != -1:
                self.events[index]["notify"] = False
                self.events.save()
            return False

        delay = max(0, (notify_time - now).total_seconds())

        async def notification_worker():
            try:
                await asyncio.sleep(delay)

                index = await self._get_event_by_id(event_id)
                if index == -1:
                    return

                latest_event = self.events[index]
                if not latest_event.get("notify"):
                    return

                await self._notify_user(latest_event)

            except asyncio.CancelledError:
                return
            except Exception as e:
                core.log_error("[CALENDAR] notification worker failed", e)

        task = asyncio.create_task(notification_worker())
        self._notification_tasks[event_id] = task
        return True

    async def _notify_user(self, event: dict):
        if not event.get("notify"):
            return False

        channel_name = event.get("notify_channel") or self._get_default_notification_channel()
        channel = self.manager.channels.get(channel_name)

        if not channel:
            return False

        try:
            event_time = self._parse_date(event["date"])
        except Exception as e:
            core.log_error("[CALENDAR] failed to parse event date during notification", e)
            return False

        now = self._now()
        diff_seconds = (event_time - now).total_seconds()
        minutes_left = int(diff_seconds / 60)

        if minutes_left <= 0:
            notify_window_str = "now"
        elif minutes_left == 1:
            notify_window_str = "in 1 minute"
        else:
            notify_window_str = f"in {minutes_left} minutes"

        message = f"🔔 **Calendar**: {event['title']} is starting {notify_window_str}"
        await channel.push(message)

        if hasattr(channel, "context") and hasattr(channel.context, "chat"):
            await channel.context.chat.add({"role": "assistant", "content": message})

        index = await self._get_event_by_id(event["id"])
        if index != -1:
            self.events[index]["notify"] = False
            self.events.save()

        return True

    async def add_event(
        self,
        title: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        should_notify: bool = True,
        notify_channel: str = None
    ):
        """
        Write tool. Adds a calendar event.

        Only call this when the user explicitly asks to create, add, or schedule
        a calendar event.

        Do not call this for checking, comparing, cross-referencing, auditing,
        verifying, summarizing, or reporting discrepancies.
        """
        try:
            event_date = datetime.datetime(
                year=int(year),
                month=int(month),
                day=int(day),
                hour=int(hour),
                minute=int(minute)
            )
        except Exception as e:
            return self.result(f"Error: Invalid date: {e}", success=False)

        event = {
            "id": self._new_id(),
            "title": title,
            "date": event_date.isoformat(),
            "notify": bool(should_notify),
            "notify_channel": notify_channel or self._get_default_notification_channel()
        }

        self.events.append(event)
        self.events.save()

        if event.get("notify"):
            await self._schedule_notification(event)

        return self._json_result({
            "write_executed": True,
            "message": "Calendar event added.",
            "event": self._format_event(event)
        })

    async def edit_event(
        self,
        id: str,
        title: str = None,
        year: int = None,
        month: int = None,
        day: int = None,
        hour: int = None,
        minute: int = None,
        should_notify: bool = None,
        notify_channel: str = None
    ):
        """
        Write tool. Edits an existing calendar event.

        Only call this when the user explicitly asks to update, edit, change,
        move, or reschedule a calendar event.

        Do not call this for checking, comparing, cross-referencing, auditing,
        verifying, summarizing, or reporting discrepancies.
        """
        index = await self._get_event_by_id(id)
        if index < 0:
            return self.result("Error: Event with that ID does not exist.", success=False)

        old_event = dict(self.events[index])

        try:
            event_date = self._parse_date(old_event["date"])
            new_date = datetime.datetime(
                year=int(year) if year is not None else event_date.year,
                month=int(month) if month is not None else event_date.month,
                day=int(day) if day is not None else event_date.day,
                hour=int(hour) if hour is not None else event_date.hour,
                minute=int(minute) if minute is not None else event_date.minute
            )
        except Exception as e:
            return self.result(f"Error: Invalid updated date: {e}", success=False)

        new_event = dict(old_event)

        if title is not None:
            new_event["title"] = title

        new_event["date"] = new_date.isoformat()

        if should_notify is not None:
            new_event["notify"] = bool(should_notify)

        if notify_channel is not None:
            new_event["notify_channel"] = notify_channel

        self.events[index] = new_event
        self.events.save()

        existing_task = self._notification_tasks.get(id)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        if new_event.get("notify"):
            await self._schedule_notification(new_event)

        return self._json_result({
            "write_executed": True,
            "message": "Calendar event edited.",
            "before": self._format_event(old_event),
            "after": self._format_event(new_event)
        })

    async def delete_event(self, id: str):
        """
        Write tool. Deletes a calendar event.

        Only call this when the user explicitly asks to delete, remove, or
        cancel a calendar event.

        Do not call this for checking, comparing, cross-referencing, auditing,
        verifying, summarizing, or reporting discrepancies.
        """
        index = await self._get_event_by_id(id)
        if index < 0:
            return self.result("Error: Event with that ID does not exist.", success=False)

        event = dict(self.events[index])

        existing_task = self._notification_tasks.get(id)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        self.events.pop(index)
        self.events.save()

        return self._json_result({
            "write_executed": True,
            "message": "Calendar event deleted.",
            "event": self._format_event(event)
        })