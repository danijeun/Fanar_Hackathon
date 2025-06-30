from datetime import datetime, timedelta
import re

def resolve_natural_date(date_info):
    """
    Converts natural language date/time (like 'today', 'tomorrow', 'next week', etc.) to a concrete datetime string 'YYYY-MM-DD HH:MM'.
    Accepts either a string (e.g., 'today 17:00' or 'next day 9:00 PM') or a dict with 'date' and 'time'.
    Returns a string or None if cannot resolve.
    """
    now = datetime.now()
    original = date_info
    if isinstance(date_info, dict):
        date_str = (date_info.get('date') or '').strip().lower()
        time_str = (date_info.get('time') or '').strip()
        # Default time if not provided
        if not time_str:
            time_str = '09:00'
        # Handle natural language
        if date_str in ('', 'today'):
            date = now
        elif date_str == 'tomorrow':
            date = now + timedelta(days=1)
        elif date_str == 'next week':
            date = now + timedelta(days=7)
        elif date_str == 'next day':
            date = now + timedelta(days=1)
        elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except Exception:
                return None
        else:
            # Fallback: try to parse as a date
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except Exception:
                return None
        # Combine date and time
        try:
            hour, minute = map(int, re.findall(r'\d+', time_str)[:2])
            dt = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            print(f"[DEBUG] resolve_natural_date input: {original} => {dt.strftime('%Y-%m-%d %H:%M')}")
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            print(f"[DEBUG] resolve_natural_date input: {original} => {date.strftime('%Y-%m-%d 09:00')} (defaulted)")
            return date.strftime('%Y-%m-%d 09:00')
    elif isinstance(date_info, str):
        s = date_info.strip().lower()
        # Try to extract time (24h or 12h)
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', s)
        hour, minute = 9, 0
        found_time = False
        if time_match:
            try:
                hour = int(time_match.group(1))
                if time_match.group(2):
                    minute = int(time_match.group(2))
                if time_match.group(3):
                    if time_match.group(3) == 'pm' and hour < 12:
                        hour += 12
                    elif time_match.group(3) == 'am' and hour == 12:
                        hour = 0
                found_time = True
            except Exception:
                hour, minute = 9, 0
        # Date
        if 'today' in s:
            date = now
        elif 'tomorrow' in s:
            date = now + timedelta(days=1)
        elif 'next week' in s:
            date = now + timedelta(days=7)
        elif 'next day' in s:
            date = now + timedelta(days=1)
        else:
            # Try to parse as YYYY-MM-DD
            m = re.search(r'(\d{4}-\d{2}-\d{2})', s)
            if m:
                try:
                    date = datetime.strptime(m.group(1), '%Y-%m-%d')
                except Exception:
                    date = now
            else:
                date = now
        dt = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if found_time:
            print(f"[DEBUG] resolve_natural_date input: {original} => {dt.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"[DEBUG] resolve_natural_date input: {original} => {dt.strftime('%Y-%m-%d 09:00')} (defaulted)")
        return dt.strftime('%Y-%m-%d %H:%M')
    print(f"[DEBUG] resolve_natural_date input: {original} => None")
    return None 