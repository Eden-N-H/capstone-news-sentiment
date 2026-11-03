"""
Collapse a per-ticker JSONL of Reddit posts into a deduplicated, sorted CSV.

Shared by reddit_pull.py and reddit_dump_filter.py so the two paths produce
byte-comparable output.

Why not pandas: both scripts used to do

    df = pd.read_json(path, lines=True).drop_duplicates(...).sort_values(...)

which materialises every post body for a ticker in memory at once, twice over
(parsed rows, then the frame). r/wallstreetbets alone can contribute hundreds
of thousands of posts with full selftext. This does two passes over the file
instead and never holds more than the offset index: a post ID and an integer
per row.
"""

import csv
import json

# The canonical output schema. Both scripts emit exactly these columns, in
# this order, so downstream code cannot tell which path produced a file.
CSV_COLUMNS = [
    "ticker",
    "platform",
    "subreddit",
    "scope",
    "post_id",
    "created_utc",
    "date",
    "title",
    "selftext",
    "score",
    "num_comments",
    "flair",
    "over_18",
    "url",
]


def collapse_jsonl_to_csv(jsonl_path: str, csv_path: str):
    """
    Write a deduplicated, chronologically sorted CSV from a JSONL file.

    Returns (n_rows, n_subreddits). Duplicate post_ids keep their first
    occurrence — which matters on resume, since a pull that was interrupted
    mid-window re-fetches that window and re-appends the posts it already had.

    Pass 1 records (created_utc, post_id, byte offset) for each unique post.
    Pass 2 seeks to the offsets in sorted order and writes them out.
    """
    index = []
    seen = set()

    with open(jsonl_path, "rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = row.get("post_id")
            if post_id is None or post_id in seen:
                continue
            seen.add(post_id)
            try:
                created = int(row.get("created_utc") or 0)
            except (TypeError, ValueError):
                created = 0
            index.append((created, offset))

    index.sort()
    subreddits = set()

    with open(jsonl_path, "rb") as fh, \
            open(csv_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for _, offset in index:
            fh.seek(offset)
            row = json.loads(fh.readline())
            # Newlines inside selftext are legal CSV when quoted, but they make
            # the file miserable to eyeball and break naive line-based tools.
            for key in ("title", "selftext"):
                if isinstance(row.get(key), str):
                    row[key] = row[key].replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
            if row.get("subreddit"):
                subreddits.add(str(row["subreddit"]).lower())

    return len(index), len(subreddits)


def count_by_subreddit(jsonl_path: str) -> dict:
    """Unique posts per subreddit, streamed. Used for the volume breakdown."""
    counts, seen = {}, set()
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            post_id = row.get("post_id")
            if post_id is None or post_id in seen:
                continue
            seen.add(post_id)
            sub = row.get("subreddit")
            if sub:
                counts[str(sub).lower()] = counts.get(str(sub).lower(), 0) + 1
    return counts


def date_span(csv_path: str):
    """(first_date, last_date) read off the sorted CSV without loading it."""
    first = last = None
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if first is None:
                first = row.get("date")
            last = row.get("date")
    return first, last
