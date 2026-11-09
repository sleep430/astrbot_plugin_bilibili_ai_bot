"""对话与视频动态记忆的轻量持久化。

v4 起优先使用插件独立目录下的 SQLite。新库只创建当前表结构；历史 JSON 仅在
SQLite 尚无业务数据时作为导入源，迁移后不随正常读写更新。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .constants import (
    DEFAULT_ACTIVE_CONVERSATION_LIMIT,
    DEFAULT_ACTIVE_CONVERSATION_MEMORY_LIMIT,
    DEFAULT_ACTIVE_CONVERSATION_TTL_SECONDS,
    DEFAULT_CHAT_STATE_LIMIT,
    DEFAULT_CHAT_STATE_TTL_SECONDS,
    DEFAULT_CONVERSATION_MEMORY_LIMIT,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TODAY_ACTIVITY_LIMIT,
    DEFAULT_USER_SUMMARY_LIMIT,
    DEFAULT_VIDEO_MEMORY_LIMIT,
)
from .memory_models import (
    ChatState,
    ConversationMemory,
    MemoryStoreStats,
    TodayActivity,
    VideoMemory,
    parse_datetime,
    prune_legacy_memory_payload,
)

_UNSET = object()


class MemoryStore:
    _SCHEMA_VERSION = 5
    _VIDEO_MEMORY_TABLES = (
        "video_profiles",
        "video_memory_stats",
        "video_visual_scenes",
        "video_ocr_segments",
        "video_comments",
        "video_comment_relations",
        "video_bangumi_links",
        "video_knowledge_links",
        "video_episodes",
        "video_episode_candidates",
    )

    def __init__(self, path: str | Path):
        legacy_path = Path(path)
        self._path = (
            legacy_path
            if legacy_path.suffix.lower() == ".sqlite"
            else legacy_path.with_suffix(".sqlite")
        )
        self._legacy_json_path = (
            legacy_path.with_suffix(".json")
            if legacy_path.suffix.lower() == ".sqlite"
            else legacy_path
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sqlite_ready = False
        self._legacy_loaded = False
        self._imported_from_json = False
        self._data: dict[str, Any] = self._empty_payload()
        self._summary_texts: dict[str, str] = {}
        self._users: dict[str, dict[str, Any]] = {}
        self._conversation_memories: dict[str, ConversationMemory] = {}
        self._conversation_active: dict[str, dict[str, Any]] = {}
        self._conversation_cursors: dict[str, dict[str, str]] = {}
        self._video_memories: dict[str, VideoMemory] = {}
        self._today_activity = TodayActivity()
        self._dynamics: dict[str, dict[str, Any]] = {}
        self._chat_states: dict[str, ChatState] = {}
        self._bangumi_follows: dict[str, dict[str, Any]] = {}
        self._bangumi_history: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._event_seq = 0
        self._init_sqlite()

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {
            "version": 5,
            "summary_texts": {},
            "users": {},
            "conversation_memories": {},
            "conversation_active": {},
            "conversation_cursors": {},
            "video_memories": {},
            "today_activity": {},
            "dynamics": {},
            "chat_states": {},
            "bangumi_follows": {},
            "bangumi_history": {},
            "events": [],
            "event_seq": 0,
        }

    # -- SQLite 基础设施 -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_sqlite(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_records (
                    kind TEXT NOT NULL,
                    id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    text TEXT,
                    updated_at REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    archived_at REAL,
                    PRIMARY KEY (kind, id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_records_kind_active
                    ON memory_records(kind, is_active, updated_at);
                CREATE INDEX IF NOT EXISTS idx_memory_records_kind_updated
                    ON memory_records(kind, updated_at);
                CREATE TABLE IF NOT EXISTS memory_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_created
                    ON memory_events(created_at);
                CREATE TABLE IF NOT EXISTS video_memories (
                    bvid TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    last_watched REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS video_profiles (
                    bvid TEXT PRIMARY KEY,
                    title TEXT,
                    up_name TEXT,
                    up_uid TEXT,
                    summary_text TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS video_memory_stats (
                    bvid TEXT PRIMARY KEY,
                    watch_count INTEGER NOT NULL DEFAULT 0,
                    comment_count INTEGER NOT NULL DEFAULT 0,
                    last_watched REAL NOT NULL DEFAULT 0,
                    last_commented REAL NOT NULL DEFAULT 0,
                    first_seen REAL NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS video_visual_scenes (
                    bvid TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, scene_id)
                );
                CREATE TABLE IF NOT EXISTS video_ocr_segments (
                    bvid TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    text TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, segment_id)
                );
                CREATE TABLE IF NOT EXISTS video_comments (
                    comment_id TEXT PRIMARY KEY,
                    bvid TEXT NOT NULL,
                    author_uid TEXT,
                    author_name TEXT,
                    content TEXT,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_video_comments_bvid
                    ON video_comments(bvid, created_at);
                CREATE TABLE IF NOT EXISTS video_comment_relations (
                    bvid TEXT NOT NULL,
                    relation_id TEXT NOT NULL,
                    parent_comment_id TEXT,
                    child_comment_id TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, relation_id)
                );
                CREATE TABLE IF NOT EXISTS video_bangumi_links (
                    bvid TEXT NOT NULL,
                    link_id TEXT NOT NULL,
                    bangumi_id TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, link_id)
                );
                CREATE TABLE IF NOT EXISTS video_knowledge_links (
                    bvid TEXT NOT NULL,
                    link_id TEXT NOT NULL,
                    knowledge_id TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, link_id)
                );
                CREATE TABLE IF NOT EXISTS video_episodes (
                    bvid TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    title TEXT,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, episode_id)
                );
                CREATE TABLE IF NOT EXISTS video_episode_candidates (
                    bvid TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    title TEXT,
                    score REAL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (bvid, candidate_id)
                );
                """
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(self._SCHEMA_VERSION),),
            )
            # 不能拿「memory_records 非空」判断新旧库：仅看过番剧/视频动态的实例
            # 有 history/follow 记录，users 和 conversations 却是空的。漏迁一次后
            # _load_sqlite 也不会再读旧 JSON，用户画像就静默丢光。
            user_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE kind='user'"
                ).fetchone()[0]
            )
            conversation_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE kind='conversation_memory'"
                ).fetchone()[0]
            )
            if user_count == 0 and conversation_count == 0:
                self._import_legacy_json(conn)
        self._sqlite_ready = True
        self._load_sqlite()

    def _import_legacy_json(self, conn: sqlite3.Connection) -> None:
        if not self._legacy_json_path.exists() or self._legacy_json_path.is_dir():
            return
        try:
            raw = json.loads(self._legacy_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"读取旧 JSON 记忆失败: {exc}")
            return
        if not isinstance(raw, dict):
            return
        self._load_payload(prune_legacy_memory_payload(raw))
        self._write_payload(conn, include_events=True)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('legacy_json_imported_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(time.time()),),
        )
        self._imported_from_json = True
        self._legacy_loaded = True

    def _load_sqlite(self) -> None:
        self._reset_runtime()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, id, payload, text FROM memory_records WHERE is_active=1"
            ).fetchall()
            for row in rows:
                payload = self._decode_json(row["payload"], {})
                if not isinstance(payload, dict):
                    continue
                kind, item_id = str(row["kind"]), str(row["id"])
                if kind == "user":
                    self._users[item_id] = payload
                elif kind == "summary_text":
                    self._summary_texts[item_id] = str(row["text"] or payload.get("text") or "")
                elif kind == "conversation_memory":
                    memory = ConversationMemory.from_dict(payload)
                    if memory.uid:
                        self._conversation_memories[memory.uid] = memory
                elif kind == "conversation_active":
                    self._conversation_active[item_id] = payload
                elif kind == "conversation_cursor":
                    self._conversation_cursors[item_id] = payload
                elif kind == "today_activity":
                    self._today_activity = TodayActivity.from_dict(payload)
                elif kind == "dynamic":
                    self._dynamics[item_id] = payload
                elif kind == "chat_state":
                    state = ChatState.from_dict(payload)
                    if state.chat_id:
                        self._chat_states[state.chat_id] = state
                elif kind == "bangumi_follow":
                    self._bangumi_follows[item_id] = payload
                elif kind == "bangumi_history":
                    self._bangumi_history[item_id] = payload
            for row in conn.execute("SELECT bvid, payload FROM video_memories"):
                memory = VideoMemory.from_dict(self._decode_json(row["payload"], {}))
                if memory.bvid:
                    self._video_memories[memory.bvid] = memory
            for row in conn.execute(
                "SELECT seq, event_type, payload, created_at FROM memory_events ORDER BY seq"
            ):
                payload = self._decode_json(row["payload"], {})
                self._events.append({
                    "seq": int(row["seq"]),
                    "event_type": row["event_type"],
                    "payload": payload if isinstance(payload, dict) else {},
                    "created_at": float(row["created_at"] or 0),
                })
            self._event_seq = int(self._events[-1]["seq"]) if self._events else 0
        self._prune()

    def _reset_runtime(self) -> None:
        self._data = self._empty_payload()
        self._summary_texts = {}
        self._users = {}
        self._conversation_memories = {}
        self._conversation_active = {}
        self._conversation_cursors = {}
        self._video_memories = {}
        self._today_activity = TodayActivity()
        self._dynamics = {}
        self._chat_states = {}
        self._bangumi_follows = {}
        self._bangumi_history = {}
        self._events = []
        self._event_seq = 0

    @staticmethod
    def _decode_json(value: Any, fallback: Any) -> Any:
        try:
            return json.loads(str(value))
        except (ValueError, TypeError):
            return fallback

    @staticmethod
    def _encode_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _record_timestamp(kind: str, payload: dict[str, Any]) -> float:
        for key in ("updated_at", "last_seen", "last_active_at", "last_watched", "created_at"):
            parsed = parse_datetime(payload.get(key))
            if parsed:
                return parsed
        return time.time()

    def _upsert_record(
        self,
        conn: sqlite3.Connection,
        kind: str,
        item_id: str,
        payload: dict[str, Any],
        *,
        text: str | None = None,
        updated_at: float | None = None,
        active: bool = True,
        archived_at: float | None = None,
    ) -> None:
        timestamp = float(updated_at if updated_at is not None else self._record_timestamp(kind, payload))
        conn.execute(
            """
            INSERT INTO memory_records(kind, id, payload, text, updated_at, is_active, archived_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, id) DO UPDATE SET
                payload=excluded.payload, text=excluded.text, updated_at=excluded.updated_at,
                is_active=excluded.is_active, archived_at=excluded.archived_at
            """,
            (kind, item_id, self._encode_json(payload), text, timestamp, int(active), archived_at),
        )

    def _delete_record(self, kind: str, item_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM memory_records WHERE kind=? AND id=?", (kind, item_id)
            )

    def _write_video_related(self, conn: sqlite3.Connection, memory: VideoMemory) -> None:
        now = time.time()
        for table in self._VIDEO_MEMORY_TABLES:
            # bvid 是这些子表的业务主键；按内存中的最终状态清掉旧明细，避免重复累积。
            conn.execute(f"DELETE FROM {table} WHERE bvid=?", (memory.bvid,))
        profile = memory.video_profile.to_dict()
        stats = memory.stats.to_dict()
        conn.execute(
            """
            INSERT INTO video_profiles(bvid, title, up_name, up_uid, summary_text, payload, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bvid) DO UPDATE SET title=excluded.title, up_name=excluded.up_name,
                up_uid=excluded.up_uid, summary_text=excluded.summary_text,
                payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (memory.bvid, profile.get("title"), profile.get("up_name"), profile.get("up_uid"),
             memory.summary_text, self._encode_json(profile), now),
        )
        conn.execute(
            """
            INSERT INTO video_memory_stats(
                bvid, watch_count, comment_count, last_watched, last_commented, first_seen, payload, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bvid) DO UPDATE SET watch_count=excluded.watch_count,
                comment_count=excluded.comment_count, last_watched=excluded.last_watched,
                last_commented=excluded.last_commented, first_seen=excluded.first_seen,
                payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (memory.bvid, memory.stats.watch_count, memory.stats.comment_count,
             memory.stats.last_watched, memory.stats.last_commented, memory.stats.first_seen,
             self._encode_json(stats), now),
        )
        for scene in memory.visual_scenes:
            conn.execute(
                "INSERT OR REPLACE INTO video_visual_scenes VALUES(?, ?, ?, ?, ?, ?)",
                (memory.bvid, scene.scene_id, scene.start_time, scene.end_time,
                 self._encode_json(scene.to_dict()), now),
            )
        for segment in memory.ocr_segments:
            conn.execute(
                "INSERT OR REPLACE INTO video_ocr_segments VALUES(?, ?, ?, ?, ?, ?, ?)",
                (memory.bvid, segment.segment_id, segment.start_time, segment.end_time,
                 segment.text, self._encode_json(segment.to_dict()), now),
            )
        for comment in memory.comments:
            if not comment.comment_id:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO video_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (comment.comment_id, memory.bvid, comment.author_uid, comment.author_name,
                 comment.content, self._encode_json(comment.to_dict()), comment.created_at, now),
            )
        for relation in memory.comment_relations:
            relation_id = relation.relation_id or f"{relation.parent_comment_id}:{relation.child_comment_id}"
            conn.execute(
                "INSERT OR REPLACE INTO video_comment_relations VALUES(?, ?, ?, ?, ?, ?)",
                (memory.bvid, relation_id, relation.parent_comment_id, relation.child_comment_id,
                 self._encode_json(relation.to_dict()), now),
            )
        for link in memory.bangumi_links:
            conn.execute(
                "INSERT OR REPLACE INTO video_bangumi_links VALUES(?, ?, ?, ?, ?)",
                (memory.bvid, link.link_id, link.bangumi_id, self._encode_json(link.to_dict()), now),
            )
        for link in memory.knowledge_links:
            conn.execute(
                "INSERT OR REPLACE INTO video_knowledge_links VALUES(?, ?, ?, ?, ?)",
                (memory.bvid, link.link_id, link.knowledge_id, self._encode_json(link.to_dict()), now),
            )
        for episode in memory.episodes:
            conn.execute(
                "INSERT OR REPLACE INTO video_episodes VALUES(?, ?, ?, ?, ?)",
                (memory.bvid, episode.episode_id, episode.title, self._encode_json(episode.to_dict()), now),
            )
        for candidate in memory.episode_candidates:
            conn.execute(
                "INSERT OR REPLACE INTO video_episode_candidates VALUES(?, ?, ?, ?, ?, ?)",
                (memory.bvid, candidate.candidate_id, candidate.title, candidate.score,
                 self._encode_json(candidate.to_dict()), now),
            )

    def _write_payload(self, conn: sqlite3.Connection, *, include_events: bool = False) -> None:
        """把当前内存快照写入事务；正常业务写入应使用单记录 upsert。"""
        for kind, mapping in (
            ("user", self._users),
            ("conversation_active", self._conversation_active),
            ("conversation_cursor", self._conversation_cursors),
            ("dynamic", self._dynamics),
            ("bangumi_follow", self._bangumi_follows),
            ("bangumi_history", self._bangumi_history),
        ):
            for item_id, payload in mapping.items():
                if isinstance(payload, dict):
                    self._upsert_record(conn, kind, item_id, payload)
        for uid, text in self._summary_texts.items():
            self._upsert_record(conn, "summary_text", uid, {"text": text}, text=text)
        for uid, memory in self._conversation_memories.items():
            self._upsert_record(conn, "conversation_memory", uid, memory.to_dict())
        self._upsert_record(conn, "today_activity", "today", self._today_activity.to_dict())
        for chat_id, state in self._chat_states.items():
            self._upsert_record(conn, "chat_state", chat_id, state.to_dict())
        for memory in self._video_memories.values():
            self._write_video(conn, memory)
        if include_events:
            conn.execute("DELETE FROM memory_events")
            for event in self._events:
                conn.execute(
                    "INSERT INTO memory_events(seq, event_type, payload, created_at) VALUES(?, ?, ?, ?)",
                    (int(event["seq"]), event["event_type"], self._encode_json(event["payload"]), event["created_at"]),
                )
            self._event_seq = int(self._events[-1]["seq"]) if self._events else 0

    # -- 公开 API：对话记忆 ----------------------------------------------

    def get(self, uid: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._users.get(uid)
            return dict(value) if isinstance(value, dict) else None

    def upsert(self, uid: str, data: dict[str, Any]) -> None:
        if not uid or not isinstance(data, dict):
            return
        with self._lock, self._connect() as conn:
            payload = dict(data)
            self._users[uid] = payload
            self._upsert_record(conn, "user", uid, payload)

    def update_fields(self, uid: str, **fields: Any) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            current = dict(self._users.get(uid) or {})
            current.update(fields)
            self._users[uid] = current
            self._upsert_record(conn, "user", uid, current)
            return dict(current)

    def remove(self, uid: str) -> bool:
        with self._lock, self._connect() as conn:
            existed = uid in self._users
            self._users.pop(uid, None)
            self._summary_texts.pop(uid, None)
            self._conversation_memories.pop(uid, None)
            self._conversation_active.pop(uid, None)
            self._conversation_cursors.pop(uid, None)
            conn.execute(
                "DELETE FROM memory_records WHERE id=? AND kind IN "
                "('user', 'summary_text', 'conversation_memory', 'conversation_active', 'conversation_cursor')",
                (uid,),
            )
            return existed

    def all_users(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {uid: dict(value) for uid, value in self._users.items()}

    def _load(self) -> None:
        """兼容旧调用方；SQLite 已在初始化时载入，运行时不重复读整库。"""
        return None

    def _load_payload(self, data: dict[str, Any]) -> None:
        self._reset_runtime()
        self._data = data
        self._summary_texts = {
            str(k): str(v) for k, v in (data.get("summary_texts") or {}).items()
        }
        self._users = dict(data.get("users") or {})
        self._conversation_memories = {
            uid: ConversationMemory.from_dict(value)
            for uid, value in (data.get("conversation_memories") or {}).items()
        }
        self._conversation_active = dict(data.get("conversation_active") or {})
        self._conversation_cursors = dict(data.get("conversation_cursors") or {})
        self._video_memories = {
            bvid: VideoMemory.from_dict(value)
            for bvid, value in (data.get("video_memories") or {}).items()
        }
        self._today_activity = TodayActivity.from_dict(data.get("today_activity") or {})
        self._dynamics = dict(data.get("dynamics") or {})
        self._chat_states = {
            chat_id: ChatState.from_dict(value)
            for chat_id, value in (data.get("chat_states") or {}).items()
        }
        self._bangumi_follows = dict(data.get("bangumi_follows") or {})
        self._bangumi_history = dict(data.get("bangumi_history") or {})
        self._events = list(data.get("events") or [])
        self._event_seq = int(data.get("event_seq") or 0)

    def _save(self) -> None:
        """兼容旧调用方的快照保存；仍使用事务，不写 JSON。"""
        with self._lock, self._connect() as conn:
            self._write_payload(conn, include_events=True)

    def export_snapshot(self) -> dict[str, Any]:
        """仅用于备份/迁移调试；日常提示词请走带 limit 的读取接口。"""
        with self._lock:
            return {
                "version": 5,
                "summary_texts": dict(self._summary_texts),
                "users": self.all_users(),
                "conversation_memories": {
                    uid: memory.to_dict() for uid, memory in self._conversation_memories.items()
                },
                "conversation_active": dict(self._conversation_active),
                "conversation_cursors": dict(self._conversation_cursors),
                "video_memories": {
                    bvid: memory.to_dict() for bvid, memory in self._video_memories.items()
                },
                "today_activity": self._today_activity.to_dict(),
                "dynamics": dict(self._dynamics),
                "chat_states": {key: state.to_dict() for key, state in self._chat_states.items()},
                "bangumi_follows": dict(self._bangumi_follows),
                "bangumi_history": dict(self._bangumi_history),
                "events": list(self._events),
                "event_seq": self._event_seq,
            }

    # 下面方法保持原 MemoryStore API；只是持久化从整份 JSON 改为单记录 SQLite。
    def get_summary_text(self, uid: str) -> str | None:
        return self._summary_texts.get(uid)

    def set_summary_text(self, uid: str, text: str) -> None:
        with self._lock, self._connect() as conn:
            self._summary_texts[uid] = text
            self._upsert_record(conn, "summary_text", uid, {"text": text}, text=text)

    def get_conversation_memory(self, uid: str) -> ConversationMemory | None:
        return self._conversation_memories.get(uid)

    def save_conversation_memory(self, memory: ConversationMemory) -> None:
        with self._lock, self._connect() as conn:
            self._conversation_memories[memory.uid] = memory
            self._upsert_record(conn, "conversation_memory", memory.uid, memory.to_dict())

    def list_conversation_memories(
        self, limit: int = 200, uid: str | None = None
    ) -> list[ConversationMemory]:
        with self._lock:
            values = list(self._conversation_memories.values())
            if uid is not None:
                values = [item for item in values if item.uid == uid]
            values.sort(key=lambda item: item.updated_at, reverse=True)
            return values[: max(1, limit)]

    def get_active_conversation(self, uid: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._conversation_active.get(uid) or {})

    def set_active_conversation(self, uid: str, data: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            payload = dict(data)
            self._conversation_active[uid] = payload
            self._upsert_record(conn, "conversation_active", uid, payload)

    def clear_active_conversation(self, uid: str) -> None:
        with self._lock, self._connect() as conn:
            self._conversation_active.pop(uid, None)
            conn.execute(
                "DELETE FROM memory_records WHERE kind='conversation_active' AND id=?", (uid,)
            )

    def get_conversation_cursor(self, uid: str) -> dict[str, str]:
        with self._lock:
            return dict(self._conversation_cursors.get(uid) or {})

    def set_conversation_cursor(self, uid: str, cursor: dict[str, str]) -> None:
        with self._lock, self._connect() as conn:
            payload = dict(cursor)
            self._conversation_cursors[uid] = payload
            self._upsert_record(conn, "conversation_cursor", uid, payload)

    def clear_conversation_cursor(self, uid: str) -> None:
        with self._lock, self._connect() as conn:
            self._conversation_cursors.pop(uid, None)
            conn.execute(
                "DELETE FROM memory_records WHERE kind='conversation_cursor' AND id=?", (uid,)
            )

    def archive_conversation(self, uid: str, memory: ConversationMemory) -> None:
        self.save_conversation_memory(memory)
        self.clear_active_conversation(uid)

    # -- 视频记忆 --------------------------------------------------------

    def get_video_memory(self, bvid: str) -> VideoMemory | None:
        return self._video_memories.get(bvid)

    def _write_video(self, conn: sqlite3.Connection, memory: VideoMemory) -> None:
        now = time.time()
        conn.execute(
            """
            INSERT INTO video_memories(bvid, payload, last_watched, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(bvid) DO UPDATE SET payload=excluded.payload,
                last_watched=excluded.last_watched, updated_at=excluded.updated_at
            """,
            (memory.bvid, self._encode_json(memory.to_dict()), memory.stats.last_watched, now),
        )
        self._write_video_related(conn, memory)

    def save_video_memory(self, memory: VideoMemory) -> None:
        with self._lock, self._connect() as conn:
            self._video_memories[memory.bvid] = memory
            self._write_video(conn, memory)

    def delete_video_memory(self, bvid: str) -> bool:
        with self._lock, self._connect() as conn:
            existed = bvid in self._video_memories
            self._video_memories.pop(bvid, None)
            conn.execute("DELETE FROM video_memories WHERE bvid=?", (bvid,))
            for table in self._VIDEO_MEMORY_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE bvid=?", (bvid,))
            conn.execute("DELETE FROM video_comments WHERE bvid=?", (bvid,))
            return existed

    def list_video_memories(self, limit: int = 200) -> list[VideoMemory]:
        with self._lock:
            values = sorted(
                self._video_memories.values(),
                key=lambda item: item.stats.last_watched,
                reverse=True,
            )
            return values[: max(1, limit)]

    def get_today_activity(self) -> TodayActivity:
        return self._today_activity

    def save_today_activity(self, activity: TodayActivity) -> None:
        with self._lock, self._connect() as conn:
            self._today_activity = activity
            self._upsert_record(conn, "today_activity", "today", activity.to_dict())

    def get_dynamic(self, dynamic_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._dynamics.get(dynamic_id)
            return dict(value) if value else None

    def list_dynamics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            values = sorted(
                self._dynamics.values(),
                key=lambda item: parse_datetime(item.get("created_at")) or 0,
                reverse=True,
            )
            return [dict(item) for item in values[: max(1, limit)]]

    def save_dynamic(self, dynamic_id: str, data: dict[str, Any]) -> None:
        if not dynamic_id:
            return
        with self._lock, self._connect() as conn:
            current = dict(self._dynamics.get(dynamic_id) or {})
            current.update(data)
            self._dynamics[dynamic_id] = current
            self._upsert_record(conn, "dynamic", dynamic_id, current)

    def remove_dynamic(self, dynamic_id: str) -> bool:
        with self._lock, self._connect() as conn:
            existed = dynamic_id in self._dynamics
            self._dynamics.pop(dynamic_id, None)
            conn.execute("DELETE FROM memory_records WHERE kind='dynamic' AND id=?", (dynamic_id,))
            return existed

    def get_chat_state(self, chat_id: str) -> ChatState | None:
        return self._chat_states.get(chat_id)

    def save_chat_state(self, state: ChatState) -> None:
        with self._lock, self._connect() as conn:
            self._chat_states[state.chat_id] = state
            self._upsert_record(conn, "chat_state", state.chat_id, state.to_dict())

    def get_bangumi_follow(self, season_id: int) -> dict[str, Any] | None:
        with self._lock:
            value = self._bangumi_follows.get(str(season_id))
            return dict(value) if value else None

    def save_bangumi_follow(self, season_id: int, data: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            payload = dict(data)
            self._bangumi_follows[str(season_id)] = payload
            self._upsert_record(conn, "bangumi_follow", str(season_id), payload)

    def list_bangumi_follows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._bangumi_follows.values()]

    def remove_bangumi_follow(self, season_id: int) -> bool:
        with self._lock, self._connect() as conn:
            existed = str(season_id) in self._bangumi_follows
            self._bangumi_follows.pop(str(season_id), None)
            conn.execute(
                "DELETE FROM memory_records WHERE kind='bangumi_follow' AND id=?",
                (str(season_id),),
            )
            return existed

    def get_bangumi_history(self, uid: int) -> dict[str, Any] | None:
        with self._lock:
            value = self._bangumi_history.get(str(uid))
            return dict(value) if value else None

    def save_bangumi_history(self, uid: int, data: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            payload = dict(data)
            self._bangumi_history[str(uid)] = payload
            self._upsert_record(conn, "bangumi_history", str(uid), payload)

    def list_bangumi_history(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            values = sorted(
                self._bangumi_history.values(),
                key=lambda item: parse_datetime(item.get("last_seen")) or 0,
                reverse=True,
            )
            return [dict(item) for item in values[: max(1, limit)]]

    # -- 事件、维护、统计 ------------------------------------------------

    def append_event(self, event_type: str, payload: dict[str, Any]) -> int:
        with self._lock, self._connect() as conn:
            created_at = time.time()
            cursor = conn.execute(
                "INSERT INTO memory_events(event_type, payload, created_at) VALUES(?, ?, ?)",
                (event_type, self._encode_json(payload), created_at),
            )
            self._event_seq = int(cursor.lastrowid)
            self._events.append({
                "seq": self._event_seq,
                "event_type": event_type,
                "payload": dict(payload),
                "created_at": created_at,
            })
            if len(self._events) > 1000:
                self._events = self._events[-1000:]
                conn.execute(
                    "DELETE FROM memory_events WHERE seq NOT IN "
                    "(SELECT seq FROM memory_events ORDER BY seq DESC LIMIT 1000)"
                )
            return self._event_seq

    def get_events_since(self, seq: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._events if int(item["seq"]) > seq][: max(1, limit)]

    def _prune(self) -> None:
        now = time.time()
        conversation_ttl = max(3600, int(DEFAULT_ACTIVE_CONVERSATION_TTL_SECONDS))
        chat_ttl = max(3600, int(DEFAULT_CHAT_STATE_TTL_SECONDS))
        expired_conversations = [
            uid for uid, value in self._conversation_active.items()
            if now - (parse_datetime(value.get("last_active_at")) or 0) > conversation_ttl
        ]
        expired_cursors = [
            uid for uid, value in self._conversation_cursors.items()
            if now - (parse_datetime(value.get("updated_at")) or 0) > conversation_ttl
        ]
        expired_chats = [
            chat_id for chat_id, state in self._chat_states.items()
            if now - (parse_datetime(state.last_active_at) or 0) > chat_ttl
        ]
        if len(self._conversation_active) > DEFAULT_ACTIVE_CONVERSATION_LIMIT:
            ordered = sorted(
                self._conversation_active.items(),
                key=lambda item: parse_datetime(item[1].get("last_active_at")) or 0,
            )
            expired_conversations.extend(
                uid for uid, _ in ordered[: len(ordered) - DEFAULT_ACTIVE_CONVERSATION_LIMIT]
                if uid not in expired_conversations
            )
        if len(self._conversation_memories) > DEFAULT_CONVERSATION_MEMORY_LIMIT:
            ordered = sorted(
                self._conversation_memories.items(), key=lambda item: item[1].updated_at
            )
            for uid, _ in ordered[: len(ordered) - DEFAULT_CONVERSATION_MEMORY_LIMIT]:
                self._conversation_memories.pop(uid, None)
                self._delete_record("conversation_memory", uid)
        if len(self._video_memories) > DEFAULT_VIDEO_MEMORY_LIMIT:
            ordered = sorted(
                self._video_memories.items(), key=lambda item: item[1].stats.last_watched
            )
            for bvid, _ in ordered[: len(ordered) - DEFAULT_VIDEO_MEMORY_LIMIT]:
                self._video_memories.pop(bvid, None)
                self.delete_video_memory(bvid)
        self._today_activity.items = self._today_activity.items[: DEFAULT_TODAY_ACTIVITY_LIMIT]
        if len(self._dynamics) > DEFAULT_MEMORY_LIMIT:
            ordered = sorted(
                self._dynamics.items(),
                key=lambda item: parse_datetime(item[1].get("created_at")) or 0,
                reverse=True,
            )
            for dynamic_id, _ in ordered[DEFAULT_MEMORY_LIMIT:]:
                self._dynamics.pop(dynamic_id, None)
                self._delete_record("dynamic", dynamic_id)
        if len(self._chat_states) > DEFAULT_CHAT_STATE_LIMIT:
            ordered = sorted(
                self._chat_states.items(), key=lambda item: item[1].last_active_at
            )
            expired_chats.extend(
                chat_id for chat_id, _ in ordered[: len(ordered) - DEFAULT_CHAT_STATE_LIMIT]
                if chat_id not in expired_chats
            )
        if len(self._bangumi_history) > DEFAULT_MEMORY_LIMIT:
            ordered = sorted(
                self._bangumi_history.items(),
                key=lambda item: parse_datetime(item[1].get("last_seen")) or 0,
                reverse=True,
            )
            for uid, _ in ordered[DEFAULT_MEMORY_LIMIT:]:
                self._bangumi_history.pop(uid, None)
                self._delete_record("bangumi_history", uid)
        if len(self._events) > 1000:
            self._events = self._events[-1000:]
        with self._lock, self._connect() as conn:
            for uid in expired_conversations:
                self._conversation_active.pop(uid, None)
                conn.execute(
                    "DELETE FROM memory_records WHERE kind='conversation_active' AND id=?", (uid,)
                )
            for uid in expired_cursors:
                self._conversation_cursors.pop(uid, None)
                conn.execute(
                    "DELETE FROM memory_records WHERE kind='conversation_cursor' AND id=?", (uid,)
                )
            for chat_id in expired_chats:
                self._chat_states.pop(chat_id, None)
                conn.execute(
                    "DELETE FROM memory_records WHERE kind='chat_state' AND id=?", (chat_id,)
                )
            self._upsert_record(conn, "today_activity", "today", self._today_activity.to_dict())
            if len(self._events) >= 1000:
                conn.execute(
                    "DELETE FROM memory_events WHERE seq NOT IN "
                    "(SELECT seq FROM memory_events ORDER BY seq DESC LIMIT 1000)"
                )

    def prune(self) -> None:
        with self._lock:
            self._prune()

    def stats(self) -> MemoryStoreStats:
        with self._lock, self._connect() as conn:
            active_records = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE is_active=1"
                ).fetchone()[0]
            )
            archived_records = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE is_active=0"
                ).fetchone()[0]
            )
            event_count = int(conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0])
        try:
            db_size = self._path.stat().st_size
        except OSError:
            db_size = 0
        with self._lock:
            return MemoryStoreStats(
                user_count=len(self._users),
                video_count=len(self._video_memories),
                dynamic_count=len(self._dynamics),
                bangumi_follow_count=len(self._bangumi_follows),
                bangumi_history_count=len(self._bangumi_history),
                active_conversation_count=len(self._conversation_active),
                conversation_memory_count=len(self._conversation_memories),
                chat_state_count=len(self._chat_states),
                event_count=event_count,
                active_record_count=active_records,
                archived_record_count=archived_records,
                db_size_bytes=db_size,
            )

    # -- 上下文裁剪（保持原行为） ----------------------------------------

    @staticmethod
    def trim_text(text: str, limit: int) -> str:
        text = " ".join(str(text or "").split())
        if limit <= 0:
            return ""
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    @classmethod
    def trim_dict(cls, value: Any, depth: int = 0, *, item_limit: int = 8) -> Any:
        if depth > 5:
            return None
        if isinstance(value, dict):
            result = {}
            for key, item in list(value.items())[:item_limit]:
                if item not in (None, "", [], {}):
                    result[str(key)] = cls.trim_dict(item, depth + 1, item_limit=item_limit)
            return result
        if isinstance(value, list):
            return [
                cls.trim_dict(item, depth + 1, item_limit=item_limit)
                for item in value[:item_limit]
            ]
        if isinstance(value, str):
            return cls.trim_text(value, 240)
        return value

    @staticmethod
    def _confidence_weight(value: Any) -> float:
        if isinstance(value, dict):
            value = value.get("confidence", 0.5)
        try:
            return float(value or 0.5)
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def _rank_dict_items(cls, values: Any, limit: int) -> list[tuple[str, Any]]:
        if not isinstance(values, dict):
            return []
        items = [(str(key), value) for key, value in values.items() if value not in (None, "", [], {})]
        items.sort(key=lambda item: cls._confidence_weight(item[1]), reverse=True)
        return items[: max(0, limit)]

    @classmethod
    def compact_profile(cls, raw: dict[str, Any] | None, *, max_chars: int = 650) -> dict[str, Any]:
        raw = raw if isinstance(raw, dict) else {}
        result: dict[str, Any] = {}
        for key in ("display_name", "sex", "persona", "identity_notes"):
            value = raw.get(key)
            if isinstance(value, list):
                result[key] = [cls.trim_text(item, 100) for item in value[:4] if str(item).strip()]
            elif value not in (None, ""):
                result[key] = cls.trim_text(str(value), 100 if key != "persona" else 180)
        likes = raw.get("likes") if isinstance(raw.get("likes"), dict) else {}
        dislikes = raw.get("dislikes") if isinstance(raw.get("dislikes"), dict) else {}
        for key, values, text_key, limit in (
            ("likes", likes, "preferences", 4),
            ("dislikes", dislikes, "dislikes", 2),
        ):
            ranked = cls._rank_dict_items(values, limit)
            if ranked:
                result[key] = [cls.trim_text(key, 80) for key, _ in ranked]
                if key == "likes":
                    result[text_key] = [cls.trim_text(key, 80) for key, _ in ranked]
        preference_signals = raw.get("preference_signals")
        if isinstance(preference_signals, dict):
            signals = cls._rank_dict_items(preference_signals, 4)
            if signals:
                result["preference_signals"] = {
                    key: cls.trim_dict(value, 1, item_limit=4) for key, value in signals
                }
        for key in ("anniversary", "nickname"):
            value = raw.get(key)
            if isinstance(value, dict):
                value = value.get("value") or value.get("date") or value.get("name")
            if value not in (None, ""):
                result[key] = cls.trim_text(str(value), 80)
        if raw.get("long_term_summary"):
            result["long_term_summary"] = cls.trim_text(str(raw["long_term_summary"]), 300)
        affinity = raw.get("affinity")
        if isinstance(affinity, dict):
            compact_affinity = {
                key: affinity[key]
                for key in ("score", "level", "last_interaction", "last_change_reason")
                if affinity.get(key) not in (None, "")
            }
            if compact_affinity:
                result["affinity"] = compact_affinity
        return json.loads(cls.trim_text(json.dumps(result, ensure_ascii=False), max_chars)) if False else result

    @classmethod
    def compact_video_memory(cls, memory: VideoMemory | None, *, max_chars: int = 1500) -> dict[str, Any]:
        if memory is None:
            return {}
        profile = memory.video_profile
        result: dict[str, Any] = {
            "bvid": memory.bvid,
            "title": cls.trim_text(profile.title, 100),
            "up_name": cls.trim_text(profile.up_name, 50),
            "summary_text": cls.trim_text(memory.summary_text, 360),
            "highlights": [cls.trim_text(item, 100) for item in memory.highlights[:4]],
            "watch_count": memory.stats.watch_count,
        }
        if memory.emotion:
            result["emotion"] = cls.trim_text(memory.emotion, 30)
        if profile.tags:
            result["tags"] = [cls.trim_text(item, 40) for item in profile.tags[:8]]
        if profile.desc:
            result["desc"] = cls.trim_text(profile.desc, 180)
        if memory.ocr_segments:
            result["ocr"] = [cls.trim_text(item.text, 90) for item in memory.ocr_segments[:4] if item.text]
        if memory.visual_scenes:
            result["visual_scenes"] = [
                cls.trim_dict(scene.to_dict(), item_limit=5) for scene in memory.visual_scenes[:3]
            ]
        if memory.comments:
            result["comments"] = [
                cls.trim_dict(comment.to_dict(), item_limit=6) for comment in memory.comments[:5]
            ]
        for key, values in (
            ("episodes", memory.episodes),
            ("bangumi_links", memory.bangumi_links),
            ("knowledge_links", memory.knowledge_links),
        ):
            if values:
                result[key] = [cls.trim_dict(item.to_dict(), item_limit=6) for item in values[:3]]
        if len(json.dumps(result, ensure_ascii=False)) > max_chars:
            result["summary_text"] = cls.trim_text(memory.summary_text, 220)
            result.pop("comments", None)
            result.pop("visual_scenes", None)
            result.pop("ocr", None)
        return result

    @classmethod
    def build_conversation_context(
        cls,
        raw: dict[str, Any] | None,
        summary_text: str | None,
        *,
        max_chars: int = 650,
    ) -> str:
        profile = cls.compact_profile(raw, max_chars=max_chars)
        if summary_text:
            profile["summary"] = cls.trim_text(summary_text, 220)
        if not profile:
            return ""
        return cls.trim_text(json.dumps(profile, ensure_ascii=False), max_chars)

    @classmethod
    def build_video_context(
        cls,
        memory: VideoMemory | None,
        *,
        max_chars: int = 1500,
    ) -> str:
        value = cls.compact_video_memory(memory, max_chars=max_chars)
        return cls.trim_text(json.dumps(value, ensure_ascii=False), max_chars) if value else ""
