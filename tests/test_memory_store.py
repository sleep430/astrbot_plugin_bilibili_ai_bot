import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bilibili_bot.memory_models import ConversationMemory, TodayActivity
from bilibili_bot.memory_store import MemoryStore


def test_sqlite_store_round_trip(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    store.upsert("u1", {"display_name": "林语眠", "likes": {"画画": 0.9}})
    store.set_summary_text("u1", "喜欢安静的夜晚和画画。")
    store.save_conversation_memory(ConversationMemory(uid="u1", summary="一次对话", key_facts=["喜欢画画"]))
    store.set_active_conversation("u1", {"messages": [{"role": "user", "text": "你好"}], "last_active_at": datetime.now().isoformat()})
    store.save_dynamic("d1", {"dynamic_id": "d1", "text": "动态"})

    assert store.get("u1")["display_name"] == "林语眠"
    assert store.get_summary_text("u1") == "喜欢安静的夜晚和画画。"
    assert store.get_conversation_memory("u1").key_facts == ["喜欢画画"]
    assert store.get_active_conversation("u1")["messages"][0]["text"] == "你好"
    assert store.get_dynamic("d1")["text"] == "动态"
    assert store.stats().db_size_bytes > 0

    # 新实例从 SQLite 读取，不依赖 JSON 持续更新。
    reopened = MemoryStore(tmp_path / "memory.json")
    assert reopened.get("u1")["display_name"] == "林语眠"
    assert reopened.get_summary_text("u1") == "喜欢安静的夜晚和画画。"
    assert reopened.get_dynamic("d1")["text"] == "动态"


def test_legacy_json_import_once(tmp_path: Path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "version": 5,
        "summary_texts": {"u1": "摘要"},
        "users": {"u1": {"display_name": "A"}},
        "conversation_memories": {"u1": {"uid": "u1", "summary": "old"}},
        "video_memories": {"BV1": {"bvid": "BV1", "video_profile": {"title": "视频"}}},
    }, ensure_ascii=False), encoding="utf-8")
    store = MemoryStore(path)
    assert store.get("u1")["display_name"] == "A"
    assert store.get_video_memory("BV1").video_profile.title == "视频"
    store.upsert("u2", {"display_name": "B"})
    path.write_text("{}", encoding="utf-8")
    reopened = MemoryStore(path)
    assert reopened.get("u2")["display_name"] == "B"


def test_bangumi_only_sqlite_does_not_skip_json_import(tmp_path: Path):
    """只用过番剧功能的旧实例，users/conversations 为空也必须迁 JSON。"""
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.save_bangumi_history(114514, {"uid": 114514, "titles": ["无职转生"]})
    path.write_text(json.dumps({
        "version": 5,
        "summary_texts": {"u1": "摘要"},
        "users": {"u1": {"display_name": "A"}},
        "conversation_memories": {"u1": {"uid": "u1", "summary": "old"}},
    }, ensure_ascii=False), encoding="utf-8")
    reopened = MemoryStore(path)
    assert reopened.get("u1")["display_name"] == "A"
    assert reopened.get_conversation_memory("u1").summary == "old"


def test_empty_sqlite_does_not_reimport_cleared_json(tmp_path: Path):
    """有业务数据后清掉某个用户，重载不能把旧 JSON 再灌回来。"""
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.upsert("u1", {"display_name": "A"})
    path.write_text(json.dumps({
        "version": 5,
        "users": {"u1": {"display_name": "STALE"}},
    }, ensure_ascii=False), encoding="utf-8")
    reopened = MemoryStore(path)
    assert reopened.get("u1")["display_name"] == "A"


def test_prune_expires_active_conversation_cursor_and_chat(tmp_path: Path):
    """过期游标/会话/聊天状态不能无限留下。"""
    store = MemoryStore(tmp_path / "memory.json")
    old = datetime.fromtimestamp(time.time() - 8 * 86400).isoformat()
    store.set_active_conversation("u1", {"messages": [], "last_active_at": old})
    store.set_conversation_cursor("u1", {"updated_at": old})
    from bilibili_bot.memory_models import ChatState
    store.save_chat_state(ChatState(chat_id="g1", last_active_at=old))
    store.prune()
    assert store.get_active_conversation("u1") == {}
    assert store.get_conversation_cursor("u1") == {}
    assert store.get_chat_state("g1") is None


def test_get_dynamic_and_remove_dynamic_round_trip(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    assert store.get_dynamic("d1") is None
    assert store.remove_dynamic("d1") is False
    store.save_dynamic("d1", {"text": "A", "created_at": "2026-01-01T00:00:00"})
    store.save_dynamic("d1", {"text": "B"})
    assert store.get_dynamic("d1")["text"] == "B"
    assert store.remove_dynamic("d1") is True
    assert store.get_dynamic("d1") is None


def test_compact_video_memory_handles_oversized_summary(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    from bilibili_bot.memory_models import VideoMemory
    memory = VideoMemory(bvid="BV1", summary_text="x" * 2000)
    store.save_video_memory(memory)
    compact = store.compact_video_memory(memory)
    assert len(compact["summary_text"]) <= 360
    assert len(store.build_video_context(memory)) <= 1500
