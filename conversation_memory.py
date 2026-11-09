"""对话记忆生命周期：活跃会话滚动总结 + 后台低功耗维护。"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable

from astrbot.api import logger

from .constants import (
    DEFAULT_ACTIVE_CONVERSATION_MEMORY_LIMIT,
    DEFAULT_ACTIVE_CONVERSATION_TTL_SECONDS,
    DEFAULT_MEMORY_MODEL_INTERVAL_SECONDS,
    DEFAULT_RECENT_SUMMARY_INTERVAL_SECONDS,
    DEFAULT_USER_SUMMARY_LIMIT,
)
from .memory_models import ConversationMemory, parse_datetime
from .memory_store import MemoryStore
from .utils import extract_json_object, now_iso


class ConversationMemoryManager:
    """负责近期会话分段、重要旧会话压缩和每日后台维护。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        plugin_config: Any = None,
        owner_id_getter: Callable[[], str] | None = None,
    ):
        self.store = store
        self.plugin_config = plugin_config
        self.owner_id_getter = owner_id_getter
        self._tasks: dict[str, asyncio.Task] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._maintenance_task: asyncio.Task | None = None
        self._maintenance_running = False
        self._lock = asyncio.Lock()

    def _cfg(self, name: str, default: Any) -> Any:
        getter = getattr(self.plugin_config, "get", None)
        if callable(getter):
            try:
                return getter(name, default)
            except Exception:
                pass
        if isinstance(self.plugin_config, dict):
            return self.plugin_config.get(name, default)
        return getattr(self.plugin_config, name, default)

    def _owner_id(self) -> str:
        return str(self.owner_id_getter() or "") if self.owner_id_getter else ""

    def _provider(self):
        getter = getattr(self.plugin_config, "get", None)
        if callable(getter):
            for key in (
                "memory_model_provider_id",
                "scheduled_model_provider_id",
                "review_model_provider_id",
            ):
                value = getter(key, "")
                if value:
                    return value
        return ""

    async def _call_model(self, prompt: str, *, timeout: int = 90) -> str:
        context = getattr(self.plugin_config, "context", None)
        if context is None:
            return ""
        provider_id = self._provider()
        provider = None
        if provider_id:
            try:
                provider = await context.get_provider_by_id(provider_id)
            except Exception:
                provider = None
        if provider is None:
            try:
                provider = await context.get_current_provider()
            except Exception:
                provider = None
        if provider is None:
            return ""
        try:
            response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt, contexts=[]), timeout=timeout
            )
            return str(getattr(response, "completion_text", "") or "").strip()
        except Exception as exc:
            logger.debug(f"对话记忆模型调用失败: {exc}")
            return ""

    @staticmethod
    def _message_text(message: Any) -> str:
        if isinstance(message, dict):
            return str(message.get("text") or message.get("content") or "")
        return str(message or "")

    @staticmethod
    def _message_role(message: Any) -> str:
        if isinstance(message, dict):
            return str(message.get("role") or "user")
        return "user"

    def schedule_conversation(
        self,
        uid: str,
        user_name: str,
        messages: list[dict[str, Any]],
        *,
        chat_id: str = "",
        event: Any = None,
    ) -> None:
        if not uid or not messages:
            return
        now = time.time()
        with_lock = self._pending.get(uid)
        if with_lock is None:
            self._pending[uid] = {
                "uid": uid,
                "user_name": user_name,
                "chat_id": chat_id,
                "messages": deque(maxlen=80),
                "last_active": now,
                "event": event,
            }
            with_lock = self._pending[uid]
        with_lock["last_active"] = now
        with_lock["user_name"] = user_name or with_lock.get("user_name", "")
        with_lock["chat_id"] = chat_id or with_lock.get("chat_id", "")
        with_lock["event"] = event or with_lock.get("event")
        with_lock["messages"].extend(messages[-20:])
        old = self._tasks.get(uid)
        if old and not old.done():
            old.cancel()
        delay = max(15, int(self._cfg("recent_summary_interval_seconds", DEFAULT_RECENT_SUMMARY_INTERVAL_SECONDS)))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._tasks[uid] = loop.create_task(self._delayed_flush(uid, delay))

    async def _delayed_flush(self, uid: str, delay: int) -> None:
        try:
            await asyncio.sleep(delay)
            await self.flush(uid)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug(f"延迟总结失败 uid={uid}: {exc}")

    async def flush(self, uid: str) -> bool:
        pending = self._pending.get(uid)
        if not pending:
            return False
        messages = list(pending.get("messages") or [])
        if not messages:
            self._pending.pop(uid, None)
            return False
        active = self.store.get_active_conversation(uid)
        old_messages = list(active.get("messages") or [])
        merged = old_messages + messages
        if len(merged) > 36:
            overflow = merged[:-24]
            merged = merged[-24:]
            await self._summarize_overflow(uid, pending, overflow, active)
            active = self.store.get_active_conversation(uid)
        active.update(
            {
                "uid": uid,
                "user_name": pending.get("user_name", ""),
                "chat_id": pending.get("chat_id", ""),
                "messages": merged,
                "last_active_at": now_iso(),
                "started_at": active.get("started_at") or now_iso(),
            }
        )
        self.store.set_active_conversation(uid, active)
        self._pending.pop(uid, None)
        return True

    async def _summarize_overflow(
        self,
        uid: str,
        pending: dict[str, Any],
        messages: list[dict[str, Any]],
        active: dict[str, Any],
    ) -> None:
        text = "\n".join(
            f"{self._message_role(item)}: {self._message_text(item)}"
            for item in messages
            if self._message_text(item)
        )
        if not text:
            return
        prompt = (
            "请把以下对话压缩成简短记忆，保留用户透露的稳定事实、喜好、重要事件、"
            "未完成的约定和关系变化。不要编造。用 JSON 返回："
            '{"summary":"...","facts":["..."],"topics":["..."],"importance":0.0}\n'
            + text
        )
        raw = await self._call_model(prompt, timeout=75)
        parsed = extract_json_object(raw) if raw else {}
        summary = str(parsed.get("summary") or raw or "").strip()[:1200]
        facts = parsed.get("facts") if isinstance(parsed.get("facts"), list) else []
        topics = parsed.get("topics") if isinstance(parsed.get("topics"), list) else []
        importance = parsed.get("importance", 0.45)
        try:
            importance = max(0.0, min(1.0, float(importance)))
        except (TypeError, ValueError):
            importance = 0.45
        memory = self.store.get_conversation_memory(uid) or ConversationMemory(uid=uid)
        memory.user_name = pending.get("user_name", memory.user_name)
        memory.chat_id = pending.get("chat_id", memory.chat_id)
        memory.summary = "\n".join(filter(None, [memory.summary, summary]))[-6000:]
        memory.key_facts = list(dict.fromkeys(memory.key_facts + [str(x)[:300] for x in facts]))[-40:]
        memory.topics = list(dict.fromkeys(memory.topics + [str(x)[:80] for x in topics]))[-20:]
        memory.message_count += len(messages)
        memory.importance = max(memory.importance, importance)
        memory.started_at = memory.started_at or active.get("started_at") or now_iso()
        memory.updated_at = now_iso()
        self.store.save_conversation_memory(memory)

    async def flush_all(self) -> int:
        count = 0
        for uid in list(self._pending):
            if await self.flush(uid):
                count += 1
        return count

    async def archive_inactive(self) -> int:
        ttl = int(self._cfg("active_conversation_ttl_seconds", DEFAULT_ACTIVE_CONVERSATION_TTL_SECONDS))
        now = time.time()
        count = 0
        for uid in list(self._pending):
            pending = self._pending.get(uid) or {}
            if now - float(pending.get("last_active") or now) >= ttl:
                await self.flush(uid)
        for uid, active in list(self.store._conversation_active.items()):
            last = parse_datetime(active.get("last_active_at")) or 0
            if now - last < ttl:
                continue
            messages = list(active.get("messages") or [])
            if messages:
                await self._summarize_overflow(uid, active, messages, active)
            self.store.clear_active_conversation(uid)
            count += 1
        return count

    async def compress_old_memories(self) -> int:
        limit = int(self._cfg("active_conversation_memory_limit", DEFAULT_ACTIVE_CONVERSATION_MEMORY_LIMIT))
        memories = self.store.list_conversation_memories(limit=max(limit * 3, 300))
        if len(memories) <= limit:
            return 0
        count = 0
        for memory in memories[limit:]:
            if len(memory.summary) <= 1600:
                continue
            prompt = (
                "将旧对话记忆进一步压缩为不超过 700 字的摘要，保留关键事实和关系变化，"
                "删掉重复细节。只输出摘要正文。\n" + memory.summary
            )
            summary = await self._call_model(prompt, timeout=90)
            if summary:
                memory.summary = summary[:1600]
                memory.updated_at = now_iso()
                self.store.save_conversation_memory(memory)
                count += 1
        return count

    async def nightly_user_summaries(self) -> int:
        users = self.store.all_users()
        count = 0
        for uid, profile in users.items():
            active = self.store.get_active_conversation(uid)
            memory = self.store.get_conversation_memory(uid)
            text = "\n".join(
                filter(
                    None,
                    [
                        str(profile.get("long_term_summary") or ""),
                        str((memory.summary if memory else "") or ""),
                        "\n".join(self._message_text(x) for x in (active.get("messages") or [])[-12:]),
                    ],
                )
            )
            if not text.strip():
                continue
            prompt = (
                "请根据以下用户画像与近期对话，生成一段不超过 500 字的长期用户摘要。"
                "只保留稳定、对以后交流有用的信息，不要编造。\n" + text
            )
            summary = await self._call_model(prompt, timeout=90)
            if summary:
                self.store.set_summary_text(uid, summary[:1200])
                profile["long_term_summary"] = summary[:1200]
                self.store.upsert(uid, profile)
                count += 1
            await asyncio.sleep(1)
        return count

    async def run_daily_maintenance(self) -> dict[str, int]:
        if self._maintenance_running:
            return {"archived": 0, "compressed": 0, "summaries": 0}
        self._maintenance_running = True
        try:
            archived = await self.archive_inactive()
            compressed = await self.compress_old_memories()
            summaries = await self.nightly_user_summaries()
            self.store.prune()
            return {"archived": archived, "compressed": compressed, "summaries": summaries}
        except Exception as exc:
            logger.warning(f"对话记忆后台维护失败: {exc}")
            return {"archived": 0, "compressed": 0, "summaries": 0}
        finally:
            self._maintenance_running = False

    async def start(self) -> None:
        if self._maintenance_task and not self._maintenance_task.done():
            return
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                if datetime.now().hour != 4:
                    continue
                # 每天四点只跑一次；避免事件循环/时区边界重复触发。
                marker = datetime.now().strftime("%Y-%m-%d")
                if getattr(self, "_last_maintenance_day", "") == marker:
                    continue
                self._last_maintenance_day = marker
                await self.run_daily_maintenance()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(f"对话记忆维护循环异常: {exc}")

    async def close(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._maintenance_task:
            self._maintenance_task.cancel()
        await self.flush_all()
