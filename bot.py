"""Бот для канала про нейросети.

3 раза в день: ищет свежую новость -> пишет пост на русском (Gemini) ->
берёт картинку из статьи -> присылает админу на одобрение -> после кнопки
«Опубликовать» выкладывает в канал.
"""
import asyncio
import calendar
import html
import logging
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import aiohttp
import feedparser
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("news_bot")

# ---------- настройки (берутся из переменных окружения) ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
ADMIN_ID = int(os.environ["ADMIN_ID"])          # твой Telegram ID (куда слать на одобрение)
CHANNEL_ID = os.environ["CHANNEL_ID"]           # @username канала или -100...
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TIMEZONE = os.environ.get("TIMEZONE", "Europe/Moscow")
POST_TIMES = [
    tuple(int(x) for x in t.strip().split(":"))
    for t in os.environ.get("POST_TIMES", "10:00,15:00,20:00").split(",")
]
DB_PATH = os.environ.get("DB_PATH", "news_bot.db")
# если основная модель перегружена или недоступна, бот пробует запасные по порядку
MODELS = [GEMINI_MODEL] + [
    m.strip()
    for m in os.environ.get("GEMINI_FALLBACKS", "gemini-3.6-flash,gemini-3.1-flash-lite").split(",")
    if m.strip()
]

FEEDS = [
    "https://habr.com/ru/rss/hubs/artificial_intelligence/all/",
    "https://openai.com/news/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://blog.google/technology/ai/rss/",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
]

MAX_AGE_DAYS = 3
CAPTION_LIMIT = 780      # лимит подписи к фото в Telegram 1024, часть уходит на кнопки/источник
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MIN_IMAGE_BYTES = 5 * 1024
UA = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}

PROMPT = """Ты автор Telegram-канала про нейросети. Напиши пост на русском языке по материалу ниже.

Правила:
- первая строка: короткий цепляющий заголовок (до 80 символов), без точки в конце;
- затем пустая строка и 2-3 коротких абзаца простым языком;
- всего не больше 650 символов;
- можно 1-3 уместных эмодзи;
- без хэштегов, без markdown-разметки, без ссылок;
- используй только факты из материала, ничего не выдумывай;
- если материала недостаточно для поста, ответь одним словом: SKIP.

Название: {title}

Материал:
{text}
"""

# ---------- база ----------
db = sqlite3.connect(DB_PATH)
db.execute("CREATE TABLE IF NOT EXISTS seen (link TEXT PRIMARY KEY, ts TEXT)")
db.execute(
    "CREATE TABLE IF NOT EXISTS posts ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, caption TEXT, source TEXT,"
    "file_id TEXT, status TEXT)"
)
db.commit()


def is_seen(link: str) -> bool:
    return db.execute("SELECT 1 FROM seen WHERE link=?", (link,)).fetchone() is not None


def mark_seen(link: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO seen VALUES (?, ?)", (link, datetime.utcnow().isoformat())
    )
    db.commit()


# ---------- поиск новостей ----------
async def fetch_feed(session: aiohttp.ClientSession, url: str) -> list[dict]:
    try:
        async with session.get(url, headers=UA, timeout=aiohttp.ClientTimeout(total=20)) as r:
            raw = await r.text()
    except Exception as e:
        log.warning("лента %s не загрузилась: %s", url, e)
        return []
    parsed = await asyncio.to_thread(feedparser.parse, raw)
    items = []
    for e in parsed.entries:
        link = getattr(e, "link", None)
        title = getattr(e, "title", None)
        if not link or not title:
            continue
        pub = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        when = datetime.fromtimestamp(calendar.timegm(pub), tz=timezone.utc) if pub else None
        items.append({"link": link, "title": title, "when": when})
    return items


async def get_candidates(session: aiohttp.ClientSession) -> list[dict]:
    results = await asyncio.gather(*(fetch_feed(session, u) for u in FEEDS))
    border = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    items = [
        i
        for lst in results
        for i in lst
        if not is_seen(i["link"]) and (i["when"] is None or i["when"] >= border)
    ]
    items.sort(key=lambda i: i["when"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items


async def read_article(session: aiohttp.ClientSession, url: str):
    """Возвращает (текст статьи, адрес картинки) или (None, None)."""
    try:
        async with session.get(url, headers=UA, timeout=aiohttp.ClientTimeout(total=20)) as r:
            page = await r.text()
    except Exception as e:
        log.warning("статья %s не загрузилась: %s", url, e)
        return None, None
    soup = BeautifulSoup(page, "html.parser")
    image = None
    for attrs in ({"property": "og:image"}, {"name": "twitter:image"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            image = urljoin(url, tag["content"])
            break
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)[:4000]
    return text, image


async def download_image(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, headers=UA, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200 or not r.headers.get("Content-Type", "").startswith("image/"):
                return None
            data = await r.content.read(MAX_IMAGE_BYTES + 1)
    except Exception as e:
        log.warning("картинка %s не скачалась: %s", url, e)
        return None
    if not (MIN_IMAGE_BYTES <= len(data) <= MAX_IMAGE_BYTES):
        return None
    return data


# ---------- написание поста ----------
async def call_gemini(session: aiohttp.ClientSession, model: str, prompt: str, thinking: bool):
    """Один запрос к Gemini. Возвращает (текст, ошибка, http-статус)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen = {"temperature": 0.7}
    if thinking:
        # у новых моделей по умолчанию «думает» долго, для поста хватает low
        gen["thinkingConfig"] = {"thinkingLevel": "low"}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
    data, status = {}, 0
    try:
        async with session.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as r:
            status = r.status
            data = await r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip(), None, status
    except Exception as e:
        err = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        return None, (err or repr(e))[:200], status


async def try_model(session: aiohttp.ClientSession, model: str, prompt: str):
    """Пробует одну модель: при перегрузке ждёт и повторяет один раз. Возвращает (текст, ошибка)."""
    err = None
    for attempt in range(2):
        out, err, status = await call_gemini(session, model, prompt, True)
        if err and status == 400 and "thinking" in err.lower():
            out, err, status = await call_gemini(session, model, prompt, False)
        if not err:
            return out, None
        log.warning("Gemini %s не ответил (попытка %s): %s", model, attempt + 1, err)
        busy = status in (429, 500, 503) or "high demand" in err or "overloaded" in err.lower()
        if busy and attempt == 0:
            await asyncio.sleep(10)
            continue
        break
    return None, err


async def write_post(session: aiohttp.ClientSession, title: str, text: str):
    """Возвращает (текст поста, ошибка). Если основная модель занята, пробует запасные."""
    prompt = PROMPT.format(title=title, text=text)
    last_err = None
    for model in MODELS:
        out, err = await try_model(session, model, prompt)
        if not err:
            out = re.sub(r"[*_`#]+", "", out).strip()
            if not out or out.upper().startswith("SKIP"):
                return None, None
            return out, None
        last_err = f"{model}: {err}"
        if "API key" in err:  # ключ неверный: запасные модели не помогут
            break
    return None, last_err


def build_caption(text: str) -> str:
    """Делает HTML-подпись: заголовок жирным, остальное обычным. Укладывается в лимит."""
    while True:
        head, _, rest = text.partition("\n")
        cap = f"<b>{html.escape(head.strip())}</b>"
        if rest.strip():
            cap += "\n\n" + html.escape(rest.strip())
        if len(cap) <= CAPTION_LIMIT or len(text) < 100:
            return cap
        text = text[: int(len(text) * 0.9)].rsplit(" ", 1)[0]


# ---------- создание поста и отправка на одобрение ----------
def keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Опубликовать", callback_data=f"ok:{post_id}"),
                InlineKeyboardButton(text="❌ Отклонить", callback_data=f"no:{post_id}"),
            ]
        ]
    )


async def create_post(bot: Bot) -> None:
    async with aiohttp.ClientSession() as session:
        candidates = await get_candidates(session)
        pool = candidates[:10]
        random.shuffle(pool)
        stats = {"no_article": 0, "no_image": 0, "bad_image": 0, "skip": 0}
        ai_error = None
        for item in pool[:6]:
            article, image_url = await read_article(session, item["link"])
            if not article or len(article) < 200:
                stats["no_article"] += 1
                mark_seen(item["link"])
                continue
            if not image_url:
                stats["no_image"] += 1
                mark_seen(item["link"])
                continue
            image = await download_image(session, image_url)
            if not image:
                stats["bad_image"] += 1
                mark_seen(item["link"])
                continue
            text, err = await write_post(session, item["title"], article)
            if err:
                # Gemini сломан: новость не «сжигаем», попробуем её в следующий раз
                ai_error = err
                continue
            mark_seen(item["link"])
            if not text:
                stats["skip"] += 1
                continue
            caption = build_caption(text)
            source = item["link"][:150]
            cur = db.execute(
                "INSERT INTO posts (caption, source, status) VALUES (?, ?, 'pending')",
                (caption, source),
            )
            db.commit()
            post_id = cur.lastrowid
            try:
                msg = await bot.send_photo(
                    ADMIN_ID,
                    BufferedInputFile(image, filename="post.jpg"),
                    caption=f"{caption}\n\n🔗 Источник (только для тебя): {html.escape(source)}",
                    reply_markup=keyboard(post_id),
                )
            except Exception as e:
                log.warning("не получилось отправить превью: %s", e)
                db.execute("UPDATE posts SET status='error' WHERE id=?", (post_id,))
                db.commit()
                continue
            db.execute(
                "UPDATE posts SET file_id=? WHERE id=?", (msg.photo[-1].file_id, post_id)
            )
            db.commit()
            return
    if not candidates:
        reason = "Новых новостей в лентах нет или ленты не загрузились."
    else:
        reason = (
            f"Новостей в лентах: {len(candidates)}. Пробовал: {len(pool[:6])}. "
            f"Нет текста статьи: {stats['no_article']}, нет картинки: {stats['no_image']}, "
            f"картинка не скачалась: {stats['bad_image']}, Gemini отказался писать: {stats['skip']}."
        )
        if ai_error:
            reason += f"\nОшибка Gemini: {ai_error}"
    await bot.send_message(ADMIN_ID, "Не получилось создать пост.\n" + html.escape(reason))


# ---------- обработчики ----------
dp = Dispatcher()


@dp.message(Command("start"), F.from_user.id == ADMIN_ID)
async def cmd_start(message: Message):
    await message.answer("Бот работает. Посты буду присылать сюда на одобрение. /now: прислать пост сейчас.")


@dp.message(Command("now"), F.from_user.id == ADMIN_ID)
async def cmd_now(message: Message, bot: Bot):
    await message.answer("Ищу новость, подожди минутку...")
    await create_post(bot)


@dp.callback_query(F.data.regexp(r"^(ok|no):\d+$"))
async def on_decision(call: CallbackQuery, bot: Bot):
    if call.from_user.id != ADMIN_ID:
        await call.answer()
        return
    action, pid = call.data.split(":")
    row = db.execute(
        "SELECT caption, file_id, status FROM posts WHERE id=?", (int(pid),)
    ).fetchone()
    if not row or row[2] != "pending":
        await call.answer("Этот пост уже обработан", show_alert=True)
        return
    caption, file_id, _ = row
    if action == "no":
        db.execute("UPDATE posts SET status='rejected' WHERE id=?", (pid,))
        db.commit()
        await call.message.edit_reply_markup(reply_markup=None)
        await call.message.reply("❌ Отклонено")
        await call.answer()
        return
    try:
        await bot.send_photo(CHANNEL_ID, file_id, caption=caption)
    except Exception as e:
        log.warning("публикация не удалась: %s", e)
        await call.answer(f"Не получилось опубликовать: {e}", show_alert=True)
        return
    db.execute("UPDATE posts SET status='published' WHERE id=?", (pid,))
    db.commit()
    await call.message.edit_reply_markup(reply_markup=None)
    await call.message.reply("✅ Опубликовано в канал")
    await call.answer()


# ---------- расписание ----------
async def scheduler(bot: Bot) -> None:
    tz = ZoneInfo(TIMEZONE)
    while True:
        now = datetime.now(tz)
        upcoming = []
        for d in (0, 1):
            for h, m in POST_TIMES:
                dt = datetime.combine(now.date() + timedelta(days=d), datetime.min.time(), tz).replace(
                    hour=h, minute=m
                )
                if dt > now:
                    upcoming.append(dt)
        nxt = min(upcoming)
        log.info("следующий пост в %s", nxt)
        await asyncio.sleep((nxt - now).total_seconds())
        try:
            await create_post(bot)
        except Exception:
            log.exception("ошибка при создании поста")
        await asyncio.sleep(2)


async def start_web() -> None:
    """Маленькая страница «ok», чтобы пингер (UptimeRobot) не давал сервису уснуть."""

    async def health(request):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", "10000"))).start()


async def main() -> None:
    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await start_web()
    asyncio.create_task(scheduler(bot))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
