import asyncio
import os
import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument

import face_recognition
 
API_ID   = int(os.getenv("TG_API_ID",   "0"))      
API_HASH =     os.getenv("TG_API_HASH", "your_hash") 
CHANNEL  = "kaliningradru"                             
MONTHS   = 4                                          
TOLERANCE = 0.55   
SAVE_DIR = Path("downloaded_images")                  
SAVE_IMAGES = True                                     

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """Декодирует bytes → numpy-массив RGB для face_recognition."""
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(pil_img)
    except Exception as exc:
        log.warning("Не удалось декодировать изображение: %s", exc)
        return None


def get_face_encodings(image_np: np.ndarray) -> list:
    """Возвращает список 128-мерных векторов лиц на изображении."""
    locations = face_recognition.face_locations(image_np, model="hog")
    if not locations:
        return []
    encodings = face_recognition.face_encodings(image_np, locations)
    return encodings


def find_duplicate_faces(encodings_per_image: list[list], tolerance: float) -> list[dict]:
    """
    Ищет одинаковые лица среди нескольких изображений одного сообщения.

    encodings_per_image: [ [enc1, enc2, ...], [enc3, ...], ... ]
    Возвращает список совпадений: {face_id, appearances: [(img_idx, face_idx), ...]}
    """
    # Собираем все (img_idx, face_idx, encoding) в плоский список
    all_faces: list[tuple[int, int, np.ndarray]] = []
    for img_idx, encs in enumerate(encodings_per_image):
        for face_idx, enc in enumerate(encs):
            all_faces.append((img_idx, face_idx, enc))

    if len(all_faces) < 2:
        return []

    visited = [False] * len(all_faces)
    groups: list[dict] = []

    for i in range(len(all_faces)):
        if visited[i]:
            continue
        group_members = [(all_faces[i][0], all_faces[i][1])]  # (img_idx, face_idx)
        visited[i] = True
        for j in range(i + 1, len(all_faces)):
            if visited[j]:
                continue
            dist = face_recognition.face_distance(
                [all_faces[i][2]], all_faces[j][2]
            )[0]
            if dist <= tolerance:
                group_members.append((all_faces[j][0], all_faces[j][1]))
                visited[j] = True

        # Интересует только совпадение на РАЗНЫХ изображениях
        unique_images = {m[0] for m in group_members}
        if len(unique_images) >= 2:
            groups.append({
                "appearances": group_members,
                "unique_images": sorted(unique_images),
            })

    return groups


async def download_photos_for_message(client: TelegramClient, message) -> list[bytes]:
    """Скачивает все фото/изображения, прикреплённые к сообщению."""
    photos: list[bytes] = []

    # Grouped album — несколько медиа в одном сообщении
    # Telethon возвращает альбомы как отдельные сообщения с одинаковым grouped_id
    # Поэтому мы группируем снаружи; здесь обрабатываем одно медиа-вложение

    media = message.media
    if media is None:
        return photos

    if isinstance(media, MessageMediaPhoto):
        data = await client.download_media(message, file=bytes)
        if data:
            photos.append(data)

    elif isinstance(media, MessageMediaDocument):
        doc = media.document
        # Берём только изображения
        mime = getattr(doc, "mime_type", "") or ""
        if mime.startswith("image/"):
            data = await client.download_media(message, file=bytes)
            if data:
                photos.append(data)

    return photos

async def main():
    if API_ID == 0 or API_HASH == "your_hash":
        log.error(
            "Задайте API_ID и API_HASH!\n"
            "  Через переменные окружения: TG_API_ID=... TG_API_HASH=...\n"
            "  Или напрямую в коде скрипта."
        )
        return

    if SAVE_IMAGES:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=MONTHS * 30)
    log.info("Парсинг канала @%s с %s", CHANNEL, cutoff.strftime("%Y-%m-%d"))

    async with TelegramClient("tg_session", API_ID, API_HASH) as client:
        # Сначала собираем сообщения, группируя альбомы по grouped_id
        albums: dict[int, list] = {}   # grouped_id → [messages]
        singles: list = []             # сообщения без grouped_id

        log.info("Загрузка истории сообщений...")
        async for msg in client.iter_messages(CHANNEL, offset_date=None, reverse=False):
            if msg.date < cutoff:
                break
            if msg.media is None:
                continue
            if not isinstance(msg.media, (MessageMediaPhoto, MessageMediaDocument)):
                continue

            if msg.grouped_id:
                albums.setdefault(msg.grouped_id, []).append(msg)
            else:
                singles.append(msg)

        log.info(
            "Найдено %d альбомов и %d одиночных медиа-сообщений.",
            len(albums), len(singles),
        )

        # Объединяем: каждый альбом — группа, одиночные — тоже группы по 1 сообщению
        # (одиночное сообщение с одним фото даст encodings_per_image из 1 элемента →
        #  совпадений не будет, что корректно)
        groups_to_check: list[tuple[str, list]] = []
        for gid, msgs in albums.items():
            label = f"Альбом grouped_id={gid} (сообщение #{msgs[0].id}, {msgs[0].date.strftime('%Y-%m-%d')})"
            groups_to_check.append((label, msgs))
        for msg in singles:
            label = f"Сообщение #{msg.id} ({msg.date.strftime('%Y-%m-%d')})"
            groups_to_check.append((label, [msg]))

        total_matches = 0

        for label, msgs in groups_to_check:
            log.info("Обработка: %s", label)

            # Скачиваем фото каждого сообщения в группе
            all_photo_bytes: list[bytes] = []
            for msg in msgs:
                photos = await download_photos_for_message(client, msg)
                all_photo_bytes.extend(photos)

            if len(all_photo_bytes) < 2:
                # Меньше двух изображений — сравнивать не с чем
                continue

            # Сохраняем на диск (опционально)
            if SAVE_IMAGES:
                msg_dir = SAVE_DIR / str(msgs[0].id)
                msg_dir.mkdir(parents=True, exist_ok=True)
                for idx, raw in enumerate(all_photo_bytes):
                    (msg_dir / f"photo_{idx}.jpg").write_bytes(raw)

            # Получаем эмбеддинги лиц для каждого фото
            encodings_per_image: list[list] = []
            for idx, raw in enumerate(all_photo_bytes):
                np_img = load_image_from_bytes(raw)
                if np_img is None:
                    encodings_per_image.append([])
                    continue
                encs = get_face_encodings(np_img)
                log.debug("  Фото %d: найдено %d лиц", idx, len(encs))
                encodings_per_image.append(encs)

            # Ищем совпадения
            matches = find_duplicate_faces(encodings_per_image, TOLERANCE)

            if not matches:
                continue

            total_matches += len(matches)
            print("\n" + "=" * 70)
            print(f"  {label}")
            print(f"  Изображений в группе: {len(all_photo_bytes)}")
            print(f"  Найдено повторяющихся лиц: {len(matches)}")
            for match_no, match in enumerate(matches, 1):
                img_list = ", ".join(
                    f"фото #{i}" for i in match["unique_images"]
                )
                print(f"    Лицо {match_no}: встречается на {img_list}")
            print("=" * 70)

        print(f"\nИтого групп с повторяющимися лицами: {total_matches}")


if __name__ == "__main__":
    asyncio.run(main())
