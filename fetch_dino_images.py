#!/usr/bin/env python3
"""
Usage examples:
    python fetch_dino_images.py \
      --species-csv dinosaurs_species_cards.csv \
      --out-dir public/assets \
      --mapping-csv dinosaur_image_mapping.csv

    python fetch_dino_images.py \
      --species-csv dinosaurs_species_cards.csv \
      --names-csv candidate_dinosaurs_top35.csv \
      --out-dir . \
      --download-mode thumb \
      --thumb-size 800
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Built-in target set: good exhibit candidates with high odds of recognizable images.
DEFAULT_TARGETS = [
    "Tyrannosaurus", "Triceratops", "Velociraptor", "Stegosaurus", "Brachiosaurus",
    "Diplodocus", "Allosaurus", "Ankylosaurus", "Spinosaurus", "Iguanodon",
    "Parasaurolophus", "Apatosaurus", "Carnotaurus", "Ceratosaurus", "Compsognathus",
    "Deinonychus", "Edmontosaurus", "Gallimimus", "Giganotosaurus", "Kentrosaurus",
    "Maiasaura", "Mamenchisaurus", "Microraptor", "Oviraptor", "Pachycephalosaurus",
    "Plateosaurus", "Protoceratops", "Psittacosaurus", "Sauropelta", "Styracosaurus",
    "Therizinosaurus", "Troodon", "Utahraptor", "Corythosaurus", "Dilophosaurus",
]


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def clean_htmlish(value: Optional[str]) -> str:
    if not value:
        return ""
    # Commons extmetadata can contain HTML fragments.
    value = html.unescape(str(value))
    value = re.sub(r"<br\s*/?>", " | ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_targets(species_df: pd.DataFrame, names_csv: Optional[str]) -> pd.DataFrame:
    if names_csv:
        df = pd.read_csv(names_csv)
        if "display_name" in df.columns:
            names = df["display_name"].dropna().astype(str).tolist()
        elif "name" in df.columns:
            names = df["name"].dropna().astype(str).tolist()
        else:
            raise ValueError("--names-csv must contain a 'display_name' or 'name' column")
    else:
        names = DEFAULT_TARGETS

    names = list(dict.fromkeys(names))  # preserve order, dedupe
    subset = species_df[species_df["display_name"].isin(names)].copy()
    missing = [n for n in names if n not in set(subset["display_name"])]
    if missing:
        print(f"[warn] {len(missing)} targets were not found in species CSV: {missing}", file=sys.stderr)

    # restore requested order
    subset["_order"] = subset["display_name"].map({n: i for i, n in enumerate(names)})
    subset = subset.sort_values("_order").drop(columns="_order")
    return subset


def query_wikipedia_pageimage(session: requests.Session, title: str, thumb_size: int) -> Dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": 1,
        "prop": "pageimages|info",
        "piprop": "name|thumbnail|original",
        "pithumbsize": thumb_size,
        "titles": title,
        "inprop": "url",
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = safe_get(data, "query", "pages", default=[]) or []
    if not pages:
        return {}
    return pages[0]


def query_commons_imageinfo(session: requests.Session, file_title: str) -> Dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "imageinfo",
        "titles": file_title,
        "iiprop": "url|extmetadata",
    }
    r = session.get(COMMONS_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = safe_get(data, "query", "pages", default=[]) or []
    if not pages:
        return {}
    return pages[0]


def choose_download_url(pageimage_info: Dict, commons_info: Dict, mode: str) -> Optional[str]:
    if mode == "original":
        return safe_get(commons_info, "imageinfo", 0, "url")
    # Prefer PageImages thumb, fallback to original.
    return pageimage_info.get("thumbnail", {}).get("source") or safe_get(commons_info, "imageinfo", 0, "url")


def ext_from_url(url: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]{2,5})(?:\?|$)", url)
    return m.group(1).lower() if m else "jpg"


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


def build_mapping_row(
    dino_row: pd.Series,
    image_mode: str,
    rel_image_path: str = "",
    article_title: str = "",
    article_url: str = "",
    commons_file_title: str = "",
    artist: str = "",
    license_short: str = "",
    credit: str = "",
    source_page: str = "",
    notes: str = "",
) -> Dict[str, str]:
    return {
        "slug": dino_row["slug"],
        "display_name": dino_row["display_name"],
        "image_mode": image_mode,
        "image_url": rel_image_path,
        "image_credit": artist or credit,
        "image_license": license_short,
        "image_source_page": source_page or article_url,
        "notes": notes,
        "article_title": article_title,
        "article_url": article_url,
        "commons_file_title": commons_file_title,
        "diet": dino_row.get("diet", ""),
        "type": dino_row.get("type", ""),
        "occurrence_count": dino_row.get("occurrence_count", ""),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-csv", required=True, help="Path to dinosaurs_species_cards.csv")
    ap.add_argument("--names-csv", default="", help="Optional CSV of dinosaur names to fetch (display_name or name col)")
    ap.add_argument("--out-dir", default=".", help="Root output dir. Images go under <out-dir>/images/dinos/")
    ap.add_argument("--mapping-csv", default="", help="Optional explicit output path for image mapping CSV")
    ap.add_argument("--download-mode", choices=["thumb", "original"], default="thumb")
    ap.add_argument("--thumb-size", type=int, default=640)
    ap.add_argument("--sleep-seconds", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    species_df = pd.read_csv(args.species_csv)
    targets_df = load_targets(species_df, args.names_csv or None)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images" / "dinos"
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else out_dir / "dinosaur_image_mapping.csv"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "dino-museum-homework-image-fetcher/1.0 (educational project)",
        "Accept": "application/json",
    })

    rows: List[Dict[str, str]] = []
    failures: List[Tuple[str, str]] = []

    print(f"[info] targets: {len(targets_df)}")
    print(f"[info] writing images to: {images_dir}")
    print(f"[info] writing mapping csv to: {mapping_csv}")

    for _, dino in targets_df.iterrows():
        name = str(dino["display_name"])
        slug = str(dino["slug"])
        print(f"[info] fetching image for {name}...")

        try:
            page = query_wikipedia_pageimage(session, name, args.thumb_size)
            article_title = page.get("title", name)
            article_url = page.get("fullurl", "")
            pageimage_name = page.get("pageimage", "")

            if not pageimage_name:
                rows.append(build_mapping_row(
                    dino, image_mode="silhouette", article_title=article_title,
                    article_url=article_url, notes="No representative Wikipedia page image found."
                ))
                failures.append((name, "no_pageimage"))
                time.sleep(args.sleep_seconds)
                continue

            commons_file_title = pageimage_name if pageimage_name.startswith("File:") else f"File:{pageimage_name}"
            commons_page = query_commons_imageinfo(session, commons_file_title)
            imageinfo = safe_get(commons_page, "imageinfo", 0, default={}) or {}
            extmeta = imageinfo.get("extmetadata", {}) or {}

            download_url = choose_download_url(page, commons_page, args.download_mode)
            if not download_url:
                rows.append(build_mapping_row(
                    dino, image_mode="silhouette", article_title=article_title,
                    article_url=article_url, commons_file_title=commons_file_title,
                    notes="Commons file found but no downloadable URL was returned."
                ))
                failures.append((name, "no_download_url"))
                time.sleep(args.sleep_seconds)
                continue

            extension = ext_from_url(download_url)
            dest = images_dir / f"{slug}.{extension}"
            rel_image_path = f"images/dinos/{slug}.{extension}"

            if args.overwrite or not dest.exists():
                ensure_parent(dest)
                download_file(session, download_url, dest)

            artist = clean_htmlish(safe_get(extmeta, "Artist", "value", default=""))
            credit = clean_htmlish(safe_get(extmeta, "Credit", "value", default=""))
            license_short = clean_htmlish(safe_get(extmeta, "LicenseShortName", "value", default=""))
            license_url = clean_htmlish(safe_get(extmeta, "LicenseUrl", "value", default=""))
            source_page = imageinfo.get("descriptionurl", "") or article_url

            notes = f"download_mode={args.download_mode}; license_url={license_url}" if license_url else f"download_mode={args.download_mode}"

            rows.append(build_mapping_row(
                dino,
                image_mode="image",
                rel_image_path=rel_image_path,
                article_title=article_title,
                article_url=article_url,
                commons_file_title=commons_file_title,
                artist=artist,
                license_short=license_short,
                credit=credit,
                source_page=source_page,
                notes=notes,
            ))
            time.sleep(args.sleep_seconds)

        except requests.HTTPError as e:
            rows.append(build_mapping_row(dino, image_mode="silhouette", notes=f"HTTPError: {e}"))
            failures.append((name, f"http_error:{e}"))
        except Exception as e:
            rows.append(build_mapping_row(dino, image_mode="silhouette", notes=f"Exception: {e}"))
            failures.append((name, f"exception:{e}"))

    fieldnames = [
        "slug", "display_name", "image_mode", "image_url", "image_credit", "image_license",
        "image_source_page", "notes", "article_title", "article_url", "commons_file_title",
        "diet", "type", "occurrence_count",
    ]
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote mapping CSV with {len(rows)} rows")
    if failures:
        print(f"[warn] {len(failures)} rows fell back to silhouette:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
