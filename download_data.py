import sys
import pandas as pd
import zipfile
import requests
from pathlib import Path

zenodo_api = "https://zenodo.org/api/records/4010759"
download_dir = Path("zips")
videos_dir = Path("videos")
metadata_csv = "include50_metadata.csv"

zip_files = [
    ("Adjectives_1of8.zip",              1303983457),
    ("Adjectives_2of8.zip",              1407281940),
    ("Adjectives_3of8.zip",              1425808613),
    ("Adjectives_4of8.zip",              1214968173),
    ("Adjectives_5of8.zip",              1310663669),
    ("Adjectives_6of8.zip",              1247813680),
    ("Adjectives_7of8.zip",              1253415149),
    ("Adjectives_8of8.zip",               834989744),
    ("Animals_1of2.zip",                 1819837734),
    ("Animals_2of2.zip",                 1069027692),
    ("Clothes_1of2.zip",                 1387609869),
    ("Clothes_2of2.zip",                 1396763182),
    ("Colours_1of2.zip",                 1268663334),
    ("Colours_2of2.zip",                 1452446819),
    ("Days_and_Time_1of3.zip",           1244115203),
    ("Days_and_Time_2of3.zip",           1016797379),
    ("Days_and_Time_3of3.zip",            852955661),
    ("Electronics_1of2.zip",              926290038),
    ("Electronics_2of2.zip",              824110880),
    ("Greetings_1of2.zip",              1573733886),
    ("Greetings_2of2.zip",              1208700458),
    ("Home_1of4.zip",                   1248459133),
    ("Home_2of4.zip",                   1338509691),
    ("Home_3of4.zip",                   1080038303),
    ("Home_4of4.zip",                    873459938),
    ("Jobs_1of2.zip",                   1503117419),
    ("Jobs_2of2.zip",                   1650647791),
    ("Means_of_Transportation_1of2.zip", 1846003966),
    ("Means_of_Transportation_2of2.zip", 1606937936),
    ("People_1of5.zip",                 1327653833),
    ("People_2of5.zip",                 1251273463),
    ("People_3of5.zip",                 1586966004),
    ("People_4of5.zip",                 1303778856),
    ("People_5of5.zip",                 1301122063),
    ("Places_1of4.zip",                 1394196824),
    ("Places_2of4.zip",                 1355857452),
    ("Places_3of4.zip",                 1466179266),
    ("Places_4of4.zip",                 1058186818),
    ("Pronouns_1of2.zip",               1419853451),
    ("Pronouns_2of2.zip",                946837765),
    ("Seasons_1of1.zip",                1251955665),
    ("Society_1of3.zip",                1436640862),
    ("Society_2of3.zip",                1358587307),
    ("Society_3of3.zip",                1105579609),
]

def download_file(name, expected_size):
    dest = download_dir/name
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"Already downloaded: {name} -- skipping")
        return
    
    url = f"{zenodo_api}/files/{name}/content"
    print(f"  Downloading {name} ({expected_size / 1e9:.1f} GB)...")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8*1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / expected_size * 100
                print(f"\r    {pct:.1f}%  ({downloaded / 1e9:.2f} / {expected_size / 1e9:.2f} GB)",end="", flush=True)
                print()

def extract_include50(zip_path, needed_paths):
    extracted = 0
        