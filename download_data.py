import sys
import requests
import zipfile
import pandas as pd
import time
from pathlib import Path


ZENODO_API     = "https://zenodo.org/api/records/4010759"
DOWNLOAD_DIR   = Path("zips")
VIDEOS_DIR     = Path("videos")
METADATA_CSV   = "include50_metadata.csv"
MAX_RETRIES    = 3
RETRY_DELAY    = 5  # seconds between retries

# All 44 zip files and their exact sizes (bytes)
ZIP_FILES = [

    
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
    dest = DOWNLOAD_DIR / name
    
    # Check if already completely downloaded
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  ✓ Already downloaded: {name} -- skipping")
        return True
    
    if dest.exists():
        actual_size = dest.stat().st_size
        print(f"  ⚠ Incomplete file found: {name} ({actual_size / 1e9:.2f} / {expected_size / 1e9:.1f} GB)")
        print(f"    Will retry download...")
    
    url = f"{ZENODO_API}/files/{name}/content"
    
    # Retry loop
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Downloading {name} ({expected_size / 1e9:.1f} GB) [attempt {attempt}/{MAX_RETRIES}]...")
            
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                downloaded = 0
                
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded / expected_size * 100
                        print(f"\r    {pct:.1f}%  ({downloaded / 1e9:.2f} / {expected_size / 1e9:.2f} GB)",
                            end="", flush=True)
                
                print()  
            
            # Verify the download completed fully
            final_size = dest.stat().st_size
            if final_size == expected_size:
                print(f"  ✓ Download complete: {name}")
                return True
            else:
                print(f"  ✗ Download incomplete: {final_size / 1e9:.2f} / {expected_size / 1e9:.1f} GB")
                if attempt < MAX_RETRIES:
                    print(f"    Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
        
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"\n  ✗ Connection interrupted: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            continue
        
        except requests.exceptions.Timeout as e:
            print(f"\n  ✗ Timeout: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            continue
        
        except requests.exceptions.ConnectionError as e:
            print(f"\n  ✗ Connection error: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            continue
        
        except Exception as e:
            print(f"\n  ✗ Unexpected error: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            continue
    
    # All retries exhausted
    print(f"  ✗ FAILED after {MAX_RETRIES} attempts: {name}")
    return False

def extract_include50(zip_path, needed_paths):
    
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member in needed_paths:
                dest = VIDEOS_DIR / member
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
    return extracted

def main():
    # Load metadata
    if not Path(METADATA_CSV).exists():
        print(f"ERROR: {METADATA_CSV} not found.")
        print("Run stage2_planning.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    needed_paths = set(df["video_path"])
    print(f"INCLUDE-50 videos needed:  {len(needed_paths)}")
    print(f"Zip files to process:      {len(ZIP_FILES)}")
    print(f"Total download size:        56.8 GB")
    print(f"Output folder:              {VIDEOS_DIR.resolve()}")
    print(f"Retry attempts per file:    {MAX_RETRIES}")
    print()
    print("Each zip is deleted immediately after extraction.")
    print("Safe to interrupt with Ctrl+C and re-run -- progress is saved.\n")

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    VIDEOS_DIR.mkdir(exist_ok=True)

    total_extracted = 0
    failed_files = []

    for i, (name, size) in enumerate(ZIP_FILES, 1):
        print(f"[{i:02d}/{len(ZIP_FILES)}] {name}")
        
        # Try to download with retries
        success = download_file(name, size)
        
        if not success:
            failed_files.append(name)
            print(f"  ⚠ Skipping extraction for {name} (download failed)\n")
            continue
        
        # Extract INCLUDE-50 files
        zip_path = DOWNLOAD_DIR / name
        print(f"  Extracting INCLUDE-50 files...")
        count = extract_include50(zip_path, needed_paths)
        total_extracted += count
        print(f"  Extracted {count} new files  ({total_extracted}/{len(needed_paths)} total so far)")
        
        # Delete the zip to reclaim space
        zip_path.unlink()
        print(f"  Deleted {name}\n")

    print("=" * 60)
    print(f"Download complete. {total_extracted} files extracted to '{VIDEOS_DIR}/'")
    print("=" * 60)

    if failed_files:
        print(f"\n⚠ WARNING: {len(failed_files)} file(s) failed to download after {MAX_RETRIES} attempts:")
        for f in failed_files:
            print(f"  - {f}")
        print("\nYou can try re-running the script to retry these files.")
    else:
        print("\n All files downloaded successfully!")

    # Verify completeness
    found = set()
    for p in VIDEOS_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(VIDEOS_DIR).as_posix()
            found.add(rel)

    missing = needed_paths - found
    if missing:
        print(f"\n⚠ {len(missing)} expected video files still missing from extraction:")
        for m in sorted(missing)[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more.")
    else:
        print(" All 943 expected INCLUDE-50 video files accounted for.")


if __name__ == "__main__":
    main()
