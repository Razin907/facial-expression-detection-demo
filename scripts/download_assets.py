import os
import urllib.request

def download_emojis():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    assets_dir = os.path.join(base_dir, 'assets', 'filters')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Twemoji unicode map for 7 basic emotions
    # Using 72x72 PNGs from Twemoji
    emojis = {
        'happy': '1f60d',      # 😍 Heart eyes
        'sad': '1f622',        # 😢 Crying
        'angry': '1f621',      # 😡 Pouting/Angry
        'surprise': '1f632',   # 😲 Astonished
        'fear': '1f628',       # 😨 Fearful
        'disgust': '1f922',    # 🤢 Nauseated
        'neutral': '1f610'     # 😐 Neutral
    }
    
    base_url = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/72x72/"
    
    print(f"Mengunduh aset emoji ke {assets_dir}...")
    for emotion, unicode_hex in emojis.items():
        url = f"{base_url}{unicode_hex}.png"
        filepath = os.path.join(assets_dir, f"{emotion}.png")
        try:
            print(f"Downloading {emotion} ({unicode_hex})...")
            urllib.request.urlretrieve(url, filepath)
            print(f"[OK] Berhasil disimpan: {filepath}")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh {emotion}: {e}")

if __name__ == "__main__":
    download_emojis()
