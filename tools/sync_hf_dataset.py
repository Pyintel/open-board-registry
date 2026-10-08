"""
Automated CI/CD Sync Script: GitHub -> Hugging Face
Converts boards.json into optimized Apache Parquet and syncs pyintel/open-board-registry on Hugging Face Hub.
"""

import os
import json
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
BOARDS_JSON = REPO_ROOT / "boards.json"
BOARDS_PARQUET = REPO_ROOT / "boards.parquet"
HF_DATASET_ID = "pyintel/open-board-registry"

DATASET_CARD = """---
language:
- en
license: apache-2.0
task_categories:
- tabular-classification
- feature-extraction
tags:
- hardware
- microcontrollers
- embedded-systems
- electronics
- robotics
- pinouts
- platformio
- pyintel
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: boards.parquet
---

# 🔌 PyIntel Open Board Registry (`pyintel/open-board-registry`)

An open-source, ground-truth database and registry containing structured technical specifications for **1,700+ microcontroller boards and embedded development platforms**.

Maintained by **[PyIntel](https://github.com/Pyintel/open-board-registry)**.

---

## 📦 What's Inside

This dataset consolidates hardware details from multiple embedded ecosystems (PlatformIO, vendor documentation, and silicon datasheets) that toolchains typically omit:
* **Memory Limits:** Exact maximum RAM and Flash/ROM capacities in bytes.
* **Electrical Parameters:** Operating voltages (e.g., 3.3V, 5V).
* **Silicon Architecture:** MCU model, CPU architecture (Cortex-M0+/M4/M7, Xtensa LX6/LX7, RISC-V, AVR), and clock speed (Hz).
* **Communication & Bus Interfaces:** Supported header interfaces (I2C, SPI, UART, CAN, USB).
* **Connectivity:** Built-in WiFi, Bluetooth / BLE, LoRa, Ethernet.
* **Onboard Peripherals:** Built-in screens/displays, buttons, LEDs, and environmental sensors.
* **Toolchain Support:** Frameworks (Arduino, Zephyr, ESP-IDF) and upload/debug protocols.

---

## 📊 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Unique board identifier (e.g. `adafruit_pygamer_m4`) |
| `name` | `string` | Marketing / Display Name |
| `vendor` | `string` | Board Vendor (e.g. `Adafruit`, `Espressif`, `SparkFun`, `ST`) |
| `platform` | `string` | Target toolchain platform (e.g. `atmelsam`, `espressif32`) |
| `mcu` | `string` | Microcontroller Unit (e.g. `SAMD51J19A`, `ESP32-S3`) |
| `cpu_arch` | `string` | CPU core architecture |
| `fcpu_hz` | `int64` | Clock frequency in Hertz |
| `ram_bytes` | `int64` | Maximum RAM size in bytes |
| `rom_bytes` | `int64` | Maximum Flash/ROM size in bytes |
| `voltage` | `string` | Operating voltage (e.g. `3.3V`) |
| `screen` | `int64` | `1` if the board features a built-in display, `0` otherwise |
| `connectivity` | `list[string]` | Wireless & wired connectivity options |
| `frameworks` | `list[string]` | Supported frameworks (`arduino`, `zephyr`, etc.) |
| `interfaces` | `list[string]` | Physical bus headers (`I2C`, `SPI`, `UART`, `CAN`) |
| `peripherals` | `list[string]` | Onboard hardware components |

---

## 🚀 Quickstart in Python

```python
from datasets import load_dataset
import pandas as pd

# Load directly from Hugging Face
ds = load_dataset("pyintel/open-board-registry", split="train")
df = ds.to_pandas()

# Find all 3.3V boards with WiFi and built-in screen
wifi_screen_boards = df[(df['screen'] == 1) & (df['connectivity'].apply(lambda c: 'wifi' in c))]
print(f"Found {len(wifi_screen_boards)} matching boards!")
```

---

## 🔄 Automated Synchronization
This dataset is automatically synced directly from the **[Pyintel/open-board-registry](https://github.com/Pyintel/open-board-registry)** GitHub repository via GitHub Actions.
"""

def sync():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Missing HF_TOKEN environment variable.")

    print(f"📥 Loading {BOARDS_JSON}...")
    with open(BOARDS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔄 Converting {len(data):,} board entries to Apache Parquet...")
    df = pd.DataFrame(data)
    df.to_parquet(BOARDS_PARQUET, index=False)
    print(f"✅ Generated {BOARDS_PARQUET} ({BOARDS_PARQUET.stat().st_size:,} bytes)")

    print(f"🚀 Connecting to Hugging Face Hub as org 'pyintel'...")
    api = HfApi(token=hf_token)

    # Create/verify repo
    print(f"📦 Ensuring repository {HF_DATASET_ID} exists...")
    api.create_repo(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True
    )

    # Upload files
    print(f"⬆️ Uploading boards.parquet -> boards.parquet...")
    api.upload_file(
        path_or_fileobj=str(BOARDS_PARQUET),
        path_in_repo="boards.parquet",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="chore: auto-sync boards.parquet from GitHub"
    )

    print(f"⬆️ Uploading boards.json -> boards.json...")
    api.upload_file(
        path_or_fileobj=str(BOARDS_JSON),
        path_in_repo="boards.json",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="chore: auto-sync boards.json from GitHub"
    )

    print(f"⬆️ Uploading README.md (Dataset Card)...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        commit_message="docs: auto-sync dataset card from GitHub"
    )

    print(f"\n🎉 SUCCESS! Dataset is live at: https://huggingface.co/datasets/{HF_DATASET_ID}")

if __name__ == "__main__":
    sync()
