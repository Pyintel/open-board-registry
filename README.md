# 🔌 Open Board Registry

An open-source database and JSON registry containing structured specifications for **1,700+ microcontroller boards**. It consolidates technical details from multiple ecosystems (PlatformIO, vendor documentation, and datasheets) into a single, queryable SQLite database and standardized JSON file.

This database provides deep hardware specifications that toolchains typically omit, including operating voltages, onboard peripherals (sensors, screens, buttons), and communication interfaces, alongside standard compiler configurations (RAM, ROM/Flash limits, and USB VIDs/PIDs).

---

## 📦 What's Inside

1. **[boards.db](file:///P:/Projects/open-board-registry/boards.db)**: A fully indexed SQLite 3 database.
2. **[boards.json](file:///P:/Projects/open-board-registry/boards.json)**: A single, comprehensive JSON file containing the full structured dataset.

---

## 🗃️ SQLite Schema

The `boards` table is defined as follows:

```sql
CREATE TABLE boards (
    id TEXT PRIMARY KEY,       -- PlatformIO / board identifier (e.g., "adafruit_pygamer_m4")
    name TEXT,                 -- Display / Marketing Name (e.g., "Adafruit PyGamer M4 Express")
    vendor TEXT,               -- Board Vendor (e.g., "Adafruit", "SparkFun", "ST")
    platform TEXT,             -- Target Toolchain Platform (e.g., "atmelsam", "espressif32")
    mcu TEXT,                  -- Microcontroller Unit name (e.g., "SAMD51J19A")
    cpu_arch TEXT,             -- CPU Core Architecture (e.g., "cortex-m4")
    fcpu_hz INTEGER,           -- CPU Frequency in Hertz (e.g., 120000000)
    ram_bytes INTEGER,         -- Maximum RAM size in bytes (e.g., 196608)
    rom_bytes INTEGER,         -- Maximum Flash/ROM size in bytes (e.g., 524288)
    vids TEXT,                 -- JSON array of USB Vendor IDs (e.g., ["239A"])
    pids TEXT,                 -- JSON array of USB Product IDs (e.g., ["803D"])
    connectivity TEXT,         -- JSON array of connectivity options (e.g., ["wifi", "bluetooth"])
    frameworks TEXT,           -- JSON array of supported frameworks (e.g., ["arduino", "zephyr"])
    protocols TEXT,            -- JSON array of upload/debug protocols (e.g., ["sam-ba", "jlink"])
    url TEXT,                  -- Official product or documentation URL
    voltage TEXT,              -- Board operating voltage (e.g., "3.3V")
    screen INTEGER,            -- 1 if the board has a built-in display, 0 otherwise
    peripherals TEXT,          -- JSON array of onboard sensors, buttons, screens, and LEDs
    interfaces TEXT            -- JSON array of physical header interfaces (e.g., ["I2C", "SPI", "UART"])
);
```

---

## 🚀 Quick Start Examples

### Reading the Database in Python

You can easily query the SQLite database locally using Python's built-in `sqlite3` module:

```python
import sqlite3
import json

# Connect to the database
conn = sqlite3.connect("boards.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Find all boards that feature a built-in screen
cursor.execute("SELECT name, mcu, peripherals FROM boards WHERE screen = 1")
rows = cursor.fetchall()

for row in rows:
    peripherals = json.loads(row["peripherals"])
    print(f"Board: {row['name']} | MCU: {row['mcu']}")
    print(f"Peripherals: {', '.join(peripherals)}\n")

conn.close()
```

### Parsing the JSON Registry in Node.js

```javascript
const fs = require('fs');

// Read the JSON file
const rawData = fs.readFileSync('boards.json', 'utf8');
const boards = JSON.parse(rawData);

// Filter for Espressif boards with WiFi connectivity
const wifiEsp32 = boards.filter(b => 
  b.vendor.toLowerCase() === 'espressif' && 
  b.connectivity.includes('wifi')
);

console.log(`Found ${wifiEsp32.length} WiFi-enabled Espressif boards.`);
```

---

## 🤝 Open Source License

This database is compiled from open-source toolchain indexes under the **Apache License 2.0**. Feel free to use it in your own build systems, hardware cataloging apps, or local IDE plugins!
