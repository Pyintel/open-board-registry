import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let cachedBoards = null;

function loadBoards() {
  if (cachedBoards) return cachedBoards;
  const jsonPath = path.join(__dirname, "..", "boards.json");
  if (fs.existsSync(jsonPath)) {
    try {
      cachedBoards = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
      return cachedBoards;
    } catch {}
  }
  return [];
}

export const query_board_spec = {
  description: "Look up detailed specs, MCU, RAM/Flash limits, operating voltage, and onboard peripherals for a development board.",
  args: {
    query: { type: "string", description: "Board name, vendor, MCU, or VID:PID (e.g. PyPortal, ESP32, 239A:8036)" }
  },
  async execute({ query }) {
    const boards = loadBoards();
    if (!boards || boards.length === 0) return "Board database file not found in module.";
    
    const q = (query || "").trim().toLowerCase();
    let matches = [];

    if (q === "*" || q === "" || q === "all") {
      matches = boards.slice(0, 50);
    } else {
      matches = boards.filter((b) => {
        const idMatch = b.id && b.id.toLowerCase().includes(q);
        const nameMatch = b.name && b.name.toLowerCase().includes(q);
        const mcuMatch = b.mcu && b.mcu.toLowerCase().includes(q);
        const platformMatch = b.platform && b.platform.toLowerCase().includes(q);
        const vendorMatch = b.vendor && b.vendor.toLowerCase().includes(q);
        const vidMatch = Array.isArray(b.vids) && b.vids.some((v) => String(v).toLowerCase().includes(q));
        const pidMatch = Array.isArray(b.pids) && b.pids.some((p) => String(p).toLowerCase().includes(q));
        return idMatch || nameMatch || mcuMatch || platformMatch || vendorMatch || vidMatch || pidMatch;
      }).slice(0, 25);
    }

    return JSON.stringify(matches, null, 2);
  }
};

export const count_boards = {
  description: "Get the exact total count of development boards available in the registry.",
  args: {},
  async execute() {
    const boards = loadBoards();
    return JSON.stringify({ total_boards: boards.length }, null, 2);
  }
};

export const hw_board_registry = query_board_spec;
export const board_spec = query_board_spec;
