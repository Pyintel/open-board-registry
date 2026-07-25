const path = require("path");
const fs = require("fs");
const { Database } = require("bun:sqlite");

module.exports = {
  query_board_spec: {
    description: "Look up detailed specs, MCU, RAM/Flash limits, operating voltage, and onboard peripherals for a development board.",
    args: {
      query: { type: "string", description: "Board name, vendor, MCU, or VID:PID (e.g. PyPortal, ESP32, 239A:8036)" }
    },
    async execute({ query }) {
      const dbPath = path.join(__dirname, "..", "boards.db");
      if (!fs.existsSync(dbPath)) return "Board database file not found in module.";
      const db = new Database(dbPath, { readonly: true });
      try {
        const q = query.trim();
        let rows = [];
        if (q === "*" || q === "" || q.toLowerCase() === "all") {
          rows = db.query("SELECT * FROM boards LIMIT 50").all();
        } else {
          rows = db.query(`
            SELECT * FROM boards 
            WHERE id LIKE ? OR name LIKE ? OR mcu LIKE ? OR platform LIKE ? OR vendor LIKE ?
            LIMIT 25
          `).all(`%${q}%`, `%${q}%`, `%${q}%`, `%${q}%`, `%${q}%`);
        }
        return JSON.stringify(rows, null, 2);
      } finally {
        db.close();
      }
    }
  }
};
