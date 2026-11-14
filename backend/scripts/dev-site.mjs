// Local preview of the whole site as Vercel serves it: the static pages plus the Express API at /api.
//   node --env-file=.env scripts/dev-site.mjs   ->  http://localhost:3000
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "../src/app.js";
import { supabaseRepo } from "../src/repo/supabase.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, "..", "..", "soccer (premier league)", "site");
const port = Number(process.env.PORT) || 3000;
const app = express();
if (process.env.API_DOWN) app.use("/api", (req, res) => res.status(503).end()); // test the fallback
app.use(express.static(site, { extensions: ["html"] }));     // like Vercel's cleanUrls
app.use(createApp({ repo: supabaseRepo }));                  // /api/*, plus JSON 404s for anything else
app.listen(port, () => console.log(`TaBet site + API on http://localhost:${port}`));
