// Local development server: `npm run dev` (reads backend/.env). On Vercel, api/index.mjs serves the
// same app as a serverless function instead.
import { createApp } from "./app.js";
import { supabaseRepo } from "./repo/supabase.js";

const port = Number(process.env.PORT) || 3001;
createApp({ repo: supabaseRepo }).listen(port, () => {
  console.log(`TaBet API on http://localhost:${port}/api`);
});
