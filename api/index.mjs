// Vercel serverless entry: every /api/* request is rewritten here (vercel.json) and handled by the
// Express app in backend/.
import { createApp } from "../backend/src/app.js";
import { supabaseRepo } from "../backend/src/repo/supabase.js";

export default createApp({ repo: supabaseRepo });
