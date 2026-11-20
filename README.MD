# AthleteLens

AthleteLens is an academic prototype for preliminary video-based sports movement assessment. Assessment outputs are experimental and are not scientifically validated.

## Web application

The React, Vite, TypeScript, Tailwind CSS, and React Router application is in `web/`.

```powershell
cd web
npm install
Copy-Item .env.example .env
npm run dev
```

Vite serves the frontend at `http://localhost:5173` and forwards `/api` requests to `http://127.0.0.1:8000`. Start the FastAPI backend separately before connecting assessment requests. Set `VITE_API_BASE_URL` in `web/.env` to use a different API base URL.

Create and preview a production build with:

```powershell
npm run build
npm run preview
```

### Demo access

The frontend includes client-only demo role access; it is not production authentication. Use these public demo credentials:

| Role | Email | Password |
| --- | --- | --- |
| Athlete | `athlete@demo.athletelens` | `athlete-demo` |
| Administrator | `admin@demo.athletelens` | `admin-demo` |

The demo role session and actual assessment API responses are stored in this browser's local storage. The administrator diagnostics page can inspect only responses saved in that browser. No credentials, videos, users, or assessments are stored by the backend, and clearing browser storage removes the local demo session and history. Registration remains unavailable.

Assessment MP4 visualizations are written to `backend/storage/annotated_videos/` by default and are ignored by Git. Set `ATHLETELENS_ANNOTATED_VIDEO_DIR` to choose another local output directory. The API returns a generated-video URL only after export and full decode verification; the original uploaded video remains temporary and is removed after processing.
