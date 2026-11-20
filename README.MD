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

The demo role session is stored in this browser. Completed assessment responses are persisted by the backend and associated with the client-supplied demo email; this is a demo convenience, not verified identity or production authentication. The browser retains a local copy for the immediate result view, while athlete history and admin diagnostics read records from the backend. Registration remains unavailable.

Assessment history uses SQLAlchemy 2.x and Alembic. Local development defaults to SQLite at `backend/storage/assessments.sqlite3`; set `DATABASE_URL` to choose another database. Supabase PostgreSQL is supported with `postgresql+psycopg://...`. The legacy `ATHLETELENS_ASSESSMENT_DATABASE` path remains a fallback for older local configurations. Apply schema changes with Alembic (`alembic upgrade head`). The admin history endpoint uses a client-supplied demo role marker and is not secure authorization.

Assessment MP4 visualizations are written to `backend/storage/annotated_videos/` by default and are ignored by Git. Set `ATHLETELENS_ANNOTATED_VIDEO_DIR` to choose another local output directory. The API returns a generated-video URL only after export and full decode verification; the original uploaded video remains temporary and is removed after processing.

## Local network demonstration

The Vite development server listens on all laptop interfaces at port 5173. Its `/api` proxy forwards to FastAPI at `127.0.0.1:8000` on the same laptop. The browser therefore uses same-origin API and annotated-video URLs; do not set `VITE_API_BASE_URL` to `localhost` for the phone. `VITE_API_PROXY_TARGET` can change the server-side proxy destination; its default is `http://127.0.0.1:8000`.

### Start both services

From the repository root, run this script in PowerShell:

```powershell
.\scripts\start-lan-demo.ps1
```

The script discovers an IPv4 address on an adapter with a default gateway, adds that address to the local CORS allowlist, and opens visible PowerShell windows for FastAPI and Vite. Keep both windows open. If automatic discovery chooses the wrong adapter, pass the laptop Wi-Fi IPv4 address explicitly:

```powershell
.\scripts\start-lan-demo.ps1 -LanIPv4 192.168.1.25
```

Ports 8000 and 5173 must be free first. If an earlier backend or Vite window is still running, stop it with Ctrl+C before using the startup script.

To start services manually in separate PowerShell windows:

```powershell
# Window 1: backend
Set-Location .\backend
$env:ATHLETELENS_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://<LAPTOP-LAN-IP>:5173"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Window 2: frontend
Set-Location .\web
$env:VITE_API_BASE_URL = "/api"
npm run dev -- --host 0.0.0.0 --port 5173
```

Find the laptop IPv4 address with:

```powershell
Get-NetIPConfiguration | Where-Object IPv4DefaultGateway | Select-Object InterfaceAlias,IPv4Address
```

On the phone, join the same trusted Wi-Fi network and open:

```text
http://<LAPTOP-LAN-IP>:5173
```

For example, if the laptop address is `192.168.1.25`, open `http://192.168.1.25:5173`. The backend health check is `http://<LAPTOP-LAN-IP>:8000/health`. Assessment calls and annotated MP4 retrievals normally pass through the Vite `/api` proxy, so the phone does not need direct access to port 8000.

Windows Firewall must allow inbound TCP port 5173 on the laptop's **Private** network profile. If opening the backend health URL directly from the phone, also allow TCP port 8000. An administrator can add narrow Private-profile rules when needed:

```powershell
New-NetFirewallRule -DisplayName "AthleteLens Vite demo (Private)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5173 -Profile Private
New-NetFirewallRule -DisplayName "AthleteLens API demo (Private)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
```

Use only a trusted private network, avoid guest Wi-Fi or client-isolated networks, and remove temporary firewall rules after the demonstration if they are no longer needed. This setup is for a shared local network only; it does not publish the application to the internet. Verify phone-browser upload and MP4 playback manually on the demonstration device.

## Production deployment preparation

The backend image uses Python 3.14 on Debian Bookworm, installs the pinned AI package through `backend/requirements.txt`, runs dependency/import checks during image build, applies Alembic migrations before startup, listens on Railway's `PORT` (with 8000 as the local default), and runs as an unprivileged user. The official Lite pose model is downloaded at build time from its pinned MediaPipe URL and checked against a SHA-256 digest. The API image does not contain uploaded source videos or experimental datasets.

### Build and run the Docker backend locally

Docker Engine and the Compose plugin must be installed. From the repository root:

```powershell
New-Item -ItemType Directory -Force backend/storage/models, backend/storage/annotated_videos
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -OutFile backend/storage/models/pose_landmarker_lite.task
Copy-Item .env.docker.example .env
notepad .env
```

Set `ATHLETELENS_CORS_ORIGINS` to the exact frontend browser origin. For the local Vite app, use `http://localhost:5173`; for Vercel, use its exact `https://...vercel.app` origin (or configured custom domain). Comma-separated exact origins are allowed. Wildcards and URL paths are rejected.

```powershell
docker compose -f compose.yaml up --build -d
docker compose -f compose.yaml ps
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Assessment history, generated MP4s, and the model file live under the mounted host directory (`ATHLETELENS_DATA_DIR`, default `backend/storage`). The Compose example defaults `DATABASE_URL` to the mounted SQLite file. Keep that directory on persistent storage and back it up. Use `docker compose -f compose.yaml logs -f backend` to inspect startup errors; stop the service with `docker compose -f compose.yaml down` (the mounted data directory is retained). Set `DATABASE_URL` in `.env` to a PostgreSQL URL to use Supabase instead; annotated MP4s still stay in the mounted file directory, never in PostgreSQL.

### Deploy the backend on Railway

This repository is prepared for Railway, but no Railway service has been created or deployed. Create a Railway project from this repository and add a backend service. In the service's **Settings → Build** configuration, use the repository root (`/`) as the source/root directory and Dockerfile builder with path `/Dockerfile`. Railway detects the root `Dockerfile` automatically; do not point the service at `backend/` because the Docker build copies the AI package from the repository root. Leave the dashboard Start Command blank to use the Dockerfile command. It runs `alembic upgrade head` and then starts Uvicorn on `0.0.0.0:$PORT` (falling back to 8000 for local Docker).

In **Settings → Deploy**, set the Healthcheck Path to `/health`. The Docker image healthcheck also follows `PORT`. In **Variables**, add:

```text
DATABASE_URL=<Supabase PostgreSQL Session Pooler URL using postgresql+psycopg:// and sslmode=require>
ATHLETELENS_POSE_MODEL=/app/ai_engine/models/pose_landmarker_lite.task
ATHLETELENS_ANNOTATED_VIDEO_DIR=/data/annotated_videos
ATHLETELENS_CORS_ORIGINS=https://your-project.vercel.app
ATHLETELENS_MAX_UPLOAD_SIZE_BYTES=104857600
RAILWAY_RUN_UID=0
```

Railway injects `PORT`; do not hardcode or manually set it. `DATABASE_URL` is a secret: set it only in Railway Variables, not in Git or frontend configuration. Set `ATHLETELENS_CORS_ORIGINS` to the exact deployed Vercel origin(s), with comma-separated origins only if needed; do not use `*`. The upload-size variable is optional and defaults to 100 MiB.

Attach a Railway Volume to the backend service and set its **Mount Path** to exactly `/data/annotated_videos`. Generated MP4 files are written there; assessment JSON remains in Supabase PostgreSQL. Railway mounts volumes with root ownership. This image runs as UID 10001, so keep `RAILWAY_RUN_UID=0` set as Railway documents for services with non-root images and volumes, allowing the app to write annotated videos. The model is included in the image at `/app/ai_engine/models/pose_landmarker_lite.task`; it does not require a separate model volume. The Docker build downloads the official pinned model and verifies its checksum.

The Dockerfile uses `alembic upgrade head` in its startup command; keep the Start Command override empty unless reproducing that migration command. If you override it, use:

```sh
sh -c 'alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"'
```

The service health-check path is `/health` (returns HTTP 200). A successful build or local migration does not mean the Railway deployment is complete; verify the deployment logs, public health URL, assessment endpoints, and annotated-video retrieval after deploying. See Railway's [Dockerfile](https://docs.railway.com/builds/dockerfiles), [healthcheck](https://docs.railway.com/deployments/healthchecks), and [volume](https://docs.railway.com/volumes) documentation for current dashboard details.

### Create and connect a Supabase PostgreSQL project

1. Create a Supabase project and set/save its database password.
2. In the project dashboard, open **Connect**. For a long-running backend, use the direct connection when the host supports its IP version. For IPv4-only hosts, select **Session pooler** and copy that connection string; use the host and username Supabase provides rather than constructing them.
3. Convert its scheme to `postgresql+psycopg://` for SQLAlchemy and append `?sslmode=require` (or retain the supplied SSL query options). URL-encode reserved characters in the password. Keep the connection string in the backend host's secret/environment settings, never in frontend/Vercel variables or Git.
4. Set `DATABASE_URL` on the backend. The Docker entrypoint runs `alembic upgrade head` before Uvicorn. To apply a migration manually from the backend environment, run `alembic upgrade head` with the same `DATABASE_URL` exported.

To copy existing local SQLite assessment history into the configured PostgreSQL database, run this one-time, idempotent transfer from `backend/` after setting `DATABASE_URL` to the Supabase URL:

```powershell
Set-Location backend
$env:DATABASE_URL = "postgresql+psycopg://postgres.<project-ref>:<url-encoded-password>@<pooler-host>:5432/postgres?sslmode=require"
.\.venv\Scripts\python.exe -m app.db.copy_history --source-url "sqlite:///./storage/assessments.sqlite3"
```

The transfer preserves record IDs, timestamps, response JSON, configuration, and video identifiers. It does not copy MP4 files; move any existing annotated MP4s separately into the backend's configured persistent video directory if those historic links should keep resolving.

Supabase provides the connection string in the dashboard's **Connect** panel and recommends SSL; its shared Session pooler is the IPv4-compatible option when direct connectivity is unavailable. See [Supabase database connection guidance](https://supabase.com/docs/guides/database/connecting-to-postgres).

### Deploy the frontend to Vercel

Import the repository into Vercel and set the project root directory to `web`. Use the Vite framework preset, build command `npm run build`, and output directory `dist`. Add this Vercel environment variable for Production (and Preview only if the matching preview domain is added to backend CORS):

```text
VITE_API_BASE_URL=https://your-backend-host.example.com/api
```

Set the value to the backend's public HTTPS URL plus `/api`, not `/api/v1`; frontend API paths append `/v1/...`. Vite embeds this variable at build time, so redeploy after changing it. Add the exact Vercel site origin (without `/api`) to `ATHLETELENS_CORS_ORIGINS` on the backend. The backend must be reachable from the public internet over HTTPS for Vercel users, but this configuration does not publish or provision a cloud service by itself.

For manual endpoint smoke checks, use the deployed OpenAPI page at `https://your-backend-host.example.com/docs` to confirm all four POST routes are present. Real assessment requests also require a valid pose model file and a supported video upload; the automated backend suite exercises all four route contracts using mocked processors.
