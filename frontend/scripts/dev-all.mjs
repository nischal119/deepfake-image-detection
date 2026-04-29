import fs from "node:fs"
import path from "node:path"
import net from "node:net"
import { spawn, spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const repoRoot = path.resolve(__dirname, "..", "..")
const frontendDir = path.resolve(__dirname, "..")
const backendDir = path.join(repoRoot, "backend")
const videoProjectDir = path.join(repoRoot, "projects", "deepfake-detector-video")
const venvDir = path.join(repoRoot, ".venv")
const venvBin = path.join(venvDir, "bin")

const backendPort = 5003
const redisPort = 6379
const redisCeleryDb = 1
const postgresPort = 5432

const cacheDir = path.join(frontendDir, ".cache", "dev-all")
const pidsDir = path.join(cacheDir, "pids")
const logsDir = path.join(cacheDir, "logs")
const backendConfigPath = path.join(cacheDir, "backend-config.json") // legacy marker; not used for db selection
fs.mkdirSync(pidsDir, { recursive: true })
fs.mkdirSync(logsDir, { recursive: true })
const frontendEnvPath = path.join(frontendDir, ".env")

function pidFile(name) {
  return path.join(pidsDir, `${name}.pid`)
}

function isPidAlive(pid) {
  if (!pid) return false
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

function readPid(name) {
  try {
    const s = fs.readFileSync(pidFile(name), "utf8").trim()
    const n = Number(s)
    return Number.isFinite(n) ? n : null
  } catch {
    return null
  }
}

function writePid(name, pid) {
  fs.writeFileSync(pidFile(name), String(pid))
}

function wait(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

async function isPortOpen(host, port, timeoutMs = 1500) {
  return await new Promise((resolve) => {
    const socket = net.createConnection({ host, port })
    let done = false

    const finish = (v) => {
      if (done) return
      done = true
      socket.destroy()
      resolve(v)
    }

    socket.setTimeout(timeoutMs)
    socket.once("connect", () => finish(true))
    socket.once("timeout", () => finish(false))
    socket.once("error", () => finish(false))
  })
}

async function waitForPort(host, port, timeoutMs) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (await isPortOpen(host, port)) return true
    await wait(500)
  }
  return false
}

function which(cmd) {
  const res = spawnSync("sh", ["-lc", `command -v ${cmd} || true`], { encoding: "utf8" })
  const out = (res.stdout || "").trim()
  return out.length > 0 ? out : null
}

function fileAppendStream(p) {
  return fs.openSync(p, "a")
}

function spawnToLog({ name, cmd, args, cwd, env }) {
  const outPath = path.join(logsDir, `${name}.log`)
  const fd = fileAppendStream(outPath)
  const child = spawn(cmd, args, {
    cwd,
    env,
    detached: true,
    stdio: ["ignore", fd, fd],
  })
  // Don't keep the parent process alive for detached children
  child.unref()
  writePid(name, child.pid)
}

function readBackendConfig() {
  try {
    const raw = fs.readFileSync(backendConfigPath, "utf8")
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function writeBackendConfig(config) {
  fs.writeFileSync(backendConfigPath, JSON.stringify(config, null, 2))
}

function loadEnvFileIntoProcess(envPath) {
  if (!fs.existsSync(envPath)) return
  const raw = fs.readFileSync(envPath, "utf8")
  for (const line0 of raw.split("\n")) {
    const line = line0.trim()
    if (!line || line.startsWith("#")) continue
    const eqIdx = line.indexOf("=")
    if (eqIdx === -1) continue
    const key = line.slice(0, eqIdx).trim()
    let val = line.slice(eqIdx + 1).trim()
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1)
    }
    if (process.env[key] === undefined) process.env[key] = val
  }
}

async function ensureRedisRunning() {
  if (await isPortOpen("127.0.0.1", redisPort)) return

  console.log("[dev-all] Starting Redis...")

  const redisServer = which("redis-server")
  if (!redisServer) {
    console.warn("[dev-all] redis-server not found. Start Redis manually on :6379")
    return
  }

  const child = spawn(redisServer, ["--port", String(redisPort), "--bind", "127.0.0.1"], {
    detached: true,
    stdio: "ignore",
  })
  child.unref()

  const ok = await waitForPort("127.0.0.1", redisPort, 10000)
  if (!ok) console.warn("[dev-all] Redis did not become ready in time. Continuing anyway.")
}

async function tryStartPostgres() {
  if (await isPortOpen("127.0.0.1", postgresPort, 1500)) return true

  console.log("[dev-all] Starting Postgres (best effort)...")

  const brew = which("brew")
  if (!brew) {
    console.warn("[dev-all] Homebrew not found; please start Postgres manually on :5432")
    return false
  }

  // Best-effort: install+start is slow and may fail; we'll fall back to SQLite if it doesn't work.
  const pgFormula = "postgresql@16"
  spawnSync("brew", ["list", "--versions", pgFormula], { stdio: "ignore" })

  // If brew services is available, try starting. (If it fails, we'll fall back.)
  spawnSync("brew", ["services", "start", pgFormula], { stdio: "inherit" })

  const ok = await waitForPort("127.0.0.1", postgresPort, 60000)
  if (!ok) return false

  return true
}

function getDatabaseUrlFromEnv() {
  const databaseUrl = process.env.DATABASE_URL || "postgresql://localhost/deepfake_video"
  if (databaseUrl.startsWith("sqlite:") || databaseUrl.includes(":./dev.db")) {
    throw new Error(
      "[dev-all] SQLite is not allowed. Set DATABASE_URL to a Postgres connection string in `frontend/.env`."
    )
  }
  if (!databaseUrl.startsWith("postgresql://") && !databaseUrl.startsWith("postgres://")) {
    throw new Error(
      `[dev-all] DATABASE_URL must be a Postgres URL. Got: ${databaseUrl}`
    )
  }
  return databaseUrl
}

async function ensureBackendRunning({ databaseUrl }) {
  if (await isPortOpen("127.0.0.1", backendPort, 1000)) {
    throw new Error(
      `[dev-all] Backend port :${backendPort} is already in use. Stop the existing backend and rerun.`
    )
  }

  console.log(`[dev-all] Starting backend on :${backendPort} (Postgres)...`)

  if (!fs.existsSync(path.join(venvBin, "python"))) {
    console.warn("[dev-all] Python venv not found at ./.venv. Install backend deps first (backend/requirements.txt).")
  }

  const python = path.join(venvBin, "python")
  const code = `
import os
from app import create_app
app = create_app()
app.run(host="0.0.0.0", port=${backendPort}, debug=False, use_reloader=False)
`

  const env = {
    ...process.env,
    PYTHONPATH: `${backendDir}:${videoProjectDir}:${process.env.PYTHONPATH || ""}`,
    DATABASE_URL: databaseUrl,
    CELERY_BROKER_URL: `redis://localhost:${redisPort}/${redisCeleryDb}`,
    CELERY_RESULT_BACKEND: `redis://localhost:${redisPort}/${redisCeleryDb}`,
    FLASK_ENV: "development",
  }

  const fd = fileAppendStream(path.join(logsDir, "backend.log"))
  const child = spawn(python, ["-c", code], {
    cwd: backendDir,
    env,
    detached: true,
    stdio: ["ignore", fd, fd],
  })
  child.unref()
  writePid("backend", child.pid)
  writeBackendConfig({ databaseUrl }) // legacy marker

  // Wait for the API to respond (or fail fast).
  const ready = await waitForPort("127.0.0.1", backendPort, 20000)
  if (!ready) return false
  return true
}

async function ensureCeleryRunning({ databaseUrl }) {
  const existingPid = readPid("celery")
  if (isPidAlive(existingPid)) return true

  console.log("[dev-all] Starting Celery worker...")

  const celery = path.join(venvBin, "celery")
  if (!fs.existsSync(celery)) {
    console.warn("[dev-all] celery binary not found in ./.venv/bin. Run backend deps install first.")
    return false
  }

  const env = {
    ...process.env,
    PYTHONPATH: `${backendDir}:${videoProjectDir}:${process.env.PYTHONPATH || ""}`,
    DATABASE_URL: databaseUrl,
    CELERY_BROKER_URL: `redis://localhost:${redisPort}/${redisCeleryDb}`,
    CELERY_RESULT_BACKEND: `redis://localhost:${redisPort}/${redisCeleryDb}`,
  }

  spawnToLog({
    name: "celery",
    cmd: celery,
    args: [
      "-A",
      "celery_app",
      "worker",
      "--loglevel=info",
      "-P",
      "solo",
      "--concurrency=1",
      "--without-heartbeat",
      "--without-gossip",
      "--without-mingle",
    ],
    cwd: backendDir,
    env,
  })

  return true
}

async function main() {
  loadEnvFileIntoProcess(frontendEnvPath)

  await ensureRedisRunning()

  const postgresUp = await tryStartPostgres()
  if (!postgresUp) {
    throw new Error(
      "[dev-all] Postgres is not running/reachable on localhost:5432. Start Postgres first (no SQLite fallback)."
    )
  }

  const databaseUrl = getDatabaseUrlFromEnv()

  // Start backend + celery on isolated Redis DB to avoid interference with any older dev processes.
  await ensureBackendRunning({ databaseUrl })
  await ensureCeleryRunning({ databaseUrl })

  // Ensure Prisma is pointed at Postgres and schema is applied.
  console.log("[dev-all] Running Prisma generate + migrations...")
  const prismaEnv = { ...process.env, DATABASE_URL: databaseUrl }
  const prismaGen = spawnSync("pnpm", ["prisma:generate"], { cwd: frontendDir, env: prismaEnv, stdio: "inherit" })
  if (prismaGen.status !== 0) throw new Error("[dev-all] prisma:generate failed.")
  const prismaMig = spawnSync("pnpm", ["prisma:migrate"], { cwd: frontendDir, env: prismaEnv, stdio: "inherit" })
  if (prismaMig.status !== 0) throw new Error("[dev-all] prisma:migrate failed.")

  // Launch Next.js (foreground).
  console.log(`[dev-all] Starting Next.js (FLASK_VIDEO_API_URL=http://localhost:${backendPort})...`)

  const nextBin = path.join(frontendDir, "node_modules", ".bin", "next")
  if (!fs.existsSync(nextBin)) {
    console.warn("[dev-all] Next.js binary not found (node_modules missing?). Running `pnpm dev` might fail if deps aren't installed.")
  }

  const env = {
    ...process.env,
    FLASK_VIDEO_API_URL: `http://localhost:${backendPort}`,
  }

  const next = fs.existsSync(nextBin) ? nextBin : "next"
  const child = spawn(next, ["dev"], { cwd: frontendDir, env, stdio: "inherit" })
  child.on("exit", (code) => process.exit(code ?? 0))
}

main().catch((e) => {
  console.error("[dev-all] Startup failed:", e)
  process.exit(1)
})

