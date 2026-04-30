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

const backendPort = 5001
let frontendPort = 3000
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

function hasDockerCompose() {
  const docker = which("docker")
  if (!docker) return false
  const res = spawnSync("docker", ["compose", "version"], { encoding: "utf8" })
  return res.status === 0
}

function fileAppendStream(p) {
  return fs.openSync(p, "a")
}

function assertPythonDepsAvailable(pythonPath) {
  if (!fs.existsSync(pythonPath)) {
    throw new Error(
      "[dev-all] Python venv not found at ./.venv. Create it and install backend deps:\n" +
        "  python -m venv .venv\n" +
        "  source .venv/bin/activate\n" +
        "  pip install -r backend/requirements.txt"
    )
  }

  const probe = spawnSync(
    pythonPath,
    [
      "-c",
      [
        "import flask",
        "import celery",
        "import redis",
        "import sqlalchemy",
        "import psycopg2",
        "import cv2",
        "print('ok')",
      ].join("; "),
    ],
    { encoding: "utf8" }
  )

  if (probe.status !== 0) {
    const stderr = (probe.stderr || "").trim()
    throw new Error(
      "[dev-all] Backend Python dependencies are missing in ./.venv.\n" +
        "Install them with:\n" +
        "  source .venv/bin/activate\n" +
        "  pip install -r backend/requirements.txt\n" +
        (stderr ? `\nPython error:\n${stderr}\n` : "")
    )
  }
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

async function chooseFrontendPort() {
  for (let p = 3000; p <= 3010; p += 1) {
    if (!(await isPortOpen("127.0.0.1", p, 300))) {
      frontendPort = p
      return
    }
  }
  throw new Error("[dev-all] No free frontend port found in range 3000-3010.")
}

async function ensureFrontendPortAvailable() {
  if (!(await isPortOpen("127.0.0.1", frontendPort, 500))) return

  const existingPid = readPid("next")
  if (isPidAlive(existingPid)) {
    console.log(`[dev-all] Next.js already running (pid ${existingPid}). Restarting...`)
    try {
      process.kill(existingPid, "SIGTERM")
    } catch {}
    await wait(500)
    return
  }

  console.warn(`[dev-all] Frontend port :${frontendPort} is in use. Choosing another port...`)
  await chooseFrontendPort()
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

  if (hasDockerCompose()) {
    console.log("[dev-all] Starting Redis (Docker Compose)...")
    const up = spawnSync("docker", ["compose", "up", "-d", "redis"], { cwd: repoRoot, stdio: "inherit" })
    if (up.status !== 0) throw new Error("[dev-all] Failed to start Redis via Docker Compose.")

    const ok = await waitForPort("127.0.0.1", redisPort, 20000)
    if (!ok) throw new Error("[dev-all] Redis did not become ready on :6379 in time.")
    return
  }

  console.log("[dev-all] Starting Redis (local redis-server)...")
  const redisServer = which("redis-server")
  if (!redisServer) {
    throw new Error(
      "[dev-all] Redis is not reachable on :6379, and neither Docker Compose nor redis-server are available."
    )
  }

  const child = spawn(redisServer, ["--port", String(redisPort), "--bind", "127.0.0.1"], {
    detached: true,
    stdio: "ignore",
  })
  child.unref()

  const ok = await waitForPort("127.0.0.1", redisPort, 10000)
  if (!ok) throw new Error("[dev-all] Redis did not become ready on :6379 in time.")
}

async function tryStartPostgres() {
  if (await isPortOpen("127.0.0.1", postgresPort, 1500)) return true

  if (hasDockerCompose()) {
    console.log("[dev-all] Starting Postgres (Docker Compose)...")
    const up = spawnSync("docker", ["compose", "up", "-d", "postgres"], { cwd: repoRoot, stdio: "inherit" })
    if (up.status !== 0) return false

    const ok = await waitForPort("127.0.0.1", postgresPort, 60000)
    return ok
  }

  console.log("[dev-all] Starting Postgres (local Homebrew service, best effort)...")
  const brew = which("brew")
  if (!brew) return false

  const pgFormula = "postgresql@16"
  spawnSync("brew", ["services", "start", pgFormula], { stdio: "inherit" })

  const ok = await waitForPort("127.0.0.1", postgresPort, 60000)
  return ok
}

function getDatabaseUrlFromEnv() {
  const localUser = process.env.USER || process.env.LOGNAME || "postgres"
  const fallback = hasDockerCompose()
    ? "postgresql://postgres:postgres@localhost:5432/deepfake_app"
    : `postgresql://${localUser}@localhost:5432/deepfake_app`

  const envVal = process.env.DATABASE_URL
  const databaseUrl = envVal || fallback

  if (!hasDockerCompose() && envVal && envVal.includes("postgres:postgres@localhost")) {
    console.warn(
      `[dev-all] DATABASE_URL looks like a Docker credential (${envVal}), but Docker Compose is not available. Falling back to local user URL: ${fallback}`
    )
    return fallback
  }

  if (databaseUrl.startsWith("sqlite:") || databaseUrl.includes(":./dev.db")) {
    if (hasDockerCompose()) {
      console.warn(
        `[dev-all] DATABASE_URL is set to SQLite (${databaseUrl}). Overriding to Postgres for this dev stack: ${fallback}`
      )
      return fallback
    }
    throw new Error(
      `[dev-all] SQLite is not allowed. Set DATABASE_URL to a Postgres URL in frontend/.env (example: ${fallback}).`
    )
  }
  if (!databaseUrl.startsWith("postgresql://") && !databaseUrl.startsWith("postgres://")) {
    throw new Error(
      `[dev-all] DATABASE_URL must be a Postgres URL. Got: ${databaseUrl}`
    )
  }
  return databaseUrl
}

function getDbNameFromUrl(databaseUrl) {
  try {
    const u = new URL(databaseUrl)
    const name = u.pathname.replace(/^\//, "")
    return name || null
  } catch {
    return null
  }
}

function ensurePostgresDbExists(databaseUrl) {
  const dbName = getDbNameFromUrl(databaseUrl)
  if (!dbName) return

  const createdb = which("createdb")
  if (!createdb) return

  // Best-effort: if DB already exists, createdb exits non-zero; that's fine.
  spawnSync(createdb, [dbName], { stdio: "ignore" })
}

function getBackendDatabaseUrlFromEnv(prismaDatabaseUrl) {
  const explicit = process.env.BACKEND_DATABASE_URL || process.env.VIDEO_DATABASE_URL
  if (explicit) {
    if (!hasDockerCompose() && explicit.includes("postgres:postgres@localhost")) {
      const localUser = process.env.USER || process.env.LOGNAME || "postgres"
      try {
        const u = new URL(explicit)
        return `postgresql://${localUser}@${u.hostname}${u.port ? `:${u.port}` : ""}${u.pathname}${u.search}${u.hash}`
      } catch {
        return `postgresql://${localUser}@localhost:5432/deepfake_video`
      }
    }
    return explicit
  }

  try {
    const u = new URL(prismaDatabaseUrl)
    const dbName = u.pathname.replace(/^\//, "")
    if (dbName === "deepfake_video") return prismaDatabaseUrl
    u.pathname = "/deepfake_video"
    return u.toString()
  } catch {
    return prismaDatabaseUrl
  }
}

async function ensureBackendRunning({ databaseUrl }) {
  if (await isPortOpen("127.0.0.1", backendPort, 1000)) {
    const existingPid = readPid("backend")
    if (isPidAlive(existingPid)) {
      console.log(`[dev-all] Backend already running (pid ${existingPid}). Restarting...`)
      try {
        process.kill(existingPid, "SIGTERM")
      } catch {}
      const start = Date.now()
      while (Date.now() - start < 3000) {
        if (!(await isPortOpen("127.0.0.1", backendPort, 500))) break
        await wait(200)
      }
    }

    if (await isPortOpen("127.0.0.1", backendPort, 1000)) {
      throw new Error(
        `[dev-all] Backend port :${backendPort} is already in use. Stop the existing backend and rerun.`
      )
    }
  }

  console.log(`[dev-all] Starting backend on :${backendPort} (Postgres)...`)

  const python = path.join(venvBin, "python")
  assertPythonDepsAvailable(python)
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
  if (isPidAlive(existingPid)) {
    console.log(`[dev-all] Celery worker already running (pid ${existingPid}). Restarting...`)
    try {
      process.kill(existingPid, "SIGTERM")
    } catch {}
    await wait(500)
  }

  console.log("[dev-all] Starting Celery worker...")

  const celery = path.join(venvBin, "celery")
  if (!fs.existsSync(celery)) {
    throw new Error(
      "[dev-all] Celery is not installed in ./.venv.\n" +
        "Install backend deps with:\n" +
        "  source .venv/bin/activate\n" +
        "  pip install -r backend/requirements.txt"
    )
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

  const prismaDatabaseUrl = getDatabaseUrlFromEnv()
  const backendDatabaseUrl = getBackendDatabaseUrlFromEnv(prismaDatabaseUrl)
  ensurePostgresDbExists(prismaDatabaseUrl)
  ensurePostgresDbExists(backendDatabaseUrl)

  // Start backend + celery on isolated Redis DB to avoid interference with any older dev processes.
  await ensureBackendRunning({ databaseUrl: backendDatabaseUrl })
  await ensureCeleryRunning({ databaseUrl: backendDatabaseUrl })

  // Ensure Prisma is pointed at Postgres and schema is applied.
  console.log("[dev-all] Running Prisma generate + migrations...")
  const prismaEnv = { ...process.env, DATABASE_URL: prismaDatabaseUrl }
  const prismaGen = spawnSync("pnpm", ["prisma:generate"], { cwd: frontendDir, env: prismaEnv, stdio: "inherit" })
  if (prismaGen.status !== 0) throw new Error("[dev-all] prisma:generate failed.")
  const prismaMig = spawnSync("pnpm", ["prisma:migrate"], { cwd: frontendDir, env: prismaEnv, stdio: "inherit" })
  if (prismaMig.status !== 0) throw new Error("[dev-all] prisma:migrate failed.")

  // Launch Next.js (foreground).
  console.log(`[dev-all] Starting Next.js on :${frontendPort} (FLASK_VIDEO_API_URL=http://localhost:${backendPort})...`)

  const nextBin = path.join(frontendDir, "node_modules", ".bin", "next")
  if (!fs.existsSync(nextBin)) {
    console.warn("[dev-all] Next.js binary not found (node_modules missing?). Running `pnpm dev` might fail if deps aren't installed.")
  }

  await chooseFrontendPort()
  await ensureFrontendPortAvailable()

  const env = {
    ...process.env,
    FLASK_VIDEO_API_URL: `http://localhost:${backendPort}`,
    // macOS can hit file-descriptor limits with native watchers (EMFILE).
    // Polling is slower but far more reliable for large repos.
    WATCHPACK_POLLING: "true",
    WATCHPACK_POLLING_INTERVAL: "1000",
  }

  const next = fs.existsSync(nextBin) ? nextBin : "next"
  const child = spawn(next, ["dev", "-p", String(frontendPort)], { cwd: frontendDir, env, stdio: "inherit" })
  writePid("next", child.pid)
  child.on("exit", (code) => process.exit(code ?? 0))
}

main().catch((e) => {
  console.error("[dev-all] Startup failed:", e)
  process.exit(1)
})

