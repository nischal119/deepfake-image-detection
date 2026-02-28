import { prisma } from "@/lib/db";
import { spawn } from "node:child_process";
import sizeOf from "image-size";
import fs from "node:fs";
import path from "node:path";

function pathFromUrl(publicUrl: string): string {
  // converts "/uploads/abc.jpg" -> "<cwd>/public/uploads/abc.jpg"
  const rel = publicUrl.replace(/^\//, "");
  return path.join(process.cwd(), "public", rel);
}

function ensurePublicAsset(
  absOrData: string | undefined,
  jobId: string
): string {
  if (!absOrData) return "";
  if (absOrData.startsWith("data:")) return absOrData; // already data URL
  if (/^https?:\/\//i.test(absOrData)) return absOrData; // remote URL
  // Might be a filesystem path or relative path
  const abs = path.isAbsolute(absOrData)
    ? absOrData
    : path.join(process.cwd(), absOrData);
  try {
    if (fs.existsSync(abs)) {
      const ext = path.extname(abs) || ".png";
      const outDir = path.join(process.cwd(), "public", "heatmaps");
      fs.mkdirSync(outDir, { recursive: true });
      const outPath = path.join(outDir, `${jobId}${ext}`);
      if (abs !== outPath) fs.copyFileSync(abs, outPath);
      const publicUrl = `/heatmaps/${jobId}${ext}`;
      return publicUrl;
    }
  } catch {}
  // Maybe it's base64 without a prefix
  if (/^[A-Za-z0-9+/=]+$/.test(absOrData.slice(0, 50))) {
    return `data:image/png;base64,${absOrData}`;
  }
  return "";
}

let started = false;

export function ensureWorkerStarted() {
  if (started) return;
  started = true;

  // Poll every second to advance queued/analyzing jobs
  setInterval(async () => {
    try {
      const jobs = await prisma.detectionJob.findMany({
        where: { status: { in: ["queued", "analyzing", "postprocessing"] } },
        orderBy: { createdAt: "asc" },
        take: 5,
      });

      for (const job of jobs) {
        let nextProgress = Math.min(1, Number(job.progress) + 0.12);
        let nextStatus = job.status;
        let nextStep = job.step;

        if (nextProgress < 0.4) {
          nextStatus = "analyzing";
          nextStep = "Analyzing";
        } else if (nextProgress < 0.98) {
          nextStatus = "postprocessing";
          nextStep = "Post-processing";
        } else {
          // Near the end: wait for analyzer to persist result; do not error prematurely
          const hasResult = await prisma.detectionResult
            .findUnique({ where: { jobId: job.id } })
            .catch(() => null);
          if (hasResult) {
            nextStatus = "complete";
            nextStep = "Complete";
          } else {
            nextStatus = "postprocessing";
            nextStep = "Post-processing";
          }
        }

        // When reaching analyzing step for first time, run external analyzer once
        if (nextStatus === "analyzing" && job.step !== "Analyzing") {
          const cmd = process.env.PYTHON_PREDICT_CMD; // e.g. "../.venv/bin/python ../inference/infer_vit.py --input {file}"
          if (cmd) {
            // Resolve python binary and script path safely (no shell quoting issues)
            const pythonBin =
              process.env.PYTHON_BIN ||
              path.join(process.cwd(), "..", ".venv", "bin", "python");
            // Try to locate infer_vit.py path from the command
            const scriptMatch = cmd.match(/[\w\/.\-]+infer_vit\.py/);
            const scriptPath = scriptMatch
              ? scriptMatch[0]
              : path.join(process.cwd(), "..", "inference", "infer_vit.py");
            const filePath = pathFromUrl(job.filePath);
            const outFile = path.join(process.cwd(), "tmp", `${job.id}.json`);
            await fs.promises.mkdir(path.dirname(outFile), { recursive: true });
            const args: string[] = [
              scriptPath,
              "--input",
              filePath,
              "--explain",
            ]; // request heatmap/artifacts
            // Prefer env checkpoint; otherwise fallback to local fine-tuned checkpoint if present
            let ckpt = process.env.MODEL_CHECKPOINT_DIR;
            if (!ckpt) {
              const fallbackCkpt = path.join(
                process.cwd(),
                "..",
                "deepfake_vs_real_image_detection",
                "checkpoint-14282"
              );
              if (fs.existsSync(fallbackCkpt)) ckpt = fallbackCkpt;
            }
            if (ckpt && !cmd.includes("--checkpoint")) {
              args.push("--checkpoint", ckpt);
            }

            const temp = process.env.MODEL_TEMP || "1.0";
            if (!cmd.includes("--temp")) {
              args.push("--temp", String(temp));
            }

            args.push("--out", outFile, "--quiet");
            const bin = fs.existsSync(pythonBin) ? pythonBin : "python";
            const child = spawn(bin, args, { shell: false });
            let out = "";
            let err = "";
            child.on("error", async (err) => {
              await prisma.detectionJob.update({
                where: { id: job.id },
                data: { status: "error", step: String(err) },
              });
            });
            child.stdout.on("data", async (d) => {
              out += String(d);
            });
            child.stderr.on("data", async (d) => {
              err += String(d);
            });
            child.on("close", async (code) => {
              if (code !== 0) {
                await prisma.detectionJob.update({
                  where: { id: job.id },
                  data: {
                    status: "error",
                    step: `Analyzer failed: ${err.slice(0, 180)}`,
                  },
                });
                return;
              }
              // Prefer reading JSON file, fallback to stdout
              let score: number | null = null;
              let verdict:
                | "likely_fake"
                | "likely_real"
                | "inconclusive"
                | undefined;
              let heatmap: string | undefined;
              let artifacts: any[] | undefined;
              let metadata: Record<string, any> | undefined;
              let timeline: Array<{ t: number; score: number }> | undefined;
              let reportUrl: string | undefined;
              const tryParse = (s: string) => {
                try {
                  const parsed = JSON.parse(s);
                  if (typeof parsed.score === "number") score = parsed.score;
                  if (typeof parsed.heatmap === "string")
                    heatmap = parsed.heatmap;
                  if (parsed.artifacts && Array.isArray(parsed.artifacts))
                    artifacts = parsed.artifacts;
                  if (parsed.metadata && typeof parsed.metadata === "object")
                    metadata = parsed.metadata;
                  if (parsed.timeline && Array.isArray(parsed.timeline))
                    timeline = parsed.timeline;
                  if (typeof parsed.reportUrl === "string")
                    reportUrl = parsed.reportUrl;
                  if (
                    parsed.verdict === "likely_fake" ||
                    parsed.verdict === "likely_real" ||
                    parsed.verdict === "inconclusive"
                  )
                    verdict = parsed.verdict;
                } catch {}
              };
              // 1) file
              try {
                if (fs.existsSync(outFile)) {
                  const fileData = await fs.promises.readFile(outFile, "utf8");
                  tryParse(fileData);
                }
              } catch {}
              // 2) stdout (extract last JSON block if any)
              if (score === null) {
                const match = out.match(/\{[\s\S]*\}$/);
                if (match) tryParse(match[0]);
              }
              if (score === null) {
                await prisma.detectionJob.update({
                  where: { id: job.id },
                  data: { status: "error", step: "Analyzer returned no JSON" },
                });
                return;
              }

              // normalize heatmap to public URL or data URL
              const heatmapPublic = ensurePublicAsset(heatmap, job.id);

              let width = 0,
                height = 0;
              try {
                if (
                  job.type === "image" &&
                  fs.existsSync(pathFromUrl(job.filePath))
                ) {
                  const dim = sizeOf(pathFromUrl(job.filePath));
                  width = dim.width || 0;
                  height = dim.height || 0;
                } else {
                  width = 1920;
                  height = 1080;
                }
              } catch {}

              const finalVerdict =
                verdict ??
                (score > 0.6
                  ? "likely_fake"
                  : score < 0.3
                  ? "likely_real"
                  : "inconclusive");
              const exists = await prisma.detectionResult
                .findUnique({ where: { jobId: job.id } })
                .catch(() => null);
              if (!exists) {
                await prisma.detectionResult.create({
                  data: {
                    jobId: job.id,
                    score,
                    verdict: finalVerdict as any,
                    inputType: job.type,
                    inputWidth: width,
                    inputHeight: height,
                    durationSec: job.type === "video" ? 0 : 0,
                    heatmap: heatmapPublic || "",
                    artifacts: artifacts || [],
                    metadata: metadata || {},
                    timeline: timeline || [],
                    reportUrl: reportUrl || `/api/reports/${job.id}`,
                  },
                });
              }
              await prisma.detectionJob.update({
                where: { id: job.id },
                data: { status: "complete", step: "Complete", progress: 1 },
              });
            });
          }
        }

        await prisma.detectionJob.update({
          where: { id: job.id },
          data: {
            progress: nextProgress,
            status: nextStatus,
            step: nextStep,
            events: {
              create: [
                {
                  type: nextStatus === "complete" ? "complete" : "progress",
                  message: nextStep,
                  progress: nextProgress,
                },
              ],
            },
          },
        });

        // removed placeholder completion creation
      }
    } catch (e) {
      // avoid noisy logs in prod
    }
  }, 1000);
}
