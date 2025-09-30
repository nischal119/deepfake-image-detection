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
            const args: string[] = [scriptPath, "--input", filePath];
            const ckpt = process.env.MODEL_CHECKPOINT_DIR;
            if (ckpt && !cmd.includes("--checkpoint")) {
              args.push("--checkpoint", ckpt);
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
              const tryParse = (s: string) => {
                try {
                  const parsed = JSON.parse(s);
                  if (typeof parsed.score === "number") score = parsed.score;
                  if (typeof parsed.heatmap === "string")
                    heatmap = parsed.heatmap;
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
                (score > 0.7
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
                    heatmap: heatmap || "",
                    artifacts: [],
                    metadata: {},
                    timeline: [],
                    reportUrl: "",
                  },
                });
              }
              await prisma.detectionJob.update({
                where: { id: job.id },
                data: {
                  status: "complete",
                  step: "Complete",
                  progress: 1,
                },
              });
            });
          }
        }

        const updated = await prisma.detectionJob.update({
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
