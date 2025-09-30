import { prisma } from "@/lib/db";

export async function GET(
  _: Request,
  { params }: { params: { jobId: string } }
) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let cancelled = false;

      const send = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(`event: ${event}\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };

      const poll = async () => {
        if (cancelled) return;
        const job = await prisma.detectionJob.findUnique({
          where: { id: params.jobId },
        });
        if (!job) {
          send("error", { message: "not_found" });
          controller.close();
          return;
        }
        send("progress", { progress: job.progress, step: job.step });
        if (job.status === "complete" || job.status === "error") {
          send(job.status === "complete" ? "complete" : "error", {
            jobId: job.id,
          });
          controller.close();
          return;
        }
        setTimeout(poll, 1200);
      };

      poll();

      return () => {
        cancelled = true;
      };
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
