import { NextResponse } from "next/server";

export async function GET(_: Request, { params }: { params: { id: string } }) {
  // For demo: return a simple PDF placeholder
  const content = Buffer.from("25 50 44 46", "hex");
  return new NextResponse(content, {
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename=report-${params.id}.pdf`,
    },
  });
}
