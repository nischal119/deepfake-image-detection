import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function DELETE(
  _: Request,
  { params }: { params: { id: string } }
) {
  const existing = await prisma.detectionResult.findUnique({
    where: { id: params.id },
  });
  if (!existing)
    return NextResponse.json({ message: "Not found" }, { status: 404 });

  await prisma.detectionResult.delete({ where: { id: params.id } });
  return new NextResponse(null, { status: 204 });
}
