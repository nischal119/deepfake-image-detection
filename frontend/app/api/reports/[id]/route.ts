// Reports feature removed per requirements
export async function GET() {
  return new Response(JSON.stringify({ error: "Reports feature removed" }), {
    status: 404,
    headers: { "Content-Type": "application/json" },
  });
}
