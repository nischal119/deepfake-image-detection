"use client"

import { Card } from "@/components/ui/card"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

interface TimelineChartProps {
  data: Array<{ t: number; score: number }>;
  xAxisLabel?: string;
}

export function TimelineChart({ data, xAxisLabel = "Time (s)" }: TimelineChartProps) {
  return (
    <Card className="border-border/50 bg-muted/20 p-4">
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="oklch(0.65 0.2 15)" stopOpacity={0.8} />
              <stop offset="100%" stopColor="oklch(0.65 0.2 15)" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.269 0 0)" />
          <XAxis
            dataKey="t"
            label={{ value: xAxisLabel, position: "insideBottom", offset: -5 }}
            stroke="oklch(0.708 0 0)"
            tick={{ fill: "oklch(0.708 0 0)" }}
          />
          <YAxis
            domain={[0, 1]}
            label={{ value: "Confidence", angle: -90, position: "insideLeft" }}
            stroke="oklch(0.708 0 0)"
            tick={{ fill: "oklch(0.708 0 0)" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "oklch(0.145 0 0)",
              border: "1px solid oklch(0.269 0 0)",
              borderRadius: "8px",
            }}
            labelStyle={{ color: "oklch(0.985 0 0)" }}
          />
          <Area type="monotone" dataKey="score" stroke="oklch(0.65 0.2 15)" fill="url(#scoreGradient)" />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  )
}
