"use client"
import { Badge } from "@/components/ui/badge"
import type { Verdict } from "@/lib/types"
import { formatScore } from "@/lib/format"

interface ScoreGaugeProps {
  score: number
  verdict: Verdict
}

const verdictConfig: Record<Verdict, { label: string; color: string; bgColor: string; description: string }> = {
  likely_real: {
    label: "Likely Real",
    color: "oklch(0.7 0.15 165)",
    bgColor: "oklch(0.7 0.15 165 / 0.15)",
    description: "Low probability of manipulation detected",
  },
  inconclusive: {
    label: "Inconclusive",
    color: "oklch(0.8 0.15 85)",
    bgColor: "oklch(0.8 0.15 85 / 0.15)",
    description: "Some signs of manipulation, further review recommended",
  },
  likely_fake: {
    label: "Likely Fake",
    color: "oklch(0.65 0.2 15)",
    bgColor: "oklch(0.65 0.2 15 / 0.15)",
    description: "High probability of manipulation detected",
  },
}

export function ScoreGauge({ score, verdict }: ScoreGaugeProps) {
  const config = verdictConfig[verdict]
  const percentage = score * 100
  const rotation = score * 180 - 90 // -90 to 90 degrees for semi-circle

  return (
    <div className="space-y-6">
      <div className="flex flex-col items-center">
        <div className="relative h-48 w-full max-w-sm">
          {/* Semi-circle gauge background */}
          <svg viewBox="0 0 200 100" className="w-full">
            {/* Background arc */}
            <path
              d="M 20 90 A 80 80 0 0 1 180 90"
              fill="none"
              stroke="currentColor"
              strokeWidth="12"
              className="text-muted/30"
            />

            {/* Colored segments */}
            <path
              d="M 20 90 A 80 80 0 0 1 66.4 26.4"
              fill="none"
              stroke="oklch(0.7 0.15 165)"
              strokeWidth="12"
              opacity="0.6"
            />
            <path
              d="M 66.4 26.4 A 80 80 0 0 1 133.6 26.4"
              fill="none"
              stroke="oklch(0.8 0.15 85)"
              strokeWidth="12"
              opacity="0.6"
            />
            <path
              d="M 133.6 26.4 A 80 80 0 0 1 180 90"
              fill="none"
              stroke="oklch(0.65 0.2 15)"
              strokeWidth="12"
              opacity="0.6"
            />

            {/* Needle */}
            <g transform={`rotate(${rotation} 100 90)`}>
              <line x1="100" y1="90" x2="100" y2="30" stroke={config.color} strokeWidth="3" strokeLinecap="round" />
              <circle cx="100" cy="90" r="6" fill={config.color} />
            </g>
          </svg>

          {/* Score display */}
          <div className="absolute inset-x-0 bottom-0 text-center">
            <div className="text-4xl font-bold" style={{ color: config.color }}>
              {formatScore(score)}
            </div>
            <div className="text-sm text-muted-foreground">Deepfake Confidence</div>
          </div>
        </div>

        <div className="mt-6 text-center">
          <Badge
            className="mb-2"
            style={{ backgroundColor: config.bgColor, color: config.color, borderColor: config.color }}
          >
            {config.label}
          </Badge>
          <p className="text-sm text-muted-foreground">{config.description}</p>
        </div>
      </div>

      {/* Legend */}
      <div className="grid grid-cols-3 gap-4 text-center text-sm">
        <div>
          <div className="mb-1 font-medium" style={{ color: "oklch(0.7 0.15 165)" }}>
            0.00 - 0.30
          </div>
          <div className="text-xs text-muted-foreground">Likely Real</div>
        </div>
        <div>
          <div className="mb-1 font-medium" style={{ color: "oklch(0.8 0.15 85)" }}>
            0.30 - 0.70
          </div>
          <div className="text-xs text-muted-foreground">Inconclusive</div>
        </div>
        <div>
          <div className="mb-1 font-medium" style={{ color: "oklch(0.65 0.2 15)" }}>
            0.70 - 1.00
          </div>
          <div className="text-xs text-muted-foreground">Likely Fake</div>
        </div>
      </div>
    </div>
  )
}
