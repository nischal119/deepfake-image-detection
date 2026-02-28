"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, Upload, Zap, Eye, ArrowRight } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background orbs */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <div
          className="absolute top-[-20%] left-[-10%] h-[600px] w-[600px] rounded-full opacity-20 blur-[120px]"
          style={{
            background: "oklch(0.62 0.19 280)",
            animation: "floatOrb 18s ease-in-out infinite",
          }}
        />
        <div
          className="absolute bottom-[-15%] right-[-10%] h-[500px] w-[500px] rounded-full opacity-15 blur-[100px]"
          style={{
            background: "oklch(0.72 0.12 195)",
            animation: "floatOrb 22s ease-in-out infinite reverse",
          }}
        />
        <div
          className="absolute top-[40%] right-[20%] h-[300px] w-[300px] rounded-full opacity-10 blur-[80px]"
          style={{
            background: "oklch(0.75 0.15 350)",
            animation: "floatOrb 15s ease-in-out infinite 3s",
          }}
        />
      </div>

      {/* Subtle grid pattern */}
      <div
        className="pointer-events-none fixed inset-0 z-0 opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      {/* Hero Section */}
      <section className="relative z-10">
        <div className="container py-24 md:py-36">
          <div className="mx-auto max-w-4xl text-center">
            <Badge
              variant="outline"
              className="mb-6 border-primary/50 text-primary px-4 py-1.5 text-sm animate-in fade-in slide-in-from-bottom-2 duration-700"
            >
              ✦ AI-Powered Verification
            </Badge>

            <h1 className="mb-6 text-4xl font-bold tracking-tight text-balance md:text-6xl lg:text-7xl animate-in fade-in slide-in-from-bottom-3 duration-700 delay-100">
              Detect Deepfakes with{" "}
              <span
                className="bg-clip-text text-transparent"
                style={{
                  backgroundImage:
                    "linear-gradient(135deg, oklch(0.62 0.19 280), oklch(0.72 0.12 195), oklch(0.75 0.15 350))",
                }}
              >
                Confidence
              </span>
            </h1>

            <p className="mb-10 text-lg text-muted-foreground text-pretty md:text-xl animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200 max-w-2xl mx-auto">
              Upload an image or video and get instant authenticity analysis
              powered by our fine-tuned deep learning model with gradient saliency heatmaps.
            </p>

            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row animate-in fade-in slide-in-from-bottom-5 duration-700 delay-300">
              <Button asChild size="lg" className="min-w-48 h-12 text-base group">
                <Link href="/detect">
                  Start Detection
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
            </div>

            {/* Hero visual */}
            <div className="mt-20 animate-in fade-in slide-in-from-bottom-6 duration-1000 delay-500">
              <div className="relative rounded-2xl border border-border/30 bg-card/30 p-3 backdrop-blur-sm shadow-2xl shadow-primary/5">
                <div
                  className="absolute -inset-px rounded-2xl opacity-50"
                  style={{
                    background:
                      "linear-gradient(135deg, oklch(0.62 0.19 280 / 0.2), transparent 40%, oklch(0.72 0.12 195 / 0.2))",
                  }}
                />
                <img
                  src="/split-face-showing-deepfake-detection-with-digital.jpg"
                  alt="DeepFake detection visualization"
                  className="relative w-full rounded-xl"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative z-10 border-t border-border/20 py-24">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-16">
            <Badge variant="outline" className="mb-4 border-secondary/50 text-secondary">
              How It Works
            </Badge>
            <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">
              Three Simple Steps
            </h2>
            <p className="text-lg text-muted-foreground">
              Verify the authenticity of your media in seconds
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3 max-w-5xl mx-auto">
            {[
              {
                icon: Upload,
                title: "1. Upload",
                desc: "Drag & drop any image or video. We auto-detect the format and route to the right pipeline.",
                color: "oklch(0.62 0.19 280)",
                bg: "oklch(0.62 0.19 280 / 0.1)",
              },
              {
                icon: Zap,
                title: "2. Analyze",
                desc: "Our AI model runs frame-by-frame analysis with gradient saliency to highlight manipulated regions.",
                color: "oklch(0.72 0.12 195)",
                bg: "oklch(0.72 0.12 195 / 0.1)",
              },
              {
                icon: Eye,
                title: "3. Results",
                desc: "Get a clear verdict with confidence scores, per-frame heatmaps, and downloadable reports.",
                color: "oklch(0.75 0.15 350)",
                bg: "oklch(0.75 0.15 350 / 0.1)",
              },
            ].map((step) => (
              <Card
                key={step.title}
                className="group relative overflow-hidden border-border/30 bg-card/50 p-8 backdrop-blur-sm transition-all duration-300 hover:border-border/60 hover:shadow-lg hover:shadow-primary/5 hover:-translate-y-1"
              >
                <div
                  className="absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100"
                  style={{
                    background: `radial-gradient(circle at 50% 0%, ${step.bg}, transparent 70%)`,
                  }}
                />
                <div className="relative">
                  <div
                    className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl transition-transform group-hover:scale-110"
                    style={{ backgroundColor: step.bg }}
                  >
                    <step.icon className="h-7 w-7" style={{ color: step.color }} />
                  </div>
                  <h3 className="mb-3 text-xl font-semibold">{step.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{step.desc}</p>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 border-t border-border/20 py-16">
        <div className="container">
          <div className="mx-auto max-w-4xl grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: "ResNet50", label: "Model Architecture" },
              { value: "16", label: "Frames Analyzed" },
              { value: "GradCAM", label: "Saliency Maps" },
              { value: "<30s", label: "Processing Time" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div
                  className="text-2xl md:text-3xl font-bold mb-1 bg-clip-text text-transparent"
                  style={{
                    backgroundImage:
                      "linear-gradient(135deg, oklch(0.62 0.19 280), oklch(0.72 0.12 195))",
                  }}
                >
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 border-t border-border/20 py-24">
        <div className="container">
          <Card className="relative overflow-hidden border-border/30 bg-card/50 p-12 text-center backdrop-blur-sm max-w-3xl mx-auto">
            <div
              className="absolute inset-0 opacity-20"
              style={{
                background:
                  "radial-gradient(circle at 30% 50%, oklch(0.62 0.19 280 / 0.3), transparent), radial-gradient(circle at 70% 50%, oklch(0.72 0.12 195 / 0.3), transparent)",
              }}
            />
            <div className="relative">
              <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">
                Ready to Verify Your Media?
              </h2>
              <p className="mb-8 text-lg text-muted-foreground max-w-lg mx-auto">
                Start detecting deepfakes in seconds with AI-powered analysis and visual explanations
              </p>
              <Button asChild size="lg" className="min-w-48 h-12 text-base group">
                <Link href="/detect">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
            </div>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/20 py-12">
        <div className="container">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2 font-semibold">
              <Shield className="h-5 w-5" style={{ color: "oklch(0.62 0.19 280)" }} />
              <span>DeepFake Detector</span>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2025 DeepFake Detector. All rights reserved.
            </p>
          </div>
        </div>
      </footer>

      {/* CSS Animations */}
      <style jsx global>{`
        @keyframes floatOrb {
          0%, 100% { transform: translate(0, 0) scale(1); }
          25% { transform: translate(30px, -40px) scale(1.05); }
          50% { transform: translate(-20px, 20px) scale(0.95); }
          75% { transform: translate(40px, 10px) scale(1.02); }
        }
      `}</style>
    </div>
  );
}
