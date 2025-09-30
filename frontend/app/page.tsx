import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Shield, Upload, Zap, Eye, CheckCircle2 } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Navbar */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Shield className="h-6 w-6" style={{ color: "oklch(0.62 0.19 280)" }} />
            <span className="text-lg">DeepFake Detector</span>
          </Link>

          <Button asChild size="lg">
            <Link href="/detect">Try Detection</Link>
          </Button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Subtle aurora gradient background */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            background:
              "radial-gradient(ellipse 80% 50% at 50% -20%, oklch(0.62 0.19 280 / 0.3), transparent), radial-gradient(ellipse 60% 50% at 80% 50%, oklch(0.72 0.12 195 / 0.2), transparent)",
          }}
        />

        <div className="container relative py-24 md:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <Badge variant="outline" className="mb-6 border-primary/50 text-primary">
              AI-Powered Verification
            </Badge>

            <h1 className="mb-6 text-4xl font-bold tracking-tight text-balance md:text-6xl lg:text-7xl">
              Detect Deepfakes with{" "}
              <span className="bg-gradient-to-r from-[oklch(0.62_0.19_280)] to-[oklch(0.72_0.12_195)] bg-clip-text text-transparent">
                Confidence
              </span>
            </h1>

            <p className="mb-10 text-lg text-muted-foreground text-pretty md:text-xl">
              Upload an image or video and get instant authenticity scores powered by your fine-tuned model.
            </p>

            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Button asChild size="lg" className="min-w-40">
                <Link href="/detect">Try Detection</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="min-w-40 bg-transparent">
                <Link href="#demo">View Demo</Link>
              </Button>
            </div>

            {/* Hero Image */}
            <div className="mt-16 rounded-xl border border-border/50 bg-muted/30 p-4 backdrop-blur">
              <img
                src="/split-face-showing-deepfake-detection-with-digital.jpg"
                alt="DeepFake detection visualization"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="border-t border-border/40 py-24">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-16">
            <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">How It Works</h2>
            <p className="text-lg text-muted-foreground">Three simple steps to verify the authenticity of your media</p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            <Card className="relative overflow-hidden border-border/50 bg-card/50 p-8 backdrop-blur">
              <div
                className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl"
                style={{ backgroundColor: "oklch(0.62 0.19 280 / 0.15)" }}
              >
                <Upload className="h-7 w-7" style={{ color: "oklch(0.62 0.19 280)" }} />
              </div>
              <h3 className="mb-3 text-xl font-semibold">1. Upload</h3>
              <p className="text-muted-foreground leading-relaxed">
                Drag & drop or paste from clipboard. Supports JPG, PNG, or MP4 up to 100MB.
              </p>
            </Card>

            <Card className="relative overflow-hidden border-border/50 bg-card/50 p-8 backdrop-blur">
              <div
                className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl"
                style={{ backgroundColor: "oklch(0.72 0.12 195 / 0.15)" }}
              >
                <Zap className="h-7 w-7" style={{ color: "oklch(0.72 0.12 195)" }} />
              </div>
              <h3 className="mb-3 text-xl font-semibold">2. Analyze</h3>
              <p className="text-muted-foreground leading-relaxed">
                Server runs your model, streams progress, and aggregates artifacts.
              </p>
            </Card>

            <Card className="relative overflow-hidden border-border/50 bg-card/50 p-8 backdrop-blur">
              <div
                className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl"
                style={{ backgroundColor: "oklch(0.75 0.15 350 / 0.15)" }}
              >
                <Eye className="h-7 w-7" style={{ color: "oklch(0.75 0.15 350)" }} />
              </div>
              <h3 className="mb-3 text-xl font-semibold">3. Decide</h3>
              <p className="text-muted-foreground leading-relaxed">
                Get a clear verdict with confidence and visual explanations.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Trust Bar Section */}
      <section className="border-t border-border/40 py-16">
        <div className="container">
          <p className="mb-8 text-center text-sm font-medium uppercase tracking-wider text-muted-foreground">
            Powered by industry-leading technology
          </p>
          <div className="flex flex-wrap items-center justify-center gap-12 opacity-60 grayscale">
            <div className="text-2xl font-bold">OpenAI</div>
            <div className="text-2xl font-bold">Hugging Face</div>
            <div className="text-2xl font-bold">Vercel</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="border-t border-border/40 py-24">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-16">
            <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">Why Choose Our Detector</h2>
            <p className="text-lg text-muted-foreground">
              Advanced AI technology with transparent, explainable results
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[
              {
                title: "Real-time Analysis",
                description: "Get instant feedback with live progress updates and streaming results",
              },
              {
                title: "Visual Evidence",
                description: "Heatmaps and artifact overlays show exactly where manipulations occur",
              },
              {
                title: "Confidence Scores",
                description: "Clear numerical scores help you make informed decisions",
              },
              {
                title: "Detailed Reports",
                description: "Download comprehensive PDF reports for documentation",
              },
              {
                title: "History Tracking",
                description: "Keep track of all your detections with searchable history",
              },
              {
                title: "Model Transparency",
                description: "View model performance metrics and health status",
              },
            ].map((feature, i) => (
              <Card key={i} className="border-border/50 bg-card/50 p-6 backdrop-blur">
                <CheckCircle2 className="mb-4 h-6 w-6 text-primary" />
                <h3 className="mb-2 font-semibold">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t border-border/40 py-24">
        <div className="container">
          <Card className="relative overflow-hidden border-border/50 bg-gradient-to-br from-card/80 to-card/40 p-12 text-center backdrop-blur">
            <div
              className="absolute inset-0 opacity-20"
              style={{
                background:
                  "radial-gradient(circle at 30% 50%, oklch(0.62 0.19 280 / 0.3), transparent), radial-gradient(circle at 70% 50%, oklch(0.72 0.12 195 / 0.3), transparent)",
              }}
            />
            <div className="relative">
              <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">Ready to Verify Your Media?</h2>
              <p className="mb-8 text-lg text-muted-foreground">
                Start detecting deepfakes in seconds with our AI-powered tool
              </p>
              <Button asChild size="lg" className="min-w-48">
                <Link href="/detect">Get Started</Link>
              </Button>
            </div>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 py-12">
        <div className="container">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2 font-semibold">
              <Shield className="h-5 w-5" style={{ color: "oklch(0.62 0.19 280)" }} />
              <span>DeepFake Detector</span>
            </div>
            <p className="text-sm text-muted-foreground">© 2025 DeepFake Detector. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
