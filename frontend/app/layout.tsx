import type React from "react";
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Analytics } from "@vercel/analytics/next";
import { Toaster } from "@/components/ui/toaster";
import "./globals.css";
import { Suspense } from "react";
import { Navbar } from "@/components/layout/navbar";

export const metadata: Metadata = {
  title: "DeepFake Detector – Verify Image and Video Authenticity",
  description:
    "Upload media to detect deepfakes with a fine-tuned model. Clear scores, evidence overlays, and downloadable reports.",

  openGraph: {
    type: "website",
    title: "DeepFake Detector – Verify Image and Video Authenticity",
    description:
      "Upload media to detect deepfakes with a fine-tuned model. Clear scores, evidence overlays, and downloadable reports.",
  },
  twitter: {
    card: "summary_large_image",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <Navbar />
        <Suspense fallback={null}>
          {children}
          <Toaster />
          <Analytics />
        </Suspense>
      </body>
    </html>
  );
}
