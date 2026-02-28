"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Shield, Scan } from "lucide-react";

const navItems = [
  { label: "Home", href: "/" },
  { label: "Detect", href: "/detect" },
  { label: "History", href: "/history" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2.5 font-semibold group">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-lg transition-transform group-hover:scale-110"
              style={{ background: "linear-gradient(135deg, oklch(0.62 0.19 280), oklch(0.72 0.12 195))" }}
            >
              <Shield className="h-4.5 w-4.5 text-white" />
            </div>
            <span className="text-lg hidden sm:inline">DeepFake Detector</span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "px-4 py-2 text-sm font-medium rounded-lg transition-all",
                  pathname === item.href
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <Link href="/detect">
          <Button size="sm" className="gap-2 h-9">
            <Scan className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New Detection</span>
            <span className="sm:hidden">Detect</span>
          </Button>
        </Link>
      </div>
    </header>
  );
}
