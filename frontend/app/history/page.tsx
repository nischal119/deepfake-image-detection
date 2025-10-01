"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Eye, Trash2, Search, ImageIcon, Video } from "lucide-react";
import type { HistoryItem, Verdict } from "@/lib/types";
import { api } from "@/lib/api";
import { formatDate, formatScore } from "@/lib/format";
import Link from "next/link";

async function fetchHistory(page = 1, pageSize = 20) {
  const res = await fetch(`/api/history?page=${page}&pageSize=${pageSize}`);
  if (!res.ok) throw new Error("Failed to load history");
  return (await res.json()) as {
    items: HistoryItem[];
    total: number;
    page: number;
    pageSize: number;
  };
}

const verdictColors: Record<Verdict, { bg: string; text: string }> = {
  likely_real: {
    bg: "oklch(0.7 0.15 165 / 0.15)",
    text: "oklch(0.7 0.15 165)",
  },
  inconclusive: { bg: "oklch(0.8 0.15 85 / 0.15)", text: "oklch(0.8 0.15 85)" },
  likely_fake: { bg: "oklch(0.65 0.2 15 / 0.15)", text: "oklch(0.65 0.2 15)" },
};

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [verdictFilter, setVerdictFilter] = useState<string>("all");
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    fetchHistory()
      .then((d) => setHistory(d.items))
      .catch(() => setHistory([]));
  }, []);

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.fileName
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === "all" || item.type === typeFilter;
    const matchesVerdict =
      verdictFilter === "all" || item.verdict === verdictFilter;
    return matchesSearch && matchesType && matchesVerdict;
  });

  const handleDelete = async (id: string) => {
    await fetch(`/api/history/${id}`, { method: "DELETE" });
    setHistory((h) => h.filter((item) => item.id !== id));
  };

  return (
    <div className="min-h-screen">
      <main className="container py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">
            Detection History
          </h1>
          <p className="text-muted-foreground">
            View and manage your previous deepfake detections
          </p>
        </div>

        {/* Filters */}
        <Card className="mb-6 p-6">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search by filename..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>

            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="image">Images</SelectItem>
                <SelectItem value="video">Videos</SelectItem>
              </SelectContent>
            </Select>

            <Select value={verdictFilter} onValueChange={setVerdictFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by verdict" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Verdicts</SelectItem>
                <SelectItem value="likely_real">Likely Real</SelectItem>
                <SelectItem value="inconclusive">Inconclusive</SelectItem>
                <SelectItem value="likely_fake">Likely Fake</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </Card>

        {/* Results Table */}
        {filteredHistory.length > 0 ? (
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-20">Preview</TableHead>
                  <TableHead>File Name</TableHead>
                  <TableHead className="w-24">Type</TableHead>
                  <TableHead className="w-32">Confidence</TableHead>
                  <TableHead className="w-36">Verdict</TableHead>
                  <TableHead className="w-48">Date</TableHead>
                  <TableHead className="w-32 text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredHistory.map((item) => (
                  <TableRow key={item.id}>
                    <TableCell>
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-border/50 bg-muted/30">
                        {item.type === "video" ? (
                          <Video className="h-5 w-5 text-muted-foreground" />
                        ) : (
                          <ImageIcon className="h-5 w-5 text-muted-foreground" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="font-medium">
                      {item.fileName}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="capitalize">
                        {item.type}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <span className="font-mono font-semibold">
                        {formatScore(item.score)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge
                        style={{
                          backgroundColor: verdictColors[item.verdict].bg,
                          color: verdictColors[item.verdict].text,
                          borderColor: verdictColors[item.verdict].text,
                        }}
                      >
                        {item.verdict.replace("_", " ")}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(item.createdAt)}
                    </TableCell>
                    <TableCell>
                      <div className="flex justify-end gap-2">
                        <Button
                          size="icon"
                          variant="ghost"
                          title="View details"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        {/* Report download removed per requirements */}
                        <Button
                          size="icon"
                          variant="ghost"
                          title="Delete"
                          onClick={() => handleDelete(item.id)}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        ) : (
          <Card className="flex min-h-[400px] flex-col items-center justify-center p-12 text-center">
            <div
              className="mb-4 flex h-16 w-16 items-center justify-center rounded-full"
              style={{ backgroundColor: "oklch(0.62 0.19 280 / 0.15)" }}
            >
              <Search
                className="h-8 w-8"
                style={{ color: "oklch(0.62 0.19 280)" }}
              />
            </div>
            <h3 className="mb-2 text-lg font-semibold">No detections found</h3>
            <p className="mb-6 text-muted-foreground">
              {searchQuery || typeFilter !== "all" || verdictFilter !== "all"
                ? "Try adjusting your filters"
                : "Start a detection to see results here"}
            </p>
            <Button asChild>
              <Link href="/detect">Start a Detection</Link>
            </Button>
          </Card>
        )}

        {/* Pagination */}
        {filteredHistory.length > 0 && (
          <div className="mt-6 flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Showing {filteredHistory.length} of {history.length} results
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled>
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled>
                Next
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
