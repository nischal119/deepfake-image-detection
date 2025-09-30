"use client";

import type React from "react";

import { useCallback, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { UploadCloud, X, ImageIcon, Video, FileWarning } from "lucide-react";
import { formatFileSize } from "@/lib/format";

interface UploadCardProps {
  onFileSelect: (file: File) => void;
  onStartAnalysis: (file: File) => void;
  disabled?: boolean;
  currentFile: File | null;
}

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const ALLOWED_TYPES = ["image/jpeg", "image/png", "video/mp4"];

export function UploadCard({
  onFileSelect,
  onStartAnalysis,
  disabled,
  currentFile,
}: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const { toast } = useToast();

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "That file type isn't supported. Please use JPG, PNG, or MP4.";
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File is too large. Maximum size is ${formatFileSize(
        MAX_FILE_SIZE
      )}.`;
    }
    return null;
  };

  const handleFile = useCallback(
    (file: File) => {
      const error = validateFile(file);
      if (error) {
        toast({
          title: "Invalid file",
          description: error,
          variant: "destructive",
        });
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect, toast]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData.items;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            handleFile(file);
          }
          break;
        }
      }
    },
    [handleFile]
  );

  const previewUrl = useMemo(
    () => (currentFile ? URL.createObjectURL(currentFile) : null),
    [currentFile]
  );

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h2 className="mb-1 text-xl font-semibold">Upload Media</h2>
          <p className="text-sm text-muted-foreground">
            Select an image or video to analyze for deepfakes
          </p>
        </div>

        {!currentFile ? (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onPaste={handlePaste}
            className={`relative flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border/50 bg-muted/20 hover:border-primary/50 hover:bg-muted/30"
            } ${disabled ? "pointer-events-none opacity-50" : ""}`}
            role="button"
            tabIndex={0}
            aria-label="Upload file"
          >
            <input
              type="file"
              accept={ALLOWED_TYPES.join(",")}
              onChange={handleFileInput}
              className="absolute inset-0 cursor-pointer opacity-0"
              disabled={disabled}
              aria-label="File input"
            />

            <UploadCloud className="mb-4 h-12 w-12 text-muted-foreground" />

            <h3 className="mb-2 text-lg font-medium">
              Drop your image or video
            </h3>
            <p className="mb-4 text-sm text-muted-foreground">
              JPG, PNG, or MP4 up to 100MB
            </p>

            <Button type="button" variant="secondary" disabled={disabled}>
              Select File
            </Button>

            <div className="mt-6 space-y-1 text-center">
              <p className="text-xs text-muted-foreground">
                You can also paste from clipboard
              </p>
              <p className="text-xs text-muted-foreground">
                Try webcam for quick snapshots
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Card className="flex items-center gap-4 border-border/50 bg-muted/30 p-4">
              <div
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg"
                style={{ backgroundColor: "oklch(0.62 0.19 280 / 0.15)" }}
              >
                {currentFile.type.startsWith("video") ? (
                  <Video
                    className="h-6 w-6"
                    style={{ color: "oklch(0.62 0.19 280)" }}
                  />
                ) : (
                  <ImageIcon
                    className="h-6 w-6"
                    style={{ color: "oklch(0.62 0.19 280)" }}
                  />
                )}
              </div>

              <div className="flex-1 overflow-hidden">
                <p className="truncate font-medium">{currentFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {formatFileSize(currentFile.size)}
                </p>
              </div>

              <Button
                size="icon"
                variant="ghost"
                onClick={() => onFileSelect(null as any)}
                disabled={disabled}
                aria-label="Remove file"
              >
                <X className="h-4 w-4" />
              </Button>
            </Card>

            {currentFile.type.startsWith("image/") && previewUrl && (
              <div className="max-w-md overflow-hidden rounded-lg border border-border/50">
                <img
                  src={previewUrl}
                  alt="Selected image preview"
                  className="h-auto w-full"
                />
              </div>
            )}
            {currentFile.type.startsWith("video/") && previewUrl && (
              <div className="max-w-md overflow-hidden rounded-lg border border-border/50">
                <video src={previewUrl} controls className="h-auto w-full" />
              </div>
            )}

            <Button
              onClick={() => onStartAnalysis(currentFile)}
              disabled={disabled}
              className="w-full"
              size="lg"
            >
              Start Analysis
            </Button>
          </div>
        )}

        <div className="rounded-lg border border-border/50 bg-muted/20 p-4">
          <div className="flex gap-3">
            <FileWarning className="h-5 w-5 shrink-0 text-muted-foreground" />
            <div className="space-y-1 text-sm">
              <p className="font-medium">Supported formats</p>
              <p className="text-muted-foreground">Images: JPEG, PNG</p>
              <p className="text-muted-foreground">Videos: MP4 (max 100MB)</p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
