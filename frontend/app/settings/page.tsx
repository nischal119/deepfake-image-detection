"use client"

import { useState } from "react"
import { Navbar } from "@/components/layout/navbar"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Eye, EyeOff, Copy, Check } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function SettingsPage() {
  const [theme, setTheme] = useState("dark")
  const [storeUploads, setStoreUploads] = useState(true)
  const [showExperimental, setShowExperimental] = useState(false)
  const [enableWebcam, setEnableWebcam] = useState(true)
  const [webhookUrl, setWebhookUrl] = useState("")
  const [showApiKey, setShowApiKey] = useState(false)
  const [copied, setCopied] = useState(false)
  const { toast } = useToast()

  const apiKey = "sk_test_1234567890abcdef"

  const handleCopyApiKey = () => {
    navigator.clipboard.writeText(apiKey)
    setCopied(true)
    toast({
      title: "Copied to clipboard",
      description: "API key has been copied to your clipboard",
    })
    setTimeout(() => setCopied(false), 2000)
  }

  const handleSaveWebhook = () => {
    toast({
      title: "Settings saved",
      description: "Your webhook URL has been updated",
    })
  }

  return (
    <div className="min-h-screen">
      <Navbar />

      <main className="container py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">Manage your preferences and integrations</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Preferences */}
          <Card className="p-6">
            <h2 className="mb-6 text-xl font-semibold">Preferences</h2>

            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="theme">Theme</Label>
                <Select value={theme} onValueChange={setTheme}>
                  <SelectTrigger id="theme">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border/50 bg-muted/20 p-4">
                <div className="space-y-0.5">
                  <Label htmlFor="store-uploads">Store uploads for history</Label>
                  <p className="text-sm text-muted-foreground">Keep uploaded files in your detection history</p>
                </div>
                <Switch id="store-uploads" checked={storeUploads} onCheckedChange={setStoreUploads} />
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border/50 bg-muted/20 p-4">
                <div className="space-y-0.5">
                  <Label htmlFor="experimental">Show experimental artifacts</Label>
                  <p className="text-sm text-muted-foreground">Display additional detection features in beta</p>
                </div>
                <Switch id="experimental" checked={showExperimental} onCheckedChange={setShowExperimental} />
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border/50 bg-muted/20 p-4">
                <div className="space-y-0.5">
                  <Label htmlFor="webcam">Enable webcam capture</Label>
                  <p className="text-sm text-muted-foreground">Allow capturing images directly from your webcam</p>
                </div>
                <Switch id="webcam" checked={enableWebcam} onCheckedChange={setEnableWebcam} />
              </div>
            </div>
          </Card>

          {/* Integrations */}
          <Card className="p-6">
            <h2 className="mb-6 text-xl font-semibold">Integrations</h2>

            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="webhook">Webhook URL</Label>
                <p className="text-sm text-muted-foreground">Receive notifications when detections are completed</p>
                <div className="flex gap-2">
                  <Input
                    id="webhook"
                    type="url"
                    placeholder="https://your-domain.com/webhook"
                    value={webhookUrl}
                    onChange={(e) => setWebhookUrl(e.target.value)}
                  />
                  <Button onClick={handleSaveWebhook}>Save</Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="api-key">API Key</Label>
                <p className="text-sm text-muted-foreground">Use this key to authenticate API requests</p>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      id="api-key"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      readOnly
                      className="pr-10 font-mono"
                    />
                    <Button
                      size="icon"
                      variant="ghost"
                      className="absolute right-0 top-0 h-full"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                  <Button onClick={handleCopyApiKey} variant="outline">
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            </div>
          </Card>

          {/* Privacy */}
          <Card className="p-6 lg:col-span-2">
            <h2 className="mb-4 text-xl font-semibold">Privacy & Data</h2>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground leading-relaxed">
                Your uploaded files are processed securely and can be stored for history tracking if enabled. You can
                delete any detection from your history at any time. We do not share your data with third parties.
              </p>
              <div className="flex gap-3">
                <Button variant="outline">Export My Data</Button>
                <Button variant="destructive">Delete All History</Button>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  )
}
