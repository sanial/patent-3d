# Deploy patent-3d to Cloud Run.
# Usage:
#   pwsh ./deploy.ps1                       # uses defaults below
#   pwsh ./deploy.ps1 -Project foo -Region us-west1

param(
    [string]$Project = "patent-view-495221",
    [string]$Region  = "us-central1",
    [string]$Service = "patent-3d",
    [string]$Repo    = "patent-3d",
    [string]$Secret  = "gemma-api-key"
)

$ErrorActionPreference = "Stop"

Write-Host "→ Setting project to $Project" -ForegroundColor Cyan
gcloud config set project $Project | Out-Null

Write-Host "→ Enabling required services" -ForegroundColor Cyan
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    secretmanager.googleapis.com | Out-Null

# ── Artifact Registry repo ──────────────────────────────────────────────────
gcloud artifacts repositories describe $Repo --location=$Region --format="value(name)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "→ Creating Artifact Registry repo '$Repo'" -ForegroundColor Cyan
    gcloud artifacts repositories create $Repo `
        --repository-format=docker `
        --location=$Region `
        --description="patent-3d container images"
} else {
    Write-Host "→ Artifact Registry repo '$Repo' already exists" -ForegroundColor DarkGray
}

# ── Secret Manager: GEMMA_API_KEY ───────────────────────────────────────────
gcloud secrets describe $Secret --format="value(name)" *> $null
if ($LASTEXITCODE -ne 0) {
    $envPath = Join-Path $PSScriptRoot "python\.env"
    if (-not (Test-Path $envPath)) { throw "Missing $envPath with GEMMA_API_KEY=" }
    $line = (Get-Content $envPath | Where-Object { $_ -match '^GEMMA_API_KEY=' } | Select-Object -First 1)
    if (-not $line) { throw "GEMMA_API_KEY not found in $envPath" }
    $key = ($line -replace '^GEMMA_API_KEY=', '').Trim().Trim('"').Trim("'")
    if (-not $key) { throw "GEMMA_API_KEY is empty in $envPath" }

    Write-Host "→ Creating secret '$Secret'" -ForegroundColor Cyan
    gcloud secrets create $Secret --replication-policy=automatic | Out-Null
    $tmp = New-TemporaryFile
    try {
        Set-Content -Path $tmp -Value $key -NoNewline -Encoding utf8
        gcloud secrets versions add $Secret --data-file=$tmp | Out-Null
    } finally {
        Remove-Item $tmp -Force
    }
} else {
    Write-Host "→ Secret '$Secret' already exists (skipping creation)" -ForegroundColor DarkGray
}

# Grant Cloud Run runtime SA access to the secret.
$projectNum = (gcloud projects describe $Project --format="value(projectNumber)").Trim()
$runtimeSa  = "$projectNum-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding $Secret `
    --member="serviceAccount:$runtimeSa" `
    --role="roles/secretmanager.secretAccessor" --quiet | Out-Null

# ── Build & deploy in one shot ──────────────────────────────────────────────
$image = "${Region}-docker.pkg.dev/${Project}/${Repo}/${Service}:latest"

Write-Host "→ Building image with Cloud Build → $image" -ForegroundColor Cyan
gcloud builds submit --tag $image .

Write-Host "→ Deploying to Cloud Run service '$Service'" -ForegroundColor Cyan
gcloud run deploy $Service `
    --image $image `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --memory 4Gi `
    --cpu 2 `
    --timeout 600 `
    --concurrency 4 `
    --max-instances 3 `
    --port 8080 `
    --set-secrets "GEMMA_API_KEY=${Secret}:latest"

$url = (gcloud run services describe $Service --region $Region --format="value(status.url)").Trim()
Write-Host ""
Write-Host "✓ Deployed: $url" -ForegroundColor Green
Write-Host "  Health:  $url/api/health"
