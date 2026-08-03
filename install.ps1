# Install the moe binary. Usage:
#   irm https://raw.githubusercontent.com/JGalego/Moe/main/install.ps1 | iex
# Environment:
#   $env:MOE_VERSION   release tag to install (default: latest)
#   $env:MOE_BIN_DIR   install location (default: %LOCALAPPDATA%\moe\bin)
$ErrorActionPreference = 'Stop'

$repo    = 'JGalego/Moe'
$version = if ($env:MOE_VERSION) { $env:MOE_VERSION } else { 'latest' }
$binDir  = if ($env:MOE_BIN_DIR) { $env:MOE_BIN_DIR } else { "$env:LOCALAPPDATA\moe\bin" }
$target  = 'x86_64-pc-windows-msvc'

$url = if ($version -eq 'latest') {
  "https://github.com/$repo/releases/latest/download/moe-$target.tar.gz"
} else {
  "https://github.com/$repo/releases/download/$version/moe-$target.tar.gz"
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  Write-Host "downloading $url"
  Invoke-WebRequest -Uri $url -OutFile "$tmp\moe.tar.gz"
  tar -xzf "$tmp\moe.tar.gz" -C $tmp
  New-Item -ItemType Directory -Force -Path $binDir | Out-Null
  Copy-Item "$tmp\moe-$target\moe.exe" "$binDir\moe.exe" -Force
  Write-Host "installed to $binDir\moe.exe"

  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if ($userPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$binDir", 'User')
    Write-Host "added $binDir to your PATH (restart your shell)"
  }
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
