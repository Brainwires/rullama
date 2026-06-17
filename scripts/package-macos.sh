#!/usr/bin/env bash
# Package the desktop app as an (unsigned) macOS .app bundle.
# Signing + notarization need an Apple Developer ID and run separately.
#
# Usage: scripts/package-macos.sh [arm64|x64]   (default: host arch)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${1:-$([ "$(uname -m)" = "arm64" ] && echo arm64 || echo x64)}"
RID="osx-$ARCH"
APP="Rullama"
OUT="$ROOT/dist/$APP.app"

echo "==> Building rust-core (release)"
cargo build --release --manifest-path "$ROOT/rust-core/Cargo.toml"

echo "==> dotnet publish ($RID, self-contained)"
PUB="$ROOT/app/Rullama.Desktop/bin/Release/net9.0/$RID/publish"
dotnet publish "$ROOT/app/Rullama.Desktop/Rullama.Desktop.csproj" \
  -c Release -r "$RID" --self-contained true -p:PublishSingleFile=false

echo "==> Assembling $OUT"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp -R "$PUB/." "$OUT/Contents/MacOS/"
cat > "$OUT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$APP</string>
  <key>CFBundleDisplayName</key><string>rullama</string>
  <key>CFBundleIdentifier</key><string>dev.brainwires.rullama</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>Rullama.Desktop</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
chmod +x "$OUT/Contents/MacOS/Rullama.Desktop"
echo "Done: $OUT  (unsigned — run: codesign / notarytool with a Developer ID to distribute)"
