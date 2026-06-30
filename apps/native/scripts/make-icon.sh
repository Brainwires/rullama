#!/usr/bin/env bash
# Build a macOS-native app icon (rounded-rect "squircle", enlarged art, standard
# margin) + a window PNG, from the public rullama icon. Requires ImageMagick + iconutil.
#   Output: app/Rullama.Desktop/rullama.icns , app/Rullama/Assets/rullama-512.png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-/Users/nightness/Source/Brainwires/rullama/web/public/icons/icon-512.png}"
TMP="$(mktemp -d)"

# 1. Enlarge the art: crop the dark margin so the llama+gear fills the tile.
magick "$SRC" -gravity center -crop 350x350+0+0 +repage -resize 840x840 "$TMP/art.png"

# 2. Rounded-rect (squircle-ish) mask, then apply.
magick -size 840x840 xc:none -draw "roundrectangle 0,0,839,839,188,188" "$TMP/mask.png"
magick "$TMP/art.png" "$TMP/mask.png" -alpha set -compose DstIn -composite "$TMP/tile.png"

# 3. Center on a 1024 transparent canvas (macOS ~10% margin).
magick -size 1024x1024 xc:none "$TMP/tile.png" -gravity center -composite "$TMP/icon-1024.png"

# 4. Window PNG (Avalonia Window.Icon).
sips -z 512 512 "$TMP/icon-1024.png" --out "$ROOT/app/Rullama/Assets/rullama-512.png" >/dev/null

# 5. .icns for the .app bundle.
ISET="$TMP/rullama.iconset"; mkdir -p "$ISET"
for s in 16 32 64 128 256 512; do
  sips -z $s $s "$TMP/icon-1024.png" --out "$ISET/icon_${s}x${s}.png" >/dev/null
  d=$((s*2)); sips -z $d $d "$TMP/icon-1024.png" --out "$ISET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ISET" -o "$ROOT/app/Rullama.Desktop/rullama.icns"
echo "Wrote app/Rullama.Desktop/rullama.icns + app/Rullama/Assets/rullama-512.png"
