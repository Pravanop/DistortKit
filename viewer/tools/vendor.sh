#!/bin/sh
# Rebuild viewer/vendor/{three.js,gifenc.js} from npm (needs node/npm). The outputs are committed so the viewer runs with no build.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
cd "$TMP"
npm init -y >/dev/null
npm install three@0.186.1 gifenc@1.0.3 esbuild >/dev/null
cp "$HERE/vendor/three-entry.js" entry.js
npx esbuild entry.js --bundle --minify --format=esm --legal-comments=none --outfile="$HERE/vendor/three.js"
cp node_modules/gifenc/dist/gifenc.esm.js "$HERE/vendor/gifenc.js"
cp node_modules/three/LICENSE "$HERE/vendor/THREE-LICENSE" 2>/dev/null || true
cp node_modules/gifenc/LICENSE.md "$HERE/vendor/GIFENC-LICENSE.md" 2>/dev/null || true
rm -rf "$TMP"
ls -la "$HERE/vendor"
