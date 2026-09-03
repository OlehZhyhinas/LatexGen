// Deployment configuration.
//
// This file describes the *full* build: weights served from this origin and a
// server tier that may be reachable. scripts/build-static.mjs replaces it for
// the static build (GitHub Pages), where nothing is served but files — see
// docs/deploy.md.
//
// STATIC_BUILD switches off everything that needs a server process: the
// server tier in the ladder, the Tab API and the compute mesh. The on-device
// ladder, validation, history, export and share links are unaffected.
export const STATIC_BUILD = false;

// Where transformers.js fetches the ONNX weights from. null means this
// origin's models/ directory; otherwise a CDN host plus the path template
// that turns a model key ("intellitex") into a URL on it.
export const MODEL_HOST = null;
export const MODEL_PATH_TEMPLATE = null;
