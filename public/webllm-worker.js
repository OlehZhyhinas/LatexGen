// WebLLM runs here, off the main thread, so multi-GB model loads and token
// generation never freeze the UI. The page talks to it through
// CreateWebWorkerMLCEngine, which mirrors the normal MLCEngine API.
import { WebWorkerMLCEngineHandler } from "/vendor/webllm/index.js";

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => handler.onmessage(msg);
