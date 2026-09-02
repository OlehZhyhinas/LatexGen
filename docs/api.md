# Tab API

A browser tab cannot accept connections, so LatexGen turns the problem around. With the Tab API on (the plug icon in the header), the tab long-polls the server for jobs and answers them with the models it already has loaded. Each tab has a stable, unguessable address; the drawer shows it with a copy button, a live status, the jobs running now, and a request log.

Jobs run on the same headless pipeline as the buttons, in the background and concurrently (up to three at once), without touching what the user sees.

```bash
B=https://<host>/api/tab/<id>
curl -X POST $B/convert -H 'content-type: application/json' \
  -d '{"text": "the sum from n equals 1 to infinity of 1 over n squared"}'
```

```json
{"latex":"$$\\sum_{n=1}^{\\infty}\\frac{1}{n^{2}}$$","ok":true,"issues":[],"model":"IntelliTeX · specialist","note":"","ms":412}
```

## Endpoints

| Endpoint | Body | Mirrors in the UI |
|---|---|---|
| `POST /convert` | `{"text"}` plus options: `format` (`latex`, `display`, `inline`, `mathml`, `png`), `strict: true` (waits for the second-opinion judge, returned as `judge`), `engine` (`browser` or `server`) | Convert button, including batch lines, escalation, repair |
| `POST /convert` | `{"imageBase64", "mime", "ocr": "auto" or "texo" or "texify"}` | Image drop or paste, "read with the other model" |
| `POST /convert` or `/check` | `{"latex"}` | Check LaTeX tab (validate and repair) |
| `POST /refine` | `{"latex", "instruction", "original"}` | Refine chat |
| `POST /format` | `{"latex", "format"}` | Copy menu formats |
| `POST /status` | `{}` | models loaded, runtime, speed, server |
| `POST /history` | `{}` | this browser's conversion history |

Every response carries the validator verdict (`ok`, `issues`), which tier answered (`model`), the routing note, and timing. The relay forwards bytes only, holds a request for up to 25 seconds, and stores nothing.

## Addresses

The id lives in the tab's `sessionStorage`: it survives reloads of that tab and is different for every tab, so two tabs never share an address. **Regenerate** revokes the current address. Whoever has the URL can use it, so treat it like a password.
