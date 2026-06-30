# Remote Bridge

The Remote Bridge connects your local `rullama` CLI to a backend (Brainwires Studio or a compatible server), enabling real-time monitoring and control of your local AI agents from a web dashboard.

---

## Overview

Without the Remote Bridge, the CLI is fully self-contained: you interact with AI agents locally via the terminal or TUI. The Remote Bridge adds an optional outbound connection that lets a web dashboard:

- See all running agents and their status
- Stream live output from any agent
- Send input or commands to an agent
- Spawn new agent sessions remotely

The CLI initiates all connections — no inbound ports are opened.

---

## Architecture

```
  ┌──────────────────────────────┐
  │         Your Machine         │
  │                              │
  │  ┌──────────────┐            │
  │  │  rullama  │  HTTPS/WSS │       ┌─────────────────────┐
  │  │     CLI      │◄──────────────────►│  Backend (Studio)   │
  │  └──────┬───────┘            │       └──────────┬──────────┘
  │         │ Unix socket         │                  │ SSE / WebSocket
  │  ┌──────▼───────┐            │       ┌──────────▼──────────┐
  │  │  AI Agents   │            │       │    Web Dashboard    │
  │  │ (subprocesses│            │       │    /cli/remote      │
  │  └──────────────┘            │       └─────────────────────┘
  └──────────────────────────────┘
```

**Key properties:**
- **Outbound only** — CLI connects to the backend; backend never initiates a connection back
- **API key auth** — all requests carry your `bw_*` API key in an `Authorization: Bearer` header
- **Dual transport** — prefers Supabase Realtime (WebSocket) for commands; falls back to HTTP polling (heartbeat) automatically
- **Agent IPC** — the bridge talks to local agents via encrypted Unix sockets (`~/.rullama/sessions/`)

---

## Authentication

### Brainwires Studio users

1. Log in to Studio, navigate to **Settings → API Keys**, and generate a key.
2. Authenticate the CLI:
   ```bash
   rullama auth login
   # Paste your bw_prod_... key when prompted
   ```
3. The CLI exchanges the key for a user profile and Supabase credentials via `POST /api/cli/auth`.

The Studio backend **never returns your provider API keys** (Anthropic, OpenAI, etc.) — those stay on the server side. The CLI uses your local provider config or its own API keys.

### Direct provider auth (no Studio)

If you don't have a Studio account, authenticate directly with an AI provider:

```bash
# Anthropic
rullama auth login --provider anthropic

# OpenAI
rullama auth login --provider openai

# Ollama (no key required)
rullama auth login --provider ollama

# AWS Bedrock (uses ~/.aws/credentials)
rullama auth login --provider bedrock

# Google Vertex AI (uses GOOGLE_APPLICATION_CREDENTIALS)
rullama auth login --provider vertex-ai
```

Direct provider auth gives you full CLI functionality (chat, agents, tools) but disables the Remote Bridge — there is no backend to relay through.

### API key format

```
bw_<env>_<32 lowercase alphanumeric chars>

Examples:
  bw_prod_k4j2h5g3n8m9p1q7r6s2t4v8w3x5y9z0
  bw_dev_abcdefghijklmnopqrstuvwxyz123456
```

- `prod` keys → `https://brainwires.studio`
- `dev` keys → `https://dev.brainwires.net`
- Custom backend → pass `--backend <url>` to `rullama auth login`

Keys are stored in your system keyring (not in plain files).

---

## Remote Bridge Setup

### Enable and start

```bash
# Enable the bridge (persists in ~/.rullama/config.json)
rullama remote config --enabled true

# Start the daemon (runs in the background)
rullama remote start

# Check status
rullama remote status

# View logs
rullama remote log --follow
```

### Auto-start on CLI launch

When `auto_start = true` (the default when `enabled = true`), the bridge daemon starts automatically whenever you open a chat session. You don't need to run `rullama remote start` manually.

### Stop

```bash
rullama remote stop
```

---

## Device Pairing (zero-API-key onboarding)

Pairing lets you link a machine to your Studio account without manually copying an API key — useful for headless servers or onboarding teammates.

**CLI side:**
```bash
rullama remote pair
```
The CLI displays a 6-character code, e.g. `A3K9MZ`, and polls for confirmation.

**Web side:**
Open Studio → **Settings → Devices → Pair Device**, enter the code. Studio generates an API key, and the CLI receives and stores it automatically.

**Flow:**
```
CLI                                   Backend
 │─── POST /api/remote/pair/initiate ──►│  (unauthenticated)
 │◄── { request_id, pairing_code } ─────│
 │                                      │
 │   [user enters code in web UI]        │
 │                                      │  POST /api/remote/pair/confirm
 │                                      │  (web session auth)
 │─── GET /api/remote/pair/status/{id} ─►│  (polled every 2s)
 │◄── { status: "confirmed", api_key } ─│  (one-time read)
 │                                      │
 │  [saves key, authenticates]           │
```

The pairing code expires in 5 minutes. The API key is shown exactly once.

---

## Connection Flow

Once you have an API key and the bridge is enabled:

```
CLI                                     Backend
 │─── POST /api/remote/connect ─────────►│
 │    { device_fingerprint, hostname,    │
 │      os, version, protocol_hello }    │
 │◄── { session_token,                  │
 │      realtime_token, channel_name,   │
 │      device_status, org_policies }   │
 │                                      │
 │  [device_status == "allowed"?]       │
 │                                      │
 │─── Subscribe Supabase Realtime ──────►│  (preferred)
 │         OR                            │
 │─── POST /api/remote/heartbeat ───────►│  (polling fallback, every 30s)
```

### Device allowlist

The backend tracks devices by a SHA-256 fingerprint derived from machine-id + hostname + OS. Three modes:

| Mode | Behavior |
|------|----------|
| `open` | All devices auto-approved |
| `approve_new` | First connection is "pending" until approved in web UI |
| `strict` | Only explicitly approved devices may connect |

If your device is blocked, the CLI logs the rejection and exits the bridge loop.

---

## Protocol Negotiation

At connect time, the CLI sends a `ProtocolHello`:

```json
{
  "type": "hello",
  "supported_versions": ["1.1", "1.0"],
  "preferred_version": "1.1",
  "capabilities": [
    "streaming",
    "tools",
    "attachments",
    "priority",
    "device_allowlist",
    "permission_relay"
  ]
}
```

The backend responds with `ProtocolAccept` selecting the highest mutually supported version and the enabled capability subset.

### Capability flags

| Capability | Description |
|------------|-------------|
| `streaming` | Real-time output chunks pushed to backend |
| `tools` | Tool execution support |
| `attachments` | File attachment upload/download |
| `priority` | Command priority queuing |
| `device_allowlist` | Device fingerprint verification |
| `permission_relay` | Remote approval prompts for dangerous tools |
| `presence` | Web viewer tracking |
| `compression` | Message compression |

---

## Transport Modes

### Supabase Realtime (preferred)

A WebSocket connection authenticated with a short-lived JWT. Commands from Studio arrive instantly; agent output is pushed in real time.

- **Channel**: `cli:{userId}`
- **Token**: returned in `/api/remote/connect` response
- **Reconnect**: 5-second delay, unlimited retries

### HTTP polling (fallback)

When Realtime is unavailable, the CLI falls back to polling:

```
Every {heartbeat_interval_secs} seconds:
  POST /api/remote/heartbeat
  Body:  { session_token, agents[], system_load, hostname, os, version }
  Reply: { commands: BackendCommand[] }
```

Agent stream data is pushed immediately via a separate `POST /api/remote/stream` call (does not wait for the next heartbeat).

---

## Configuration Reference

Config lives in `~/.rullama/config.json` under the `remote` key.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the remote bridge |
| `backend_url` | string | `https://brainwires.studio` | Backend base URL |
| `api_key` | string? | (session key) | Override key (falls back to session) |
| `heartbeat_interval_secs` | u32 | `30` | Polling interval (fallback mode) |
| `reconnect_delay_secs` | u32 | `5` | Delay between reconnect attempts |
| `max_reconnect_attempts` | u32 | `0` | `0` = unlimited |
| `auto_start` | bool | `true` | Start bridge when `enabled = true` and CLI opens |
| `blocked_remote_commands` | string[] | `["exec"]` | Commands backend cannot invoke |
| `warned_remote_commands` | string[] | `["exit"]` | Commands that log a warning when invoked remotely |

### `rullama remote config` options

```bash
rullama remote config                 # show current config
rullama remote config --enabled true
rullama remote config --url https://my-backend.example.com
rullama remote config --api-key bw_prod_...
rullama remote config --heartbeat 60
```

---

## `rullama remote` Command Reference

| Command | Description |
|---------|-------------|
| `remote start [--force] [--foreground]` | Start bridge daemon (background by default) |
| `remote stop` | Stop the running daemon |
| `remote status` | Show connection status, PID, config |
| `remote log [-f] [-n N] [--clear]` | View or follow bridge logs |
| `remote pair` | Pair this device with Studio (no API key needed) |
| `remote config [OPTIONS]` | View or update bridge settings |
| `remote daemon` | Internal: run the daemon process directly (debugging) |

Logs are written to `~/.rullama/remote-bridge.log`. PID file: `~/.rullama/remote-bridge.pid`.

---

## API Contract

This section documents every endpoint the CLI calls. Useful for anyone building a compatible backend.

All requests must be over HTTPS. CLI-side requests authenticate with `Authorization: Bearer <api_key>` unless noted.

### Authentication

#### `POST /api/cli/auth`

Exchange an API key for user profile and Supabase credentials.

**Request:**
```json
{ "apiKey": "bw_prod_..." }
```

**Response 200:**
```json
{
  "user": {
    "user_id": "uuid",
    "username": "john_doe",
    "display_name": "John Doe",
    "role": "user"
  },
  "supabase": {
    "url": "https://xyz.supabase.co",
    "anonKey": "eyJ..."
  },
  "keyName": "my laptop"
}
```

**Response 401:** Invalid or expired key.

#### `GET /api/cli/auth`

Health check. Returns `200 OK`.

---

### Remote Connection

#### `POST /api/remote/connect`

Register the bridge and receive a session token + Realtime credentials.

**Request:**
```json
{
  "device_fingerprint": "sha256hex",
  "hostname": "my-machine",
  "os": "linux",
  "version": "0.8.0",
  "protocol_hello": {
    "type": "hello",
    "supported_versions": ["1.1", "1.0"],
    "preferred_version": "1.1",
    "capabilities": ["streaming", "tools", ...]
  }
}
```

**Response 200:**
```json
{
  "type": "authenticated",
  "session_token": "...",
  "user_id": "uuid",
  "device_status": "allowed",
  "org_policies": {
    "blocked_tools": [],
    "permission_relay_required": false,
    "device_allowlist_mode": "open",
    "audit_all_commands": false
  },
  "realtime_token": "eyJ...",
  "realtime_url": "wss://xyz.supabase.co/realtime/v1",
  "channel_name": "cli:uuid",
  "use_realtime": true,
  "protocol": { "selected_version": "1.1", "enabled_capabilities": [...] }
}
```

`device_status` values: `"allowed"` | `"pending_approval"` | `"blocked"`. Bridge exits if `"blocked"`.

---

#### `POST /api/remote/heartbeat`

Send agent status; receive pending commands. Used in polling mode.

**Request:**
```json
{
  "session_token": "...",
  "agents": [
    {
      "session_id": "uuid",
      "model": "claude-opus-4-6",
      "is_busy": true,
      "working_directory": "/home/user/project",
      "message_count": 12,
      "last_activity": 1706000000,
      "status": "running",
      "name": "Code Reviewer"
    }
  ],
  "system_load": 0.42,
  "hostname": "my-machine",
  "os": "linux",
  "version": "0.8.0"
}
```

**Response 200:**
```json
{
  "success": true,
  "commands": [ /* BackendCommand[] — see below */ ]
}
```

---

#### `POST /api/remote/stream`

Push agent stream chunks immediately (no polling delay).

**Request:**
```json
{
  "messages": [
    {
      "type": "agent_stream",
      "agent_id": "uuid",
      "chunk_type": "text",
      "content": "I'll start by reading the file..."
    }
  ]
}
```

**Response 200:** `{ "success": true, "processed": 1 }`

---

### Device Pairing

#### `POST /api/remote/pair/initiate`

Start a pairing request. **No authentication required.**

**Request:**
```json
{
  "hostname": "my-machine",
  "os": "linux",
  "cli_version": "0.8.0",
  "device_fingerprint": "sha256hex"
}
```

**Response 200:**
```json
{
  "request_id": "uuid",
  "pairing_code": "A3K9MZ",
  "expires_at": "2026-03-30T12:05:00Z"
}
```

Rate limited: 5 requests/IP/minute.

#### `GET /api/remote/pair/status/{requestId}`

Poll for pairing confirmation. **No authentication required** (request_id is the bearer).

**Response 200:**
```json
{ "status": "pending" }
{ "status": "confirmed", "api_key": "bw_prod_..." }  // one-time, key cleared after
{ "status": "confirmed", "api_key_retrieved": true }  // if polled again after retrieval
{ "status": "expired" }
```

---

### CLI Messages → Backend

These are JSON objects the CLI sends, used in Realtime mode or embedded in heartbeat messages.

| `type` | Description | Key fields |
|--------|-------------|------------|
| `heartbeat` | Agent status update | `session_token`, `agents[]`, `system_load` |
| `command_result` | Result of a backend command | `command_id`, `success`, `output`, `error?` |
| `agent_event` | Agent lifecycle event | `event_type`, `agent_id`, `data` |
| `agent_stream` | Stream output chunk | `agent_id`, `chunk_type`, `content` |
| `permission_request` | Request tool approval | `request_id`, `agent_id`, `tool_name`, `args` |
| `pong` | Heartbeat response | — |

---

### Backend Commands → CLI

These are objects returned in heartbeat `commands[]` or pushed via Realtime.

| `type` | Description | Key fields |
|--------|-------------|------------|
| `send_input` | Send text to agent | `agent_id`, `content` |
| `slash_command` | Run a slash command | `agent_id`, `command`, `args[]` |
| `cancel_operation` | Abort current operation | `agent_id` |
| `spawn_agent` | Start a new agent | `working_directory`, `model?`, `description?` |
| `subscribe` | Start streaming an agent | `agent_id` |
| `unsubscribe` | Stop streaming an agent | `agent_id` |
| `permission_response` | Tool approval decision | `request_id`, `approved` |
| `ping` | Keepalive | — |
| `disconnect` | Graceful shutdown | — |

---

### Stream Chunk Types

| `chunk_type` | Description |
|--------------|-------------|
| `text` | Assistant text output |
| `thinking` | Extended thinking block |
| `tool_call` | Tool invocation (name + args) |
| `tool_result` | Tool output |
| `error` | Error message |
| `system` | System/status message |
| `complete` | Agent turn complete |
| `history` | Historical message replay |
| `user_input` | User message echo |
| `slash_command_result` | Slash command output |

---

## Security Model

### What's protected

- **No inbound ports** — the CLI never listens; the remote cannot initiate a connection
- **HTTPS/WSS only** — all transport is TLS-encrypted
- **API key in keyring** — never stored in plain files; only the session metadata goes to `~/.rullama/session.json`
- **Agent IPC encrypted** — local Unix socket messages use ChaCha20-Poly1305; access requires the session token
- **Device fingerprint** — SHA-256 of machine-id + hostname + OS; prevents key sharing between machines
- **Blocked commands** — by default, remote cannot invoke `exec` (raw shell); configurable via `blocked_remote_commands`

### Organisation policies

When the CLI is part of an organisation, the backend can enforce:

| Policy | Description |
|--------|-------------|
| `blocked_tools` | Array of tool names always denied, even if CLI config allows |
| `permission_relay_required` | Force approval prompts for every tool call |
| `device_allowlist_mode` | Override the user's `open`/`approve_new`/`strict` setting |
| `audit_all_commands` | Log all commands sent to the CLI |

Policies are fetched at connect time and override individual user settings.

---

## What's Not Available Without Studio

The Remote Bridge requires a compatible backend. If you're running the open-source framework without Studio, the following features are unavailable:

- Web dashboard (`/cli/remote`) — real-time bridge/agent tree view
- API key management — Studio generates and rotates `bw_*` keys
- Device allowlisting — managed in Studio settings
- Organisation policies
- Remote agent control from the browser
- Skill registry

**All other CLI features work fully without a backend:** direct chat, local agents, MCP server mode, tool execution, TUI, infinite context, and any provider you authenticate with directly.
