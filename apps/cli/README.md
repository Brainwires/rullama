# rullama-cli

> The agentic CLI of the **rullama** product family (installed binary: `rullama`).
> Lives at `apps/cli/` in the [rullama](https://github.com/Brainwires/rullama)
> monorepo. It builds on the platform crates from the sibling `rullama-framework`
> repo, resolved via path-deps (`../../../rullama-framework`); these flip to
> crates.io version pins once the harness is published.

An AI-powered agentic CLI tool for autonomous coding assistance, built in Rust.

## Features

- 🤖 **Multi-Agent Architecture**: Orchestrator and worker agents for complex task decomposition
- 🔐 **Authentication**: BYOK — bring your own provider key (Anthropic / OpenAI / Google / Ollama-local / …)
- 🛠️ **Rich Tool System**: File operations, bash execution, git integration, web operations
- 🔌 **MCP Client**: Connect to Model Context Protocol servers for extended functionality
- 💬 **Flexible Chat Modes**: Interactive, single-shot, batch, TUI, and MCP server modes
- 📝 **Multiple Output Formats**: Full, plain, and JSON formats for scripting and automation
- 🔇 **Quiet Mode**: Clean output perfect for shell scripts and pipelines
- 📊 **Cost Tracking**: Monitor API usage and costs across providers
- 🎯 **Planning Mode**: Review execution plans before running
- 🧠 **Infinite Context**: Entity extraction, relationship graphs, and semantic search for unlimited conversation memory
- 📋 **Task Management**: Interactive task lists with dependencies, time tracking, and blocking
- 🌲 **Plan Branching**: Create sub-plans, merge branches, and view plan hierarchies
- 📄 **Plan Templates**: Save and reuse plan templates with variable substitution
- 🔍 **Plan Search**: Semantic search across all stored plans
- 🌐 **Remote Control** *(legacy, off by default)*: control CLI agents over a cloud relay — behind the `remote-bridge` feature; the hosted backend is discontinued
- 🌳 **Collapsible Journal Tree**: TUI Journal view renders conversation as a navigable, collapsible tree (Turn → Message → ToolCall → SubAgentSpawn) with vim-style `j/k/h/l` navigation
- 🔭 **Sub-Agent Viewer** (`Ctrl+B`): Live split-pane view of running sub-agents with status icons, activity detail, and direct IPC messaging
- 🧪 **Local LLM Inference** *(optional)*: Run models locally via the framework's `rullama-provider` crate (`--features llama-cpp-2`)

## Installation

### From Source

```bash
cargo build --release
cargo install --path .
```

### Binary

Download the latest release from the releases page.

## Providers

rullama CLI is provider-agnostic — point it at Anthropic, OpenAI, Google, Groq, Ollama (local), Amazon Bedrock, Google Vertex AI, Together, Fireworks, MiniMax. Three ways to configure a provider:

**1. First-run picker (interactive).** The very first time you run `rullama chat` or `rullama task` with no config, you'll see a picker listing the chat-capable providers. The choice is persisted to `~/.rullama/config.json`.

**2. Environment variables.** If any of these are set, the CLI picks the provider up automatically and skips the picker:

| Provider | Env var |
|----------|---------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| OpenAI (GPT) | `OPENAI_API_KEY` |
| Google (Gemini) | `GEMINI_API_KEY` *or* `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Ollama (local) | `OLLAMA_HOST` (just presence detected) |
| Together / Fireworks / MiniMax | `TOGETHER_API_KEY` / `FIREWORKS_API_KEY` / `MINIMAX_API_KEY` |
| Bedrock | AWS credential chain (`AWS_ACCESS_KEY_ID`, …) |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` + project ID |

You can also override per-invocation with `RULLAMA_PROVIDER=anthropic` or `--provider anthropic`.

**3. Explicit login.** Store credentials in your system keyring so they persist across shells:

```bash
# Connect to any provider (BYOK)
rullama auth login --provider anthropic
rullama auth login --provider openai --model gpt-5-mini
rullama auth login --provider ollama --base-url http://localhost:11434
rullama auth login --provider bedrock --region us-west-2
rullama auth login --provider vertex-ai --project-id my-gcp-project
```

### Switching providers

Inside an interactive TUI session, use slash commands:

```
/provider                # list providers, shows current with *
/provider anthropic      # switch; persists to config
/auth status             # show credential state for the active provider
```

From the shell, just pass `--provider` to any command:

```bash
rullama chat --provider anthropic --prompt "explain ownership"
rullama task --provider ollama "summarize src/main.rs"
```

## Quick Start

### Authentication

First, pick a provider — either run `rullama chat` and use the first-run picker, or explicitly:

```bash
rullama auth login --provider anthropic        # Claude
rullama auth login --provider ollama           # local, no API key
```

Or just export an API key and the CLI picks it up:

```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here
```

### Chat Modes

rullama CLI offers multiple chat modes for different use cases:

#### Interactive Mode (Default)

Start an interactive conversation with full-screen prompts and formatting:

```bash
rullama chat
rullama chat --model claude-3-5-sonnet-20241022
```

#### Single-Shot Mode

Send a single prompt and get an immediate response (perfect for scripting):

```bash
rullama chat --prompt "Explain Rust ownership"
rullama chat --prompt "What is 2+2?" --format=plain
rullama chat --prompt "Calculate 5*3" --quiet --format=plain
```

#### Batch Mode

Process multiple prompts from stdin, one per line:

```bash
# From a file
cat questions.txt | rullama chat --batch

# Pipe multiple questions
printf "What is 2+2?\nWhat is 10-3?\n" | rullama chat --batch

# Get JSON output for batch processing
cat prompts.txt | rullama chat --batch --format=json > results.json
```

#### TUI Mode

Full-screen terminal user interface with rich formatting:

```bash
rullama chat --tui
```

#### Output Formats

Control how responses are displayed:

- **`--format=full`** (default): Rich formatting with labels and colors
- **`--format=plain`**: Just the response text, no decoration
- **`--format=json`**: Structured JSON output with metadata

```bash
rullama chat --prompt "Hello" --format=full
# Output: "Assistant: Hello! How can I help you today?"

rullama chat --prompt "Hello" --format=plain
# Output: "Hello! How can I help you today?"

rullama chat --prompt "Hello" --format=json
# Output: {"model": "...", "response": "Hello! How can I help you today?"}
```

#### Quiet Mode

Suppress decorative output for clean scripting:

```bash
# No welcome banner, spinners, or formatting
echo "What is 2+2?" | rullama chat --quiet

# Perfect for scripts
ANSWER=$(rullama chat --prompt "What is 7*8?" --quiet --format=plain)
echo "The answer is: $ANSWER"
```

#### MCP Server Mode

Expose the CLI as an MCP server over stdio:

```bash
rullama chat --mcp-server
```

#### Practical Examples

```bash
# Quick calculation
rullama chat --prompt "What is 15% of 200?" --quiet --format=plain

# Code review from stdin
cat myfile.rs | rullama chat --prompt "Review this code: $(cat -)" --format=plain

# Batch processing with results
cat questions.txt | rullama chat --batch --format=json > results.json

# Pipeline integration
git diff | rullama chat --prompt "Summarize these changes" --quiet
```

For comprehensive documentation on all chat modes, see [docs/interface/CLI_CHAT_MODES.md](docs/interface/CLI_CHAT_MODES.md).

### Configuration

View current configuration:

```bash
rullama config --list
```

Set configuration values:

```bash
rullama config --set provider=anthropic
rullama config --set permission_mode=auto
```

### MCP Servers

Add an MCP server:

```bash
rullama mcp add project-rag "project-rag serve"
```

Connect to an MCP server:

```bash
rullama mcp connect project-rag
```

List available tools:

```bash
rullama mcp tools
```

### Remote Control *(legacy — off by default)*

> The remote bridge let you drive CLI agents from a web UI over a cloud relay.
> The hosted backend (Brainwires Studio) is **discontinued**, so this is disabled
> by default and gated behind the `remote-bridge` cargo feature. The code is kept
> because a similar capability is likely to return — point `--url` at your own
> compatible relay to use it.

Build with the feature, then configure the bridge against your own backend:

```bash
cargo build --release --features remote-bridge
rullama remote config --enabled true --url <your-relay-url> --api-key <your-key>
rullama remote start      # start | stop | status | config
```

| Option | Description |
|--------|-------------|
| `--enabled true/false` | Enable/disable the remote bridge |
| `--url <backend-url>` | Relay backend URL (no default — discontinued hosted one removed) |
| `--api-key <key>` | API key for the relay |
| `--heartbeat <seconds>` | Heartbeat interval |

The bridge reports running-agent status to the backend and accepts messages,
slash commands, and cancellations. For the protocol/architecture, see
[docs/distributed-swarms/IPC_AND_REMOTE_CONTROL.md](docs/distributed-swarms/IPC_AND_REMOTE_CONTROL.md).

### View Models

List all available models:

```bash
rullama models
```

Filter by provider:

```bash
rullama models --provider anthropic
```

### Cost Tracking

View API usage and costs:

```bash
rullama cost
rullama cost --period week
```

### Slash Commands

rullama CLI includes powerful slash commands for codebase exploration and semantic search:

#### Project RAG Commands

```bash
# Index your codebase for semantic search
/project:index [path]

# Search indexed code using natural language
/project:query <search_query>

# Advanced search with filters
/project:search <query> [extensions] [languages]

# View index statistics
/project:stats

# Search git commit history
/project:git-search <query> [max_commits]

# Clear index
/project:clear
```

**Example usage:**
```bash
# Start chat and use commands
rullama chat

> /project:index ~/projects/my-app
> /project:query authentication implementation
> /project:search database rs Rust
> /project:git-search bug fix 20
```

For detailed documentation, see [docs/interface/SLASH_COMMANDS_RAG.md](docs/interface/SLASH_COMMANDS_RAG.md).

### Task Management Commands

rullama CLI includes a comprehensive task management system with dependency tracking and time estimates:

```bash
# List all tasks
/task:list

# Add a new task
/task:add <task description>

# Start working on a task (marks as in_progress)
/task:start <task_id>

# Complete a task
/task:complete <task_id>

# Skip a task
/task:skip <task_id>

# Block a task (mark as blocked)
/task:block <task_id>

# Set task dependencies
/task:depends <task_id> <depends_on_id>

# Show ready tasks (all dependencies met)
/task:ready

# Show time tracking info
/task:time [task_id]
```

**Example workflow:**
```bash
> /task:add Implement authentication
> /task:add Write unit tests
> /task:depends 2 1   # Tests depend on auth
> /task:ready         # Shows task 1 is ready
> /task:start 1
> /task:complete 1
> /task:ready         # Now task 2 is ready
```

### Plan Management Commands

Create, manage, and execute structured execution plans:

```bash
# List all plans
/plans

# Show plan details
/plan:show <plan_id>

# Activate a plan for execution
/plan:activate <plan_id>

# Execute the active plan with AI assistance
/plan:execute

# Pause plan execution
/plan:pause

# Resume a paused plan
/plan:resume
```

### Plan Branching

Create sub-plans for complex features and merge them back:

```bash
# Create a branch from the active plan
/plan:branch <branch_name> <task_description>

# View plan hierarchy tree
/plan:tree [plan_id]

# Merge a branch back (mark as completed)
/plan:merge [plan_id]

# Search plans by text
/plan:search <query>
```

**Example branching workflow:**
```bash
> /plan:activate abc123
> /plan:branch auth-feature "Implement OAuth2 authentication"
# Branch created with ID def456
> /plan:activate def456   # Work on branch
> /plan:execute
> /plan:merge             # Mark as merged
> /plan:activate abc123   # Back to parent plan
> /plan:tree              # View full hierarchy
```

### Plan Templates

Save and reuse plan templates with variable substitution:

```bash
# List all templates
/templates

# Save current plan as template
/template:save <name> [description]

# Show template details
/template:show <name>

# Create new plan from template
/template:use <name> [var1=value1] [var2=value2]

# Delete a template
/template:delete <name>
```

**Example template usage:**
```bash
# Save a plan as reusable template
> /template:save api-endpoint "Template for REST API endpoints"

# Reuse with variables
> /template:use api-endpoint resource=users method=GET

# Variables in templates use {{variable}} syntax
```

### Infinite Context Memory

rullama CLI features an advanced context management system that provides effectively unlimited conversation memory:

#### How It Works

1. **Entity Extraction**: As you chat, the system automatically extracts named entities from your messages:
   - Files (`src/main.rs`, `config.json`)
   - Functions (`fn process_data`, `function handleClick`)
   - Types (`struct User`, `class Config`)
   - Variables, errors, commands, and concepts

2. **Relationship Graph**: Entities are connected in a knowledge graph tracking:
   - Co-occurrence (entities mentioned together)
   - Containment (file contains function)
   - Dependencies and references
   - Modifications and definitions

3. **Tiered Memory**: Conversations are stored in multiple tiers:
   - **Hot**: Recent messages (always available)
   - **Warm**: Important messages from current session
   - **Cold**: Archived messages with semantic search

4. **Smart Context Injection**: When you ask a question, the system:
   - Analyzes if historical context is needed
   - Searches for relevant past messages using semantic similarity
   - Injects the most relevant context automatically

#### Benefits

- **Never lose context**: Reference code, decisions, or discussions from hours ago
- **Automatic**: No manual tagging or bookmarking required
- **Efficient**: Only retrieves what's relevant, preserving token budget
- **Persistent**: Conversations survive across sessions via LanceDB storage

For technical details, see [docs/infinite-context/INFINITE_CONTEXT.md](docs/infinite-context/INFINITE_CONTEXT.md).

## Configuration

Configuration is stored in `~/.rullama/config.json`:

```json
{
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "permission_mode": "auto",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

## Supported Providers

Provider integrations are supplied by the framework's `rullama-provider` crate
(BYOK — bring your own key):

- **Anthropic** (Claude models)
- **OpenAI** (GPT models, o1)
- **Google** (Gemini models)
- **Ollama** (local models)
- …plus Groq, Together, Fireworks, MiniMax, Amazon Bedrock, and Google Vertex AI.

## Development

### Building

```bash
cargo build                                        # Debug build
cargo build --release                              # Release build (no optional features)
cargo build --release --features llama-cpp-2         # With local LLM support (adds llama-cpp-2)
cargo build --release --features interpreter-all   # With JS/Python code interpreters
cargo build --release --features full              # Everything enabled
```

> **Note:** `llama-cpp-2` is not included in default features to keep build times fast. All local inference components have fallbacks and work without it.

### Running Tests

```bash
cargo test
```

### Development Mode

```bash
cargo run -- auth login
cargo run -- chat
```

## Architecture

rullama CLI is built on the **rullama-framework** platform crates, exposed through a feature-gated facade. The CLI depends on a single facade crate from the sibling repo:

```toml
rullama = { path = "../../../rullama-framework/crates/rullama", features = ["full"] }
```

The framework crates, grouped by function:

| Group | Crates |
|-------|--------|
| **Core** | `rullama-core`, `rullama-tool-runtime`, `rullama-tool-builtins`, `rullama-agent`, `rullama-inference` |
| **Intelligence** | `rullama-provider`, `rullama-knowledge`, `rullama-prompting`, `rullama-rag`, `rullama-storage`, `rullama-stores`, `rullama-memory` |
| **Integration** | `rullama-mcp-client`, `rullama-mcp-server`, `rullama-network` |
| **Security** | `rullama-permission` |
| **Execution** | `rullama-wasm`, `rullama-autonomy` |
| **Hardware & Training** | `rullama-hardware`, `rullama-finetune`, `rullama-a2a` |

### CLI Layer

The CLI adds application-specific code on top of the framework:

- **Commands** (`src/cli/`): `clap`-based CLI commands and chat modes (interactive, single-shot, batch, TUI, MCP server)
- **Providers** (`src/providers/`): BYOK provider factory over the framework's `ChatProviderFactory` (Anthropic, OpenAI, Google, Groq, Ollama, Bedrock, Vertex AI, Together, Fireworks, MiniMax)
- **Config**: User configuration, MCP server registry, API key storage (system keyring)

## License

Dual-licensed under either of MIT or Apache-2.0 at your option — the same terms
as the rest of the [rullama](https://github.com/Brainwires/rullama) repo.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for details.
