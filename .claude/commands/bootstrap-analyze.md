# Bootstrap Analysis - Generate Project Configuration

Analyze this repository to create Claude Code configuration for adding new embedding model support.

**Arguments:** $ARGUMENTS
(Format: `type=<type> language=<language>`)

---

## YOUR MISSION

You are analyzing this repository to create automated configuration for adding NEW VoyageAI models in the future. This includes:
- **Embedding models** (voyage-3, voyage-code-3, etc.)
- **Reranker models** (rerank-2, rerank-lite-2, etc.)

This is NOT about updating/replacing models - it's about understanding how to ADD support for additional models.

Be thorough - missing a location now means incomplete model additions later.

---

## PHASE 0: Check for Existing Configuration

**IMPORTANT:** Before starting analysis, check if this project already has Claude configuration:

### 0.1 Check for Existing CLAUDE.md
```bash
ls -la .claude/CLAUDE.md 2>/dev/null || echo "No existing CLAUDE.md"
ls -la CLAUDE.md 2>/dev/null || echo "No root CLAUDE.md"
```

If `.claude/CLAUDE.md` or `CLAUDE.md` exists:
1. **READ IT COMPLETELY** - understand what's already documented
2. Note any project-specific instructions, commands, or configurations
3. Identify sections that should be PRESERVED as-is
4. Your analysis will ENHANCE, not replace, existing documentation

### 0.2 Check for Existing Commands/Agents
```bash
ls -la .claude/commands/ 2>/dev/null || echo "No existing commands"
ls -la .claude/agents/ 2>/dev/null || echo "No existing agents"
```

If custom commands or agents exist:
1. **DO NOT OVERWRITE** them unless they're specifically for embedding models
2. Read and understand what they do
3. Integrate your new commands alongside existing ones
4. Note any naming conventions or patterns used

### 0.3 Existing Configuration Summary
After checking, create a mental note:
- **Existing CLAUDE.md:** [Yes/No - summarize key sections if yes]
- **Existing commands:** [List any found]
- **Existing agents:** [List any found]
- **Preservation strategy:** [What to keep, what to merge, what to add]

---

## PHASE 1: Deep Code Analysis

Search comprehensively for how VoyageAI models (embeddings AND rerankers) are currently configured and used:

### 1.1 Model Registry/Configuration
Search for where models are defined:
- Model enums, constants, or lists
- Configuration files with model definitions
- Supported models arrays/maps
- Model metadata (dimensions, capabilities, etc.)

Look for patterns like:
- `SUPPORTED_MODELS`, `AVAILABLE_MODELS`, `MODEL_REGISTRY`
- `voyage`, `voyageai`, `embedding`, `rerank`, `reranker`
- Embedding model strings: `voyage-2`, `voyage-3`, `voyage-large-2`, `voyage-code-2`, etc.
- Reranker model strings: `rerank-1`, `rerank-2`, `rerank-lite-1`, etc.
- Dimension mappings per model (embeddings only)

### 1.2 Model Selection/Validation
Find where models are:
- Validated (checking if model is supported)
- Selected (default model, user choice)
- Passed to APIs
- Used in configuration

### 1.3 Dimension Handling
Search for:
- Dimension constants per model
- Dynamic dimension lookup
- Vector size configurations
- Schema definitions with dimensions

### 1.4 Documentation
Search in:
- README files
- API documentation
- Configuration guides
- Model comparison docs
- Changelog/release notes

### 1.5 Tests
Search for:
- Tests per model
- Model-specific fixtures
- Parameterized tests across models
- Integration tests with model selection

---

## PHASE 2: Generate/Update .claude/CLAUDE.md

**If existing CLAUDE.md was found:**
- PRESERVE all existing sections that aren't about embedding models
- ADD a new "## Embedding Model Configuration" section
- MERGE your findings with any existing model-related content
- Keep the original structure and add to it

**If NO existing CLAUDE.md:**
- Create a new file with the structure below

### Output Location
- If project has `.claude/` directory: use `.claude/CLAUDE.md`
- Otherwise: create `.claude/CLAUDE.md`

Create or update with the following structure (adapt if merging):

```markdown
# [Project Name] - VoyageAI Model Configuration

## Project Overview
[2-3 sentence description of how this project uses VoyageAI models]

**Type:** [detected type]
**Language:** [detected language]
**Last analyzed:** [current date]
**Supports:** [Embeddings / Rerankers / Both]

---

## How Models Are Handled

### Model Registry Location
[Where the list of supported models is defined]

### Model Selection
[How users/code selects which model to use]

### Dimension Handling (Embeddings)
[How dimensions are determined per embedding model]

### Reranker Integration (if applicable)
[How rerankers are configured and used]

---

## Adding a New Embedding Model - Locations to Update

### 1. Configuration Files (REQUIRED)

| File | What to Add |
|------|-------------|
| path/to/config.yaml | Add to embedding models list |
| src/constants.ts | Add to EMBEDDING_MODELS enum |

### 2. Source Code (IF APPLICABLE)

| File | What to Add |
|------|-------------|
| src/models/registry.ts | Add model metadata + dimensions |
| src/validation.ts | Model already validated dynamically |

### 3. Documentation (REQUIRED)

| File | What to Add |
|------|-------------|
| README.md | Add to supported models section |
| docs/models.md | Add model details |

### 4. Tests (RECOMMENDED)

| File | What to Add |
|------|-------------|
| tests/embeddings.test.ts | Add test case for new model |

---

## Adding a New Reranker Model - Locations to Update

(Skip this section if project doesn't support rerankers)

### 1. Configuration Files

| File | What to Add |
|------|-------------|
| path/to/config.yaml | Add to reranker models list |
| src/constants.ts | Add to RERANK_MODELS enum |

### 2. Source Code

| File | What to Add |
|------|-------------|
| src/rerank/models.ts | Add model metadata |

### 3. Documentation

| File | What to Add |
|------|-------------|
| README.md | Add to reranker section |
| docs/reranking.md | Add model details |

### 4. Tests

| File | What to Add |
|------|-------------|
| tests/rerank.test.ts | Add test case |

---

## Current Embedding Models

| Model ID | Dimensions | Status |
|----------|------------|--------|
| voyage-2 | 1024 | Active |
| voyage-3 | 1024 | Active |

## Current Reranker Models

| Model ID | Status |
|----------|--------|
| rerank-1 | Active |
| rerank-2 | Active |

(Leave blank if project doesn't support rerankers)

---

## Commands

### Run Tests
\`\`\`bash
[test command]
\`\`\`

### Build
\`\`\`bash
[build command]
\`\`\`

---

## Special Considerations

- [Does adding a model require database changes?]
- [Are there model-specific API endpoints?]
- [Any feature flags or gradual rollout needed?]
- [Dependencies on other services?]

---

## New Embedding Model Checklist

- [ ] Add model ID to registry in [location]
- [ ] Add dimension mapping in [location] (if different)
- [ ] Update documentation in [locations]
- [ ] Add test coverage
- [ ] Verify existing models still work
- [ ] No breaking changes to existing integrations

## New Reranker Model Checklist

(If project supports rerankers)

- [ ] Add model ID to reranker registry in [location]
- [ ] Update documentation in [locations]
- [ ] Add test coverage
- [ ] Verify existing rerankers still work
```

---

## PHASE 3: Generate Slash Commands

**If existing commands found in Phase 0:**
- Use similar naming conventions (kebab-case, camelCase, etc.)
- Check if VoyageAI-related commands already exist - UPDATE rather than create new
- DO NOT overwrite unrelated commands
- Integrate with existing command structure

Create these files in `.claude/commands/` (skip if equivalent already exists):

### 3.1 `add-voyage-model.md`
Create a customized "add new model" command specific to THIS project that handles BOTH embeddings and rerankers. Include:
- Exact file paths where model needs to be added
- Code patterns to follow (copy existing model format)
- Project-specific validation requirements
- Model type parameter (embedding vs rerank)

### 3.2 `test-voyage-changes.md`
Create a test runner that:
- Tests all models (existing + new)
- Covers both embeddings and rerankers if supported
- Verifies no regressions
- Reports per-model status

### 3.3 `validate-voyage-update.md`
Create a validation checklist that:
- Verifies new model was added to all required locations
- Checks existing models still work
- Reports completeness for both model types

---

## PHASE 4: Output Summary

Print a clear summary when done:

```
═══════════════════════════════════════════════════════════════════
📦 Bootstrap Complete: [repo-name]
═══════════════════════════════════════════════════════════════════

Files analyzed: [count]
Current models found: [list]

Existing Configuration:
  [If found: "✅ Preserved existing CLAUDE.md (merged embedding config)"]
  [If found: "✅ Preserved existing commands: [list]"]
  [If found: "✅ Preserved existing agents: [list]"]
  [If none:  "📝 Created new configuration from scratch"]

Generated/Updated files:
  ✅ .claude/CLAUDE.md [created/updated]
  ✅ .claude/commands/update-embedding-model.md [created/skipped if exists]
  ✅ .claude/commands/test-embedding-changes.md [created/skipped if exists]
  ✅ .claude/commands/validate-update.md [created/skipped if exists]

Key findings:
  • Model registry: [location]
  • Dimension mapping: [location or "dynamic"]
  • Test command: [command]

To add a new model, update these locations:
  1. [primary location]
  2. [secondary location]
  ...

═══════════════════════════════════════════════════════════════════
```

---

## STRICT RULES

1. **DO NOT** modify any existing source code
2. **ONLY** create/modify files in `.claude/` directory
3. Be THOROUGH - understand the full model handling flow
4. Focus on how to ADD models, not replace them
5. Note any model-specific logic that might need extension
6. **PRESERVE** existing CLAUDE.md content - merge, don't replace
7. **PRESERVE** existing commands/agents unless they're specifically for embedding models
8. **READ FIRST** - always check for existing configuration before creating new files
