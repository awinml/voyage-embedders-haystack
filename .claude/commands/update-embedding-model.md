# Add New VoyageAI Model Support

Add support for a newly released VoyageAI model (embedding OR reranker).

**Arguments:** $ARGUMENTS
(Format: `<model_id> <branch_name> [--existing-pr] [--type=embedding|rerank]`)

Parse the arguments:
- First argument: New model identifier (e.g., `voyage-3-large` or `rerank-2-lite`)
- Second argument: Default branch name (e.g., `feat/voyage-model-voyage-3-large`)
- `--existing-pr` flag: If there's an open PR to modify
- `--type=<type>`: Model type (`embedding` or `rerank`). Auto-detect if not provided:
  - If model starts with `rerank`, it's a reranker
  - Otherwise, it's an embedding model

---

## STEP 0: Determine Model Type

Auto-detect from model ID if `--type` not provided:
- `rerank-*` → reranker model
- `voyage-*`, anything else → embedding model

**Model types have different requirements:**
- **Embeddings**: Require dimension configuration
- **Rerankers**: No dimensions, but may have score normalization settings

---

## STEP 0.5: Check for Existing PR

**FIRST**, check if there's an open PR to modify:

Look for: `.claude/model-specs/open-pr-info.json`

If this file exists, READ IT! It contains:
```json
{
  "pr_url": "https://github.com/.../pull/123",
  "pr_number": 123,
  "state": "open",
  "title": "Add VoyageAI integration",
  "head_branch": "feature/voyageai-support",  // USE THIS BRANCH!
  "base_branch": "main"
}
```

**If open PR exists:**
- DO NOT create a new branch
- Checkout the `head_branch` from the PR info
- Add the new model to the existing integration code
- The PR already has VoyageAI integration, you're just adding the new model

**If no open PR:**
- Create a new branch as usual

---

## STEP 0.6: Read Model Specification

Check for model spec at `.claude/model-specs/<model_id>-spec.json`

If found, use the verified:
- Model ID (exact string)
- Dimensions (embeddings only - from actual API test)
- Model type (embedding vs rerank)
- Code examples

---

## IMPORTANT CONTEXT

This is about **ADDING** a new model, NOT replacing an existing one!
- Existing models continue to work
- We're adding support for a NEW model option
- If there's an open PR, we're adding to that integration
- Both embedding and reranker models follow this principle

---

## STRICT RULES

1. **DO NOT COMMIT** any changes
2. **DO NOT PUSH** anything
3. **DO NOT CREATE** pull requests
4. **DO NOT REMOVE** existing model support
5. Leave all changes **uncommitted** for manual review
6. **NEVER PUT API KEYS IN SOURCE CODE** - Always use environment variables!
   - Use `os.environ.get("VOYAGE_API_KEY")` in Python
   - Use `process.env.VOYAGE_API_KEY` in JavaScript/TypeScript
   - Use appropriate env var access in other languages
   - API keys in code are a **security vulnerability**

---

## STEP 1: Git Setup (ALREADY DONE)

**NOTE**: Git setup is handled automatically by the orchestrator before you start.
The branch is already checked out and ready. Just verify with `git status` and `git branch`.

First, check `.claude/model-specs/repo-info.json` to see if this repo has an upstream:
```json
{"has_upstream": true, "upstream_url": "https://github.com/..."}  // Fork - sync with upstream
{"has_upstream": false, "upstream_url": null}                      // Owned - no sync needed
```

### If there's an EXISTING PR (open-pr-info.json exists):

```bash
# 1. First, sync main with upstream (if fork) or origin
git checkout main
git fetch --all

# 2. Check repo-info.json for upstream
HAS_UPSTREAM=$(jq -r '.has_upstream' .claude/model-specs/repo-info.json)

if [ "$HAS_UPSTREAM" = "true" ]; then
    # Add upstream remote if missing
    UPSTREAM_URL=$(jq -r '.upstream_url' .claude/model-specs/repo-info.json)
    if ! git remote | grep -q '^upstream$'; then
        echo "Adding upstream remote: $UPSTREAM_URL"
        git remote add upstream "$UPSTREAM_URL"
        git fetch upstream
    fi

    # Sync main with upstream
    git pull upstream main
    git push origin main  # Keep origin's main in sync
else
    # Owned repo - just pull from origin
    git pull origin main
fi

# 3. Get the PR branch name from the JSON file
PR_BRANCH=$(jq -r '.head_branch' .claude/model-specs/open-pr-info.json)

# 4. Checkout the PR branch
git checkout $PR_BRANCH

# 5. Rebase on latest main to get any upstream updates
git rebase main

# 6. If rebase conflicts, abort and just pull latest
# git rebase --abort && git pull origin $PR_BRANCH
```

### If NO existing PR - WITH upstream (fork):

```bash
# 1. Switch to main branch FIRST
git checkout main

# 2. Check if upstream remote exists, add it if missing
UPSTREAM_URL=$(jq -r '.upstream_url' .claude/model-specs/repo-info.json)
if ! git remote | grep -q '^upstream$'; then
    echo "Adding upstream remote: $UPSTREAM_URL"
    git remote add upstream "$UPSTREAM_URL"
fi

# 3. Fetch from upstream
git fetch upstream

# 4. Update local main from upstream
git pull upstream main

# 5. Push updated main to origin (keep fork's main in sync)
git push origin main

# 6. Create feature branch from updated main
git checkout -b <branch_name>
```

### If NO existing PR - WITHOUT upstream (owned repo):

```bash
# 1. Switch to main branch FIRST (or master)
git checkout main || git checkout master

# 2. Pull latest from origin
git pull origin main || git pull origin master

# 3. Create feature branch from updated main
git checkout -b <branch_name>
```

**Report:**
- Which path was taken (existing PR / fork / owned)
- Current branch name
- Whether main was synced with upstream

---

## STEP 2: Read Project Configuration

Read `.claude/CLAUDE.md` to understand:
- How this project handles VoyageAI models (embeddings AND/OR rerankers)
- Where model configurations are defined
- Whether it supports the type of model you're adding (embedding vs rerank)
- How to add new model support

If `.claude/CLAUDE.md` doesn't exist, analyze the codebase to find:
- Where VoyageAI models are defined
- Existing model configurations (embedding and rerank separately)
- Test patterns

**IMPORTANT:** If this project doesn't support rerankers and you're adding a reranker:
- Report that rerankers aren't supported
- Exit early with a note

---

## STEP 3: Establish Test Baseline (IMPORTANT)

**Before making ANY changes**, run the existing test suite to establish a baseline.

This tells you which tests were already failing (unrelated to your work).

```bash
# Run full test suite and capture results
npm test 2>&1 | tee /tmp/baseline-tests.txt
# OR
pytest -v 2>&1 | tee /tmp/baseline-tests.txt
# OR
go test ./... -v 2>&1 | tee /tmp/baseline-tests.txt
```

**Parse the baseline**:
- Count total tests
- Note any pre-existing failures
- Record which tests failed (save the names)

**Example output to note**:
```
Baseline before changes:
- Total: 156 tests
- Passed: 153 tests
- Failed: 3 tests (test_legacy_api, test_deprecated_feature, test_external_service)
- These 3 failures existed BEFORE our changes
```

**Why this matters**: You're only responsible for VoyageAI-related tests passing, not fixing pre-existing failures in unrelated code.

---

## STEP 4: Add New Model Support

### For EMBEDDING models:

1. Add model to embedding model registry/configuration
2. Add dimension mapping (REQUIRED - get from spec file)
3. Update documentation about supported embedding models
4. Add tests if test patterns exist

### For RERANKER models:

1. Add model to reranker model registry/configuration
2. Update documentation about supported reranker models
3. Add tests if test patterns exist
4. No dimension mapping needed

### For repos WITH existing PR (integration already exists):
The VoyageAI integration code is already there. You need to:
1. Find where models are registered/listed (separate lists for embeddings vs rerankers)
2. Add the new model to the appropriate list
3. Add configuration if needed (dimensions for embeddings)
4. Update any documentation about supported models

### For repos WITHOUT existing PR (new integration):
Follow the standard process from CLAUDE.md.

**Use dimensions from the spec file for embedding models!**

---

## STEP 5: Add Tests for New Model

**CRITICAL**: You MUST add tests for the new model before completing this task.

### 4.1 Identify Existing Test Patterns

Search for existing model tests to understand the pattern:
- Unit tests for model configuration
- Integration tests for actual API calls
- Test fixtures and mocks

Example search:
```bash
# Find existing model tests
grep -r "voyage-2\|voyage-3\|rerank-" tests/ test/ __tests__/ spec/ -l 2>/dev/null | head -10
```

### 4.2 Add Unit Tests for New Model

Create tests that verify:
- ✅ New model is in the registry/configuration
- ✅ Dimensions are correct (for embeddings)
- ✅ Model name validation works
- ✅ Configuration can be loaded

**Example patterns:**
```python
def test_new_model_in_registry():
    assert "voyage-3-large" in SUPPORTED_MODELS

def test_new_model_dimensions():
    assert get_dimensions("voyage-3-large") == 1024
```

### 4.3 Add Integration Tests (MANDATORY)

**CRITICAL**: Integration tests with real API calls are REQUIRED.

Create tests that:
- ✅ Make actual API calls with the new model
- ✅ Verify response format
- ✅ Check embedding dimensions match spec
- ✅ Validate error handling

**Use the API key from environment**: `VOYAGE_API_KEY` or `EMBEDDING_API_KEY`

**Example patterns:**

For **Embedding Models**:
```python
import os
import pytest

@pytest.mark.integration
def test_voyage_3_large_embedding():
    """Integration test for voyage-3-large model"""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        pytest.skip("VOYAGE_API_KEY not set")

    client = create_client(api_key=api_key)
    result = client.embed(
        texts=["Integration test for voyage-3-large"],
        model="voyage-3-large"
    )

    # Verify response structure
    assert result.embeddings is not None
    assert len(result.embeddings) == 1

    # Verify dimensions match spec
    assert len(result.embeddings[0]) == 1024

    # Verify usage tracking
    assert result.total_tokens > 0
```

For **Reranker Models**:
```python
@pytest.mark.integration
def test_rerank_2_lite_reranking():
    """Integration test for rerank-2-lite model"""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        pytest.skip("VOYAGE_API_KEY not set")

    client = create_client(api_key=api_key)
    result = client.rerank(
        query="What is machine learning?",
        documents=["ML is a subset of AI", "The weather is nice"],
        model="rerank-2-lite"
    )

    # Verify response structure
    assert len(result.results) == 2

    # Verify relevance scores
    assert result.results[0].relevance_score > result.results[1].relevance_score
```

### 4.4 Follow Project Test Patterns

- Use the same test framework as existing tests
- Match naming conventions (test_*, *_test.py, *.spec.ts, etc.)
- Add to appropriate test file or create new one following project structure
- Include test markers if used (@pytest.mark.integration, describe blocks, etc.)

## STEP 6: Run ALL Tests

Run the complete test suite multiple times if needed:

### 5.1 Run Unit Tests
```bash
# Find and run unit tests (project-specific command from CLAUDE.md)
npm test
# OR
pytest tests/unit/
# OR
go test ./...
```

### 5.2 Run Integration Tests with API Key

**CRITICAL**: Integration tests MUST pass with the new model.

```bash
# Ensure API key is set
export VOYAGE_API_KEY="${VOYAGE_API_KEY}"

# Run integration tests
pytest tests/integration/ -v
# OR
npm run test:integration
# OR
go test -tags=integration ./...
```

### 6.3 Analyze Test Results - Compare to Baseline

**Critical**: Compare results to the baseline from STEP 3.

If tests fail:
1. **Categorize failures**:
   - **New failures**: Tests that passed in baseline but fail now (YOUR responsibility)
   - **Pre-existing failures**: Tests that failed in baseline (NOT your responsibility)
   - **VoyageAI-related failures**: Any test mentioning voyage, embedding, rerank (YOUR responsibility)

2. **Focus on YOUR failures**:
   ```bash
   # Example: If baseline had 3 failures and now you have 5
   # You need to fix the 2 NEW failures
   # The 3 pre-existing failures can remain
   ```

3. **Identify root cause** (for NEW failures only):
   - Dimension mismatch?
   - Model name typo?
   - Missing configuration?
   - API key issue?

4. **Fix YOUR issues**
5. **Re-run tests**
6. **Repeat until all NEW and VoyageAI tests pass**

**DO NOT** spend time fixing pre-existing failures in unrelated code.

### 6.4 Run Tests Again to Confirm

After fixing issues, run the full suite one more time:
```bash
# Full test suite
npm test -- --coverage
# OR
pytest --cov=. -v
# OR
go test -v -race ./...
```

**Target**: All VoyageAI-related tests passing:
- ✅ No NEW failures introduced (regression check)
- ✅ New unit tests for the model pass
- ✅ New integration tests with real API calls pass
- ✅ Existing VoyageAI tests still pass
- ⚠️ Pre-existing unrelated failures are acceptable (document them)

---

## STEP 7: Summary Report

```
═══════════════════════════════════════════════════════════════════
✨ New Model Added: [repo-name]
═══════════════════════════════════════════════════════════════════

Model Type: [EMBEDDING / RERANKER]
Mode: [EXISTING PR / NEW BRANCH (fork) / NEW BRANCH (owned)]
Branch: <current branch>
New Model: <model_id>
Dimensions: <from spec> (embeddings only)

Repo Type: [Fork of <upstream_url> / Owned repo]

PR Info (if applicable):
  • PR: #<number> - <title>
  • URL: <pr_url>
  • Status: <state>

Changes made:
  ➕ path/to/file - Description
  ...

Tests Added:
  ✅ Unit tests for model registry/config
  ✅ Integration tests with real API calls (MANDATORY)

Test Results (Baseline Comparison):
  Baseline: X total, Y passed, Z failed
  Current:  X total, Y passed, Z failed

  ✅ New model unit tests: X/X passed
  ✅ New model integration tests: X/X passed
  ✅ Existing VoyageAI tests: X/X passed
  ✅ No new failures introduced

  ⚠️ Pre-existing failures (NOT caused by this change):
    - test_name_1 (failed in baseline)
    - test_name_2 (failed in baseline)

  Overall: ✅ VoyageAI integration PASSING

═══════════════════════════════════════════════════════════════════

⚠️  CHANGES ARE NOT COMMITTED

Next steps:
  1. Review changes: git diff
  2. If this is an existing PR:
     - Commit: git add -A && git commit -m "feat: add <model> support"
     - Push: git push origin <branch>
     - The existing PR will be updated
  3. If this is a new branch (fork):
     - Commit and create PR to upstream
  4. If this is a new branch (owned repo):
     - Commit and push to origin
     - Create PR or merge directly (your choice)

═══════════════════════════════════════════════════════════════════
```

---

## REMINDER

**DO NOT COMMIT OR PUSH** - All changes must be reviewed manually first!
**DO NOT REMOVE** existing model support!
**CHECK FOR EXISTING PR** - Use that branch if it exists!
**CHECK MODEL TYPE** - Embeddings need dimensions, rerankers don't!

---

## COMPLETION CRITERIA

**CRITICAL**: VoyageAI-related tests MUST pass before completion!

When ALL of the following are complete:
- ✅ **Test baseline established** (ran tests before making changes)
- ✅ Git setup verified (correct branch checked out)
- ✅ Model type determined (embedding/reranker)
- ✅ Project configuration read and understood
- ✅ New model support added (to appropriate registry/config)
- ✅ Documentation updated (if applicable)
- ✅ **NEW TESTS ADDED**:
  - ✅ Unit tests for model configuration
  - ✅ **Integration tests with real API calls (MANDATORY)**
- ✅ **VOYAGEAI TESTS PASSING**:
  - ✅ **No NEW failures introduced** (compared to baseline)
  - ✅ New unit tests pass
  - ✅ **New integration tests pass with actual API calls**
  - ✅ Existing VoyageAI/embedding/rerank tests still pass
- ✅ Summary report with baseline comparison
- ✅ No compilation/syntax errors
- ✅ Existing models still present and unchanged

Output exactly this line:
```
<promise>ORCHESTRATOR_TASK_COMPLETE</promise>
```

**CRITICAL RULES**:
1. **NEVER output the promise if VoyageAI tests are failing** - Fix them first!
2. **Pre-existing failures in unrelated tests are OK** - Document them in the report
3. **Integration tests are MANDATORY** - Unit tests alone are not enough
4. **Don't modify unrelated code** to fix pre-existing failures
5. **Focus on VoyageAI integration** - That's your responsibility
6. Ralph will iterate until YOUR tests pass - use this to fix issues!

**Acceptable scenarios for completion**:
- ✅ New model tests pass, 3 unrelated tests still failing (were failing in baseline)
- ✅ New model tests pass, 0 new failures, 100% pass rate
- ❌ New model tests fail - MUST fix before completion
- ❌ Introduced new failures in existing tests - MUST fix before completion

**This is the PURPOSE of Ralph**: Iterate until the integration with the new model is fully tested and working, without breaking existing functionality!
