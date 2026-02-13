# Test Embedding Changes

**PURPOSE**: Comprehensively test the new model integration with both unit and integration tests.

**CRITICAL**: This command focuses on running ALL tests and ensuring they PASS. Ralph will iterate until tests succeed.

---

## STEP 1: Load Project Configuration

Read `.claude/CLAUDE.md` to find:
- Test framework used (pytest, jest, go test, etc.)
- Unit test commands
- Integration test commands
- Required environment variables for testing
- Test file locations and patterns

---

## STEP 2: Verify Environment Setup

### 2.1 Check API Keys

**CRITICAL**: API keys are required for integration tests.

```bash
# Check for API keys
echo "VOYAGE_API_KEY: ${VOYAGE_API_KEY:+SET}"
echo "EMBEDDING_API_KEY: ${EMBEDDING_API_KEY:+SET}"
```

If keys are missing:
- Document which are missing
- Integration tests will be skipped (pytest.skip, jest.skip, etc.)
- Still create the test code

### 2.2 Verify Test Dependencies

Check test framework is available:
```bash
# Python
python -m pytest --version 2>/dev/null || pip install pytest

# Node.js
npm test -- --version 2>/dev/null || npm install

# Go
go test -h >/dev/null 2>&1 || echo "Go test available"
```

---

## STEP 3: Load Test Baseline

**CRITICAL**: Check if a baseline exists from the update command.

```bash
# Check for baseline file
if [ -f /tmp/baseline-tests.txt ]; then
    echo "✅ Baseline found - will compare results"
    grep -E "passed|failed|error" /tmp/baseline-tests.txt | tail -5
else
    echo "⚠️ No baseline - will establish one now"
    # Run tests to create baseline (if update command didn't)
fi
```

**Parse baseline** (if exists):
- Total tests in baseline
- Number of failures in baseline
- Names of failing tests in baseline

**Why this matters**: You only need to ensure VoyageAI tests pass and no NEW failures are introduced.

---

## STEP 4: Run Unit Tests

### 4.1 Identify Unit Tests

Find unit test files:
```bash
# Common patterns
find . -name "*test*.py" -o -name "*_test.go" -o -name "*.test.ts" 2>/dev/null | grep -v node_modules | head -20
```

### 4.2 Run Unit Tests

Execute unit tests (use command from CLAUDE.md):

```bash
# Python
pytest tests/unit/ -v --tb=short

# Node.js
npm run test:unit

# Go
go test ./... -short

# Other
make test-unit
```

### 4.3 Analyze Unit Test Results - Compare to Baseline

Parse output and compare:
- **Total tests**: Count (vs baseline)
- **Passed**: Count (vs baseline)
- **Failed**: List each failure

**Categorize failures**:
1. **Pre-existing**: Failed in baseline too (NOT your problem)
2. **New failures**: Passed in baseline, fail now (YOUR problem - FIX THESE)
3. **VoyageAI-related**: Any test with voyage/embed/rerank in name (YOUR problem if failing)

**Example analysis**:
```
Baseline: 150 tests, 147 passed, 3 failed
Current:  151 tests, 147 passed, 4 failed

New test added: test_voyage_3_large_dimensions (passing ✅)
Pre-existing failures (still failing, OK):
  - test_legacy_api_v1
  - test_deprecated_feature
  - test_external_service_timeout

→ Result: Good! New test passes, no new failures introduced
```

---

## STEP 5: Run Integration Tests

**CRITICAL**: Integration tests with real API calls are MANDATORY.

### 4.1 Identify Integration Tests

Find integration test files:
```bash
# Search for integration test markers
grep -r "@pytest.mark.integration\|describe.*integration\|//.*integration" tests/ test/ -l 2>/dev/null
```

### 4.2 Set Environment for Integration Tests

```bash
# Export API key
export VOYAGE_API_KEY="${VOYAGE_API_KEY}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY}"

# Verify it's set
echo "API key length: ${#VOYAGE_API_KEY}"
```

### 4.3 Run Integration Tests

Execute integration tests:

```bash
# Python - run integration tests specifically
pytest tests/integration/ -v --tb=short -m integration
# OR
pytest -v --tb=short -m "not skip and integration"

# Node.js
npm run test:integration
# OR
npm test -- --testPathPattern=integration

# Go
go test -tags=integration ./... -v

# Other
make test-integration
```

### 4.4 Analyze Integration Test Results

**CRITICAL**: Every integration test must pass.

Parse output:
- **Total integration tests**: Count
- **Passed**: Count (target: 100%)
- **Failed**: List each with full stack trace
- **Skipped**: Count (acceptable if no API key)

If failures:
1. **Read the full error message**
2. **Identify the issue**:
   - API connection error?
   - Dimension mismatch?
   - Model not found?
   - Invalid response format?
3. **Note the test name and line number**

---

## STEP 6: Verify New Model Tests Exist

**CRITICAL**: New model MUST have dedicated tests.

### 5.1 Search for New Model in Tests

```bash
# Search for the new model ID in test files
NEW_MODEL="voyage-3-large"  # Replace with actual model
grep -r "$NEW_MODEL" tests/ test/ __tests__/ spec/ 2>/dev/null
```

**Expected findings**:
- ✅ Model appears in at least ONE integration test
- ✅ Model appears in at least ONE unit test
- ✅ Test includes actual API call or mock

If NOT found:
- **FAIL**: Tests are missing for the new model
- Document this critical issue
- Return to STEP 4 of update-embedding-model.md to add tests

### 5.2 Verify Test Quality

For each test found, check:
- ✅ Makes actual API call (integration) or tests config (unit)
- ✅ Verifies response structure
- ✅ Checks dimensions (for embeddings)
- ✅ Has proper assertions
- ✅ Handles errors appropriately

---

## STEP 7: Run Full Test Suite

**Final verification**: Run EVERYTHING.

```bash
# Python - full suite with coverage
pytest --cov=. --cov-report=term-missing -v

# Node.js - full suite
npm test -- --coverage --verbose

# Go - full suite with race detection
go test -v -race -coverprofile=coverage.out ./...

# Other
make test-all
```

**Target metrics**:
- ✅ 100% of tests passing
- ✅ No regressions (existing tests still pass)
- ✅ New model covered by tests
- ✅ Integration tests pass with real API

---

## STEP 8: Fix Failures (Critical Step)

If VoyageAI-related tests fail, you MUST fix them before completing.

### 8.1 Categorize Failures - Focus on What Matters

Group failures by priority:
1. **New model test failures** - MUST FIX (your new tests)
2. **VoyageAI regression failures** - MUST FIX (broke existing VoyageAI tests)
3. **New unrelated failures** - MUST FIX (you broke something)
4. **Pre-existing unrelated failures** - IGNORE (not your problem)
5. **Environment issues** - Document (missing API keys, etc.)

### 8.2 Fix Each Failure (Only YOUR Failures)

**Important**: Only fix failures in categories 1-3 above. Ignore pre-existing failures.

For each failed test you must fix:

1. **Understand the failure**:
   ```bash
   # Re-run single failing test with verbose output
   pytest tests/test_models.py::test_voyage_3_large -vv --tb=long
   ```

2. **Identify root cause**:
   - Read the assertion error
   - Check the expected vs actual values
   - Look at the stack trace

3. **Common issues and fixes**:

   **Dimension mismatch**:
   ```python
   # Check spec file for correct dimensions
   # Update dimension constant/config
   VOYAGE_3_LARGE_DIMS = 1024  # From spec
   ```

   **Model not in registry**:
   ```python
   # Add to SUPPORTED_MODELS list
   SUPPORTED_MODELS = ["voyage-2", "voyage-3", "voyage-3-large"]
   ```

   **API call fails**:
   ```python
   # Verify model ID matches API exactly
   # Check API key is valid
   # Ensure network connectivity
   ```

   **Type errors**:
   ```typescript
   // Add model to type definition
   type VoyageModel = "voyage-2" | "voyage-3" | "voyage-3-large";
   ```

4. **Re-run the test**:
   ```bash
   # Verify fix works
   pytest tests/test_models.py::test_voyage_3_large -v
   ```

5. **If still failing, iterate**:
   - Try a different approach
   - Check project patterns again
   - Read similar working tests

### 8.3 Re-run Full Suite and Compare

After fixing YOUR issues:
```bash
# Confirm VoyageAI tests pass
pytest -v
```

**Success criteria**:
- ✅ All NEW tests passing
- ✅ All VoyageAI/embedding/rerank tests passing
- ✅ No new failures compared to baseline
- ⚠️ Pre-existing failures can remain (document them)

---

## STEP 9: Report Results with Baseline Comparison

```
═══════════════════════════════════════════════════════════════════
🧪 Test Results: [repo-name]
═══════════════════════════════════════════════════════════════════

Baseline Comparison:
  Baseline:  X total, Y passed, Z failed
  Current:   X total, Y passed, Z failed
  Change:    +N new tests, ±0 failures

Unit Tests:           ✅ PASSED (X/X tests)
Integration Tests:    ✅ PASSED (X/X tests) - WITH REAL API CALLS
VoyageAI Tests:       ✅ ALL PASSING

New Model Test Coverage:
  ✅ Unit test: test_voyage_3_large_in_registry
  ✅ Unit test: test_voyage_3_large_dimensions
  ✅ Integration test: test_voyage_3_large_embedding
  ✅ Integration test: test_voyage_3_large_error_handling

New Failures Introduced: ✅ NONE

Pre-existing Failures (NOT caused by this change):
  ⚠️ test_legacy_api_v1 (failed in baseline)
  ⚠️ test_external_service (failed in baseline)

Test Execution Summary:
  • Tests added: X
  • Unit tests run: X times
  • Integration tests run: X times
  • VoyageAI failures fixed: X
  • Final result: VoyageAI INTEGRATION PASSING ✅

Coverage (if available):
  • Overall: X%
  • New model code: X%

═══════════════════════════════════════════════════════════════════

Overall Status: ✅ READY FOR MERGE

Note: X pre-existing failures remain (unrelated to VoyageAI)

Next Steps:
  1. Review changes: git diff
  2. Commit: git add -A && git commit -m "test: add tests for voyage-3-large"
  3. Run tests one final time before pushing

═══════════════════════════════════════════════════════════════════
```

---

## COMPLETION CRITERIA

**CRITICAL**: VoyageAI-related tests MUST be passing before completion!

When ALL of the following are complete:
- ✅ Baseline loaded (or established if missing)
- ✅ Project configuration loaded
- ✅ Environment variables checked
- ✅ Unit tests executed and **VoyageAI tests PASSING**
- ✅ Integration tests executed and **VoyageAI tests PASSING**
- ✅ **New model has dedicated tests**:
  - ✅ At least one unit test
  - ✅ At least one integration test with real API call
- ✅ Full test suite executed
- ✅ **No new failures introduced** (compared to baseline)
- ✅ All VoyageAI/embedding/rerank tests passing
- ✅ Test results documented with baseline comparison

Output exactly this line:
```
<promise>ORCHESTRATOR_TASK_COMPLETE</promise>
```

**CRITICAL RULES**:
1. **NEVER output the promise if VoyageAI tests are failing**
2. **Pre-existing failures in unrelated code are OK** - Document them
3. **Integration tests are MANDATORY** - must include real API calls
4. **New model MUST have tests** - both unit and integration
5. **Don't fix unrelated code** - Focus on VoyageAI integration
6. If VoyageAI tests fail, **FIX THEM** - that's why Ralph exists!
7. Run tests multiple times if needed - iterate until VoyageAI tests pass

**Acceptable scenarios**:
- ✅ VoyageAI tests pass, 3 unrelated tests failing (were in baseline)
- ✅ VoyageAI tests pass, 0 failures total
- ❌ VoyageAI test fails - MUST fix
- ❌ New failure in unrelated test (not in baseline) - MUST fix

**Ralph's Purpose**: Keep iterating, fixing VoyageAI tests, and re-running until the integration works perfectly!
