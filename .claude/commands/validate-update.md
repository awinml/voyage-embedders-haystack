# Validate New Model Addition

Validate that the new embedding model was added correctly and existing models still work.

---

## STEP 1: Load Configuration

Read `.claude/CLAUDE.md` to get:
- All locations where the new model should be added
- Existing models that should still work
- Expected model registry format

---

## STEP 2: Verify New Model Added

For each location listed in CLAUDE.md:
1. Read the file
2. Verify the new model was added
3. Check it follows the same pattern as existing models
4. Report status

---

## STEP 3: Verify Existing Models Preserved

Check that:
- All previously existing models are still present
- No models were accidentally removed or modified
- Existing model configurations unchanged

---

## STEP 4: Verify Test Coverage for New Model

**CRITICAL**: The new model MUST have tests.

### 4.1 Search for Model in Test Files

```bash
# Replace NEW_MODEL with actual model ID
NEW_MODEL="voyage-3-large"
grep -r "$NEW_MODEL" tests/ test/ __tests__/ spec/ 2>/dev/null
```

Expected findings:
- ✅ At least ONE unit test mentioning the model
- ✅ At least ONE integration test with real API call
- ✅ Tests verify dimensions (embeddings) or scores (rerankers)

### 4.2 Verify Test Quality

For each test found, check it includes:
- ✅ Proper test framework markers (@pytest.mark, describe, etc.)
- ✅ API call with the new model ID
- ✅ Response validation (structure, dimensions, etc.)
- ✅ Error handling test cases
- ✅ Assertions that would fail if model doesn't work

### 4.3 Check Test Execution

Verify tests can run:
```bash
# Try running one of the new model's tests
pytest -k "voyage_3_large" -v
# OR
npm test -- --testNamePattern="voyage.*3.*large"
```

**If no tests found**:
- ❌ **CRITICAL FAILURE**: New model lacks tests
- Document this issue prominently
- Validation should FAIL

## STEP 5: Consistency Check

Verify:
- New model follows naming conventions
- Dimension mapping is correct (if applicable)
- Documentation matches code
- **Tests exist and cover new model (verified in STEP 4)**

---

## STEP 6: Search for Issues

Search for:
- Hardcoded model references that might need updating
- "TODO" comments about model support
- Any model-specific code that might need new branch

---

## STEP 7: Validation Report

```
═══════════════════════════════════════════════════════════════════
✅ Validation Report: [repo-name]
═══════════════════════════════════════════════════════════════════

New Model Addition:
  [x] Added to model registry
  [x] Added to configuration
  [x] Added to documentation
  [x] **Test coverage added (CRITICAL)**:
      [x] Unit tests exist
      [x] Integration tests exist with real API calls
      [x] Tests are executable and well-formed

Existing Models (Regression Check):
  [x] voyage-2 - Still present ✓
  [x] voyage-large-2 - Still present ✓
  ...

Consistency:
  [x] Naming follows convention
  [x] Dimension mapping correct
  [x] Documentation matches code

Potential Issues:
  [none / list any concerns]

Modified Files:
  M path/to/config.yaml
  M src/models.ts
  M docs/models.md
  ...

═══════════════════════════════════════════════════════════════════

Overall: ✅ VALIDATED / ⚠️ WARNINGS / ❌ ISSUES FOUND

═══════════════════════════════════════════════════════════════════
```

---

## COMPLETION CRITERIA

When ALL of the following are complete:
- ✅ Configuration loaded
- ✅ New model addition verified in all expected locations
- ✅ **Test coverage verified (MANDATORY)**:
  - ✅ Unit tests found for new model
  - ✅ Integration tests found with real API calls
  - ✅ Tests are well-formed and executable
- ✅ Existing models preservation confirmed
- ✅ Consistency checks passed
- ✅ No critical issues found
- ✅ Validation report provided with test coverage details

Output exactly this line:
```
<promise>ORCHESTRATOR_TASK_COMPLETE</promise>
```

**CRITICAL**:
- **NEVER output the promise if tests are missing** for the new model
- Integration tests are MANDATORY - not optional
- Warnings are acceptable, but missing tests is a CRITICAL failure
- If tests are missing, document it and DO NOT output the promise
