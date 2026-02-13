#!/usr/bin/env python3
"""
VoyageAI Orchestrator Event Hook
Fires on every Claude Code tool execution (PostToolUse event)
POSTs event data to dashboard server

Note: This script can run with system Python or venv Python.
It will work as long as the 'requests' module is available.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Main hook entry point"""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)

        # Get dashboard URL from environment
        dashboard_url = os.environ.get("DASHBOARD_URL", "")
        if not dashboard_url:
            sys.exit(0)  # Silent exit if no dashboard

        # Extract tool information
        tool_name = hook_input.get("tool", {}).get("name", "Unknown")
        description = hook_input.get("tool", {}).get("description", "")

        # Get repo path (current working directory)
        repo_path = os.getcwd()

        # Detect Ralph iteration (if running in Ralph loop)
        ralph_iteration = detect_ralph_iteration()

        # Extract token usage
        usage = hook_input.get("usage", {})
        tokens = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_creation": usage.get("cache_creation_input_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0)
        }

        # Extract model information
        model = hook_input.get("model", "unknown")

        # Build event payload
        event = {
            "event_type": "ToolExecuted",
            "repo_path": repo_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": {
                "tool_name": tool_name,
                "description": description[:100],  # Truncate long descriptions
                "ralph_iteration": ralph_iteration,
                "tokens": tokens,
                "model": model
            }
        }

        # POST to dashboard (fast timeout, no retry)
        post_event(dashboard_url, event)

    except Exception:
        pass  # Never fail, never log (would pollute Claude output)
    finally:
        sys.exit(0)  # ALWAYS exit 0


def detect_ralph_iteration() -> int:
    """
    Detect current Ralph iteration by checking:
    1. RALPH_ITERATION environment variable (if Ralph sets it)
    2. .claude/ralph_state.json file (if exists)
    3. Count of tool executions in current session (fallback)
    """
    # Option 1: Environment variable
    if "RALPH_ITERATION" in os.environ:
        try:
            return int(os.environ["RALPH_ITERATION"])
        except ValueError:
            pass

    # Option 2: State file
    state_file = Path(".claude/ralph_state.json")
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            return state.get("iteration", 0)
        except:
            pass

    # Option 3: Fallback (return 0 if unknown)
    return 0


def post_event(dashboard_url: str, event: dict):
    """POST event to dashboard with fast timeout"""
    try:
        import requests
        # Remove trailing slash if present
        base_url = dashboard_url.rstrip('/')
        requests.post(
            f"{base_url}/event",
            json=event,
            timeout=0.5  # 500ms max
        )
    except:
        pass  # Silent fail on network error or timeout


if __name__ == "__main__":
    main()
