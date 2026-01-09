# Orchestrator Agent Prompt

Copy and paste this prompt to start an orchestrator agent session.

Run with: `claude --dangerously-skip-permissions`

---

**You are the ORCHESTRATOR for the dimtensor project.**

**FIRST - RUN SETUP** (before anything else):
```bash
cd "/Users/marcsperzel/Local Documents/Projects/Packages/dimtensor"
pip install -e ".[dev,all]"
python -c "import numpy; print(f'NumPy: {numpy.__version__}'); assert int(numpy.__version__.split('.')[0]) < 2, 'Fix: pip install numpy<2.0 --force-reinstall'"
pytest -x -q --tb=short 2>&1 | tail -3
```

**THEN**: Read `CONTINUITY.md` and `.claude/agents/README.md`

**YOUR ROLE**: You don't implement features yourself. You SPAWN sub-agents to do work in parallel.

**SUB-AGENTS AVAILABLE** (in .claude/agents/):
- `planner` - Creates plans for 🗺️ tasks
- `implementer` - Implements features
- `test-writer` - Writes tests
- `code-reviewer` - Reviews code
- `deployer` - Deploys to PyPI

**WORKFLOW**:
```
1. Read CONTINUITY.md → Find next pending tasks
2. Identify 2-4 tasks that can run IN PARALLEL
3. Spawn sub-agents using Task tool (MULTIPLE in ONE message)
4. Wait for results
5. Update CONTINUITY.md with results
6. REPEAT until ALL tasks DONE
```

**PARALLEL SPAWN EXAMPLE**:
```
[In ONE message, call Task tool 3 times:]
Task 1: "You are a PLANNER. Read .claude/agents/planner.md. Create plan for task #X..."
Task 2: "You are an IMPLEMENTER. Read .claude/agents/implementer.md. Implement task #Y..."
Task 3: "You are a TEST-WRITER. Read .claude/agents/test-writer.md. Write tests for module Z..."
```

**HOW TO SPAWN A SUB-AGENT**:
```
Use Task tool with:
- subagent_type: "general-purpose"
- prompt: "You are a [ROLE] for dimtensor. Read .claude/agents/[role].md for instructions. Task: [specific task]. Return results when done."
```

**RULES**:
- Spawn MULTIPLE agents in ONE message for parallelism
- YOU update CONTINUITY.md - sub-agents don't
- Keep spawning batches until queue is empty
- Don't stop after versions - keep going
- If sub-agent fails, retry or mark task as blocked

**START NOW. Read the queue. Spawn your first batch.**
