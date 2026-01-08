# Plans Folder

## MANDATORY PLANNING

**BEFORE creating any new .py file, you MUST create a plan.**

This is NOT optional. This is NOT "for complex tasks only." If you're about to write a new file, stop and plan first.

---

## When Plan is REQUIRED

- Creating ANY new .py file
- Task has 🗺️ marker in CONTINUITY.md
- Adding a new module or feature
- Any work that will take more than 15 minutes

## When Plan is NOT Required

- Editing existing files only
- Running tests, deploys, commits
- Updating documentation

---

## How to Create a Plan

```bash
# 1. Copy template
cp .plans/_TEMPLATE.md .plans/$(date +%Y-%m-%d)_feature-name.md

# 2. Fill out these sections:
#    - Goal: What are we building? (1-2 sentences)
#    - Approach: How will we build it?
#    - Implementation Steps: Ordered checklist
#    - Files to Modify: List of files to create/change

# 3. THEN start coding
```

---

## Why This Matters

Your context WILL be compacted. When that happens:

- Your working memory disappears
- Your brilliant design ideas disappear
- Your understanding of edge cases disappears

But your plan file survives. Future-you can read it and continue.

**Plans are your lifeline across context compaction.**

---

## Naming Convention

```
YYYY-MM-DD_short-description.md
```

Examples:
- `2026-01-08_netcdf-support.md`
- `2026-01-08_astronomy-units.md`
- `2026-01-09_parquet-support.md`

---

## After Completion

Keep completed plans for reference. Mark status as "COMPLETED" at the top.
