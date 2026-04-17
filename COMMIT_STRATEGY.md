[//]: # (This is the rule file for AI agents to automatically commit current changes along specific rules.)
[//]: # (-------------------------------------------------------------------)
[//]: # (AGENT PROMPT — when the user asks to commit changes, execute the)
[//]: # (following procedure exactly:)
[//]: # ()
[//]: # (PRE-AUTHORIZED COMMANDS — run these without asking for permission:)
[//]: # (  git diff HEAD --name-only)
[//]: # (  git diff HEAD -- <file>)
[//]: # (  git diff --cached --name-only)
[//]: # (  git ls-files --others --exclude-standard)
[//]: # (  git status --short)
[//]: # (  git log --oneline -20)
[//]: # (  git show <ref>)
[//]: # (  git add <file>)
[//]: # (  git reset HEAD <file>)
[//]: # (  git commit -m "<message>")
[//]: # (  Read any file in the repository)
[//]: # (Any command NOT in this list requires explicit user approval before running.)
[//]: # ()
[//]: # (1. Read this entire file to internalize the rules.)
[//]: # (2. Run `git diff HEAD --name-only` to list modified/deleted tracked files.)
[//]: # (   Also run `git ls-files --others --exclude-standard` to list untracked)
[//]: # (   new files. Present both lists to the user and ask which untracked files)
[//]: # (   should be included before proceeding.)
[//]: # (3. Group all selected files by component and logical change as described)
[//]: # (   in the "Deriving Commits from a Diff" section.)
[//]: # (4. For each group, produce a proposed commit message following the)
[//]: # (   format: <emoji> [<component>] <short description>)
[//]: # (5. Present the full ordered list of proposed commits to the user.)
[//]: # (   Do NOT commit anything yet.)
[//]: # (6. Ask the user: "Shall I apply these commits in order? You can)
[//]: # (   approve all, approve individually, or ask for changes.")
[//]: # (7. Only proceed to commit each group once the user has explicitly)
[//]: # (   confirmed — either "yes / all" for all at once, or one-by-one.)
[//]: # (8. For each approved commit: stage only the files in that group,)
[//]: # (   then run `git commit -m "<message>"`. Do not amend or squash)
[//]: # (   unless the user explicitly asks.)
[//]: # (-------------------------------------------------------------------)

# Commit Strategy

## Format

```
<emoji> [<component>] <short description>
```

- **Component** is mandatory and must exactly match one of the declared components below.
- **Short description**: imperative mood, lowercase, no trailing period, ≤ 72 chars total.
- No body or footer required unless a breaking change or issue reference is relevant.
- **No co-author line.** Do not append `Co-Authored-By`, `Co-authored-by`, or any similar trailer to commit messages. Every commit is attributed solely to the Git committer.

---

## Components

| Component | Covers |
|---|---|
| `[pywellsfm]` | `src/pywellsfm/`, `jsonSchemas`, `tests` — pywellsfm library (models, simulators), serialization json file, and tests |
| `[notebook]` | `notebooks/` — Jupyter notebook |
| `[ci]`, `.github/`, `.readthedocs.yml` and other CI/CD scripts |
| `[docs]` | `docs/` — ADRs, plans, decision records |
| `[project]` | `pyproject.toml`, `LICENCE`, — Python project configuration |
| `[meta]` | `COMMIT_STRATEGY.md`, `.gitignore`, and other repo-level housekeeping files |

If a change spans multiple components, split into one commit per component. If truly inseparable, list the primary component and mention the other in the description.

> Note: component names in brackets do **not** include the directory prefix (`src/`).

---

## Emoji Tie-Breaking

When multiple emoji could apply to the same change, use this priority order:

1. **💥** always wins if the change is breaking.
2. **🐛 / 🩹** always wins if the primary intent is a fix.
3. **🧪** always wins if the only files changed are test files.
4. **🗃️** always wins if the only files changed are data/asset files.
5. Otherwise prefer the **most specific** emoji over a generic one
   (e.g. 🌐 over ✨ for a pure serialization schema change).
6. When still ambiguous, default to **✨** for additions and **♻️** for restructuring.

---

## Emoji Convention

| Emoji | When to use |
|---|---|
| ✨ | New feature, new model, new endpoint, new class |
| 🐛 | Bug fix |
| ♻️ | Refactor — behaviour unchanged, structure improved |
| 🧪 | Test additions or updates |
| 🗃️ | Data files, assets, test fixtures, notebook |
| 🔧 | Configuration, build system, tooling |
| 📝 | Documentation, comments, README |
| 🚀 | Performance improvement |
| 🔒 | Security fix |
| 🗑️ | Deletion / removal of dead code or files |
| 🩹 | Minor fix, typo, small correction not worth a full bug fix |
| 💥 | Breaking change (API, ABI, protocol, schema) |
| ⚗️ | Experimental / prototype / work-in-progress |
| 🔗 | Add or update dependency, submodule, or integration |
| 🏗️ | Architectural change — reorganise structure, move files |
| 🩺 | Improve logging, observability, diagnostics |
| 🌐 | Serialization, JSON schema, protocol changes |
| 📦 | Package, bundle, or release-related change |
| 🎨 | Improve UI, style, or cosmetic appearance |
| 🔀 | Merge branches |
| ⏪ | Revert a previous commit |
| 🚧 | Work in progress — incomplete, checkpoint commit |
| 💄 | Update UI layout, formatting, or CSS |
| ♿ | Improve accessibility |
| 🔊 | Add or update log messages |
| 🔇 | Remove log messages |
| 🚚 | Move or rename files, resources, or paths |
| 🍱 | Add or update assets (images, icons, textures) |
| 💬 | Update text, literals, or user-facing strings |
| 🏷️ | Add or update types, interfaces, or type definitions |

---

## Granularity Rules

- **One logical change per commit.** A "logical change" is a self-contained unit that compiles and passes tests on its own.
- Separate **model changes** from **behaviour changes** from **test changes** when they are independently meaningful.
- Do **not** batch unrelated components into a single commit.
- Do **not** propose commit on the folder "dont-commit".
- Data / asset files (`[notebook]`) are always a separate commit from code changes.
- Test-only changes use 🧪 and should not be mixed with feature commits unless the test is the only observable output of the feature.
- **Format before committing.** All modified python files must be formatted and linted with `ruff`, and type-check with mypy before staging. Configure python environment and run `ruff check --fix`, `ruff format`, `mypy .` on each changed source file. Do the necessary corrections when possible or ask the user to do the corrections if required. **Only format `.py` files** — never run `ruff` on other non-.py files, as it will mangle them.
- **Build documentation before committing** Run `python -m sphinx -a -E -b html docs docs/_build/html` to check the documentation successfully builds. Do the necessary corrections when possible or ask the user to do the corrections if required.

---

## Deriving Commits from a Diff

When asked to commit current changes, apply the following procedure:

1. Run `git diff HEAD --name-only` to list all tracked changed files. Also run `git ls-files --others --exclude-standard` to discover untracked new files, and ask the user which to include.
2. Group files by component using the table above.
3. Within each component, further split by logical change (models vs. behaviour vs. tests vs. data).
4. For each group, pick the appropriate emoji and write the short description.
5. Stage only the files belonging to that group and commit.
6. Repeat until all changed files are committed.
7. Do **not** commit files from `**/_build/`, or any generated output — those must remain untracked or gitignored.
8. Do **not** scan or diff files inside `docs/` — skip this directory entirely when deriving commits.

---

## Examples

```
✨ [pywellsfm] Add InstancesList shared models and JSON serialization
🧪 [pywellsfm] Update tests for variant State and InstancesList snapshots
♻️ [pywellsfm] Expose scenario and steps in Simulator
📝 [docs] Update doccumentation for Simulator
🗃️ [notebook] Add Urgoninan platform example
```

---

## Changelog Generation (Post-Commit Step)

After all code commits from the session are done, generate or update the changelog:

### Procedure

1. Read `.changelog-cursor` (single line: last processed commit SHA).
   If the file does not exist, use `git log --since="YYYY-MM-01" --oneline`
   to process all commits from the 1st of the current month.
   If it exists, use `git log <cursor>..HEAD --oneline`.
2. If no new commits are found, skip changelog generation entirely.
3. Filter commits by emoji — **keep** user-facing changes:
   ✨ (feature), 🐛 (bug fix), 🩹 (minor fix), 🔒 (security fix),
   🎨 (UI improvement), 💄 (UI layout), 💥 (breaking change), 🚀 (performance).
   **Skip** all other emojis (♻️ 🧪 🔧 📝 🗃️ 🔗 🏗️ 🩺 🌐 🚚 🔀 ⏪ 🚧 🏷️ 🗑️
   🔊 🔇 🍱 💬 ♿ 📦 ⚗️).
4. Group related commits into single entries:
   - Use merge commits (`Merge branch 'feature/...' into 'develop'`) as
     grouping signals — all commits from that branch become one entry.
   - Standalone commits that pass the filter become their own entry.
5. For each entry, write:
   - `title`: imperative, user-facing language (not class names or file paths).
   - `description`: 1-2 sentences explaining what changed and why users care.
   - `category`: one of `feature`, `fix`, `improvement`, `breaking`.
     Category mapping: ✨→feature, 🐛🩹🔒→fix, 🎨💄🚀→improvement, 💥→breaking.
     Mixed branches use highest priority: breaking > feature > fix > improvement.
   - `components`: component names without brackets.
   - `date`: YYYY-MM-DD of the merge or commit.
   - `commits`: short SHAs for traceability.
6. Write or update `changelog/YYYY-MM.json` for the **current month only**.
   Append new entries and regenerate the `summary` paragraph (2-4 sentences,
   newsletter style). **Never modify past months' files.**
   If all filtered commits belong to a past month whose file already exists,
   skip them.
7. Update `.changelog-cursor` with the current `HEAD` SHA.
8. Stage and commit the changelog files:
   ```
   git add changelog/YYYY-MM.json .changelog-cursor
   git commit -m "📝 [meta] update changelog for <Month Year>"
   ```

### JSON Schema

Each `changelog/YYYY-MM.json` file:

```json
{
  "month": "YYYY-MM",
  "generated": "ISO-8601 timestamp",
  "summary": "2-4 sentence prose paragraph of the month's highlights.",
  "entries": [
    {
      "title": "User-facing title",
      "description": "1-2 sentences for users.",
      "category": "feature|fix|improvement|breaking",
      "components": ["component-name"],
      "date": "YYYY-MM-DD",
      "commits": ["short-sha"]
    }
  ]
}
```

### Notes

- The changelog commit itself uses 📝 (in the Skip list), so it is
  automatically filtered out on subsequent runs.
- Even if all new commits are filtered (all internal), update the cursor
  to avoid re-processing them next time.
- Past month files are frozen — never modified once the month has passed.
