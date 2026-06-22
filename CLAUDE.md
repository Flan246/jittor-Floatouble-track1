# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 与本项目相关的维护约定（第六届计图 · JittorGeometric）

- **框架**：模型与提交代码须基于 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/)；图学习优先使用 [JittorGeometric](https://github.com/AlgRUC/JittorGeometric)。
- **环境**：`D:\miniconda3\envs\jittorgeometric`（Python 3.10）+ `jittor==1.3.7.16` + `JITTOR_HOME=D:\jittor_cache`。勿用系统 Python 3.13。仓库：`JittorGeometric/`（`pip install -e`）。
- **竞赛入口**：[头歌 Jittor-7](https://www.educoder.net/competitions/Jittor-7)；热身赛 1 = 引用网络论文分类。
- **开源命名**：`jittor-[战队名]-[项目名]`，B 榜须同步 GitHub + GitLink。
- **文档**：环境变更时更新 `docs/环境搭建.md`；赛题与赛程摘要放在 `docs/竞赛备忘.md`。
