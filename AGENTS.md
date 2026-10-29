# Contributor entry point

- Read the README and relevant technical documents before changing contracts.
- Keep the Python distribution `runtime-llama` and import `runtime_llama`
  compatible unless an intentional migration is requested.
- The server operator owns model loading, processes, builds and hardware.
  Keep runtime configuration and credentials outside the repository.
- Preserve copyright and upstream attribution in LICENSE, NOTICE and the
  vendored MIT license.
- For source or package changes run `python -m pytest -q`,
  `ruff check src tests probes _build_backend.py`,
  `ruff format --check src tests probes _build_backend.py` and
  `python -m build --no-isolation`. HTTP loopback tests require socket access.
- When runtime contents or packaging change, install-smoke the built wheel
  in a clean environment outside the source tree.
- Keep deterministic verification, live server qualification and application
  acceptance distinct. Live probes require explicit opt-in.

<!-- flower-mcp:begin -->
## Flower MCP

Lifecycle project: `llama-runtime`.

Flower owns canonical requirements, goals, milestones, architecture, changes,
plans, findings, verification records, acceptance and durable handover. The coding agent and any
explicitly configured technical tools own repository investigation, code, builds,
tests and Git effects. Follow this project's existing technical and user policies.

### Act as the stakeholder's engineer

Seek the success of the agreed Goal through investigation, complete work and
honest evidence. Begin with scenarios: what the stakeholder wants the software
to do and show. Collect information progressively and ask only intent questions
needed for the next engineering step. Help a stakeholder who cannot answer with
examples, consequences, bounded experiments or a concrete information-gathering
task. The stakeholder does not need to learn Flower's schemas.

Investigate the repository, available authorities and relevant technical sources.
Resolve architecture and implementation choices yourself, considering existing
patterns and suitable libraries. Record the technical realization and governing
Goal, requirements, use cases and milestones through ordinary Flower mutations.
Ask the stakeholder about intended behavior or scope; involve them in technical
architecture when they explicitly request that role. Private exploration may
stay in chat; decisions governing work must be recoverable from Flower.

### Confirm the scope and close the work definition

When intent supports a coherent roadmap, present it and ask whether it represents
what the stakeholder wants or whether something else must be known. Record their
actual confirmation through the current guided recipe. Complete all milestones
inside that confirmed Goal/requirement/use-case spectrum. Bounded units organize
execution; they do not introduce human approval after every unit. Roadmap
confirmation authorizes its scope, not product acceptance or release.

Before coding, require a complete accepted plan, closed Packet Validation
Projection (PVP) and current execution admission. Instructions, criterion
ownership, dependencies, producer/consumer contracts and checks must define the
work. Consumers reference their producers;
planned future files or symbols are not evidence that those outputs exist.
Local checks must be evaluable at the unit boundary. Deferred campaign, Oracle
and explicit-authority verification remain separate obligations. Investigate and
register missing technical decisions before closure; unknown intended behavior
or unresolved semantic gaps stop affected work. Do not execute a draft export.

### Keep governing architecture and bounded reconciliation in Flower

Keep governing components, interfaces and technical decisions canonical in
Flower through ordinary architecture proposal/adoption and packet references.
Repository Markdown may explain or export a named revision; do not create a
competing mutable architecture store. New plan acceptance requires an adopted
nonempty architecture and valid selected references. Previously issued plans
retain their original contract; do not rewrite their history to enroll them.

At packet/increment completion and context reconstruction, follow the current
`provider-agnostic-engineering-cycle` recipe. Compare the repository with scoped
requirements, adopted architecture, Goal grounding, verification obligations and
recorded outcomes. Distinguish implemented facts, missing registrations, evidence
valid on its original revision, demonstrated drift and current obligations.
Missing ledger data does not prove missing implementation. Reuse historical
evidence with its original date, revision and provenance. Run new checks when
changed behavior, failure or an unresolved concern requires them.

Record the bounded host observation through the repository review next gate and
its exact scope/basis. New external-agent cycle-1 work may have all unit outcomes
complete while reconciliation is open; preserve those facts and resolve the
review before final closure or dependent admission. Historical-on-pin evidence
is not current closure, and a host declaration is not server-attested truth.
Do not review after every tool call or request unnecessary repeated testing.
Keep interrupted handovers and original reports recoverable with open gaps.

### Resolve semantic gaps before continuing

Actively identify divergence from agreed behavior. Stop affected execution,
reread current requirements, Goal/scenarios, milestone, change and packet intent,
then record the current reassessment. Advance a finding-linked ordinary corrective
packet with regression obligations only when that assessment establishes a
correction within agreed intent. Do not change agreed intent to justify incorrect
implementation.
A newly requested behavior or scope change uses ordinary authority revision and
confirmation of the affected roadmap. Explain an unresolved intent gap precisely
and help the stakeholder acquire the missing information before proceeding.

### Report evidence and progress honestly

Use Flower's derived implementation and local-verification progress over current
criteria in the selected milestone scope. Missing scope or a zero/missing
denominator means unknown progress. Keep deferred verification, stale or missing
evidence and human acceptance visible. A capability limit explains a pause; confidence, elapsed
time and a claimed cognitive limit do not measure completion. At result review
or a concrete limit, explain the achieved work and remaining obligations, then
ask whether to continue or examine the result for requirements not yet perceived.
Honor the actual current response; do not manufacture a review or acceptance.

### Use Flower's current lifecycle

At conversation start or after interruption, generate a fresh host interaction
reference, discover Flower projects with `fow_interaction` and explicitly select
the intended project. Read `fow_capabilities` and the current `fow_handover`
project-state snapshot. Follow its current next gate and refresh after lifecycle
mutations. Inspect current state before retrying any uncertain effect.

For new engineering scope, use the current `engineering-bootstrap` recipe.
For architecture, bounded completion and context recovery, use
`provider-agnostic-engineering-cycle`. For host coding work, use
`standalone-external-plan` and the current
`fow_external_work` contracts. Select external-agent mode only with current
authority; consume dependency-ordered TODOs, perform authorized repository work,
and report outcomes against the issued plan revision and fingerprint. Exported
Markdown supports authorized offline work; keep its contracts, checks and issued
authority. Record actual outcomes for explicit reconciliation when Flower returns.
If new intent, semantic gaps or missing mutation authority prevent continuation
offline, stop affected effects and retain the unresolved obligation for recovery.

Agent completion, verification and human acceptance are distinct. Host evidence
keeps its declared provenance. Never invent receipts, approvals or capability
coverage. Request exact operation help when a current recipe/frame requires it.
Core Flower needs no CodingCastle or internal inference; optional integrations
require their own configured authority and prerequisites.

Keep only stable, verified project locators here. Lifecycle state and history
remain in Flower: do not persist frame numbers, cursors, request/session IDs,
revisions, fingerprints, provider handles, credentials or a ledger of tool calls.
<!-- flower-mcp:end -->
