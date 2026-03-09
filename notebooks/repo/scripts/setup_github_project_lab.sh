#!/usr/bin/env bash
set -euo pipefail

# Configures the GitHub Project layer for the mechanistic-interpretability lab.
# Requires: gh auth refresh -s project

OWNER="base76-research-lab"
PROJECT_NUMBER="1"
REPO="base76-research-lab/mechanistic-interpretability-"

echo "Linking project ${PROJECT_NUMBER} to ${REPO}..."
gh project link "${PROJECT_NUMBER}" --owner "${OWNER}" --repo "${REPO}" || true

echo "Creating recommended project fields..."
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "State" --data-type SINGLE_SELECT --single-select-options "IDEA,QUESTION,PROTOCOL,RUN,ANALYSIS,INTERNAL_REVIEW,PACKAGE_READY,EXTERNAL_COMMUNICATION" || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Track" --data-type SINGLE_SELECT --single-select-options "ai_microscopy,hallucinations" || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Evidence" --data-type SINGLE_SELECT --single-select-options "Exploratory,Supported,Replicated" || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Artifact Type" --data-type SINGLE_SELECT --single-select-options "question,protocol,run,analysis,report,package,external,replication" || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Model" --data-type TEXT || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Layer" --data-type TEXT || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Claim Boundary" --data-type SINGLE_SELECT --single-select-options "open,defined,needs review" || true
gh project field-create "${PROJECT_NUMBER}" --owner "${OWNER}" --name "Linked Canonical Artifact" --data-type TEXT || true

echo "Adding seed issues to the project..."
for n in 1 2 3 4 5 6; do
  gh project item-add "${PROJECT_NUMBER}" --owner "${OWNER}" --url "https://github.com/${REPO}/issues/${n}" || true
done

cat <<'EOF'

Project scaffolding complete.

Next manual step in the GitHub web UI:
1. Open the project board.
2. Create the following views:
   - Research State
   - Track
   - Evidence
   - Packaging
   - Replication
   - Insights
   - Follow-ups
   - High Priority
   - Blocked
   - Cross-model
   - Hallucinations
   - External Readiness
3. Use GITHUB_PROJECT_LAB_SPEC.md for filters, grouping, and field visibility.

EOF
