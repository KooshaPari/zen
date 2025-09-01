#!/usr/bin/env bash
set -euo pipefail

# Creates a GitHub Projects board (Projects beta) for the fork and adds roadmap issues.
# Requirements:
# - gh CLI authenticated to github.com with scopes: project,read:project
#   Run: gh auth refresh -h github.com -s project,read:project
# - jq installed

OWNER="@me"
TITLE="Zen Roadmap"
REPO="KooshaPari/zen"

if [[ "$OWNER" == "@me" ]]; then
  OWNER_LOGIN=$(gh api user --jq .login)
else
  OWNER_LOGIN="$OWNER"
fi

echo "Creating project '$TITLE' for $OWNER_LOGIN..."
pj_json=$(gh project create --owner "$OWNER_LOGIN" --title "$TITLE" --format json)
pj_num=$(echo "$pj_json" | jq -r '.number')
echo "Project number: $pj_num"

echo "Linking repository $REPO..."
gh project link --owner "$OWNER_LOGIN" "$pj_num" --repo "$REPO"

echo "Adding roadmap issues to project..."
for num in $(gh issue list --repo "$REPO" --state open --label roadmap --json number --jq '.[].number'); do
  gh project item-add --owner "$OWNER_LOGIN" "$pj_num" --url "https://github.com/$REPO/issues/$num"
  echo "  added #$num"
done

echo "Done. Open the project and set Group by → Milestone and Filter by label:roadmap if desired."
gh project view "$pj_num" --owner "$OWNER_LOGIN" --web
