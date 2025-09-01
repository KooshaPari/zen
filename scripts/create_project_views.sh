#!/usr/bin/env bash
set -euo pipefail

# Create saved views for the Zen Roadmap GitHub Project (Projects v2).
# Requirements:
#  - gh CLI authenticated with scopes: project,read:project
#    Run: gh auth refresh -h github.com -s project,read:project
#  - jq installed
#
# Usage:
#  ./scripts/create_project_views.sh [--owner @me] [--title "Zen Roadmap"]

OWNER="@me"
TITLE="Zen Roadmap"
REPO="KooshaPari/zen"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$OWNER" == "@me" ]]; then
  OWNER_LOGIN=$(gh api user --jq .login)
else
  OWNER_LOGIN="$OWNER"
fi

echo "Locating project: $TITLE (owner=$OWNER_LOGIN)" >&2

plist=$(gh project list --owner "$OWNER_LOGIN" --format json)
# Normalize regardless of output shape; prefer the project with items > 0, else highest number
proj=$(echo "$plist" | jq -r --arg t "$TITLE" '
  (if has("projects") then .projects else . end)
  | (if type=="array" then . else [.] end)
  | map(select(.title==$t))
  | sort_by([(.items.totalCount // 0), (.number // 0)])
  | last')
if [[ -z "$proj" || "$proj" == "null" ]]; then
  echo "Project not found by title. Available projects:" >&2
  echo "$plist" | jq -r 'if type=="array" then .[]|"- \(.number): \(.title)" else . end' >&2
  echo "Create it first (scripts/create_projects_board.sh) or pass --title <name>." >&2
  exit 1
fi
projNum=$(echo "$proj" | jq -r '.number')

echo "Fetching project fields..." >&2
fields_raw=$(gh project field-list "$projNum" --owner "$OWNER_LOGIN" --format json)
# Normalize to array of {id,name,...}
fields=$(echo "$fields_raw" | jq -c 'if type=="array" then . else (.fields // []) end')
get_field_id(){ echo "$fields" | jq -r --arg n "$1" '.[]|select(.name==$n)|.id' ; }

milestoneId=$(get_field_id Milestone)
labelsId=$(get_field_id Labels)

if [[ -z "$milestoneId" || "$milestoneId" == "null" ]]; then
  echo "Milestone field not found on project. Ensure repo is linked and items include repo issues." >&2
  echo "Available fields:" >&2
  echo "$fields" | jq -r '.[]|"- \(.name) (id=\(.id))"' >&2
  exit 1
fi

create_view(){
  local name="$1" filter="$2" groupBy="$3"
  gh api graphql -f query='mutation($project:ID!,$name:String!,$filter:String,$groupBy:ID){
    createProjectV2View(input:{projectId:$project,name:$name,filter:$filter,groupBy:$groupBy}){
      projectV2View{ id name }
    }
  }' -f project=$(gh api graphql -f query='query($owner:String!,$number:Int!){
    user(login:$owner){ projectV2(number:$number){ id }}
  }' -F owner="$OWNER_LOGIN" -F number=$projNum --jq '.data.user.projectV2.id') \
  -F name="$name" -F filter="$filter" -F groupBy="$groupBy" >/dev/null
  echo "Created view: $name" >&2
}

echo "NOTE: GitHub API currently does not support programmatic creation of saved views for Projects (beta) via GraphQL." >&2
echo "Open the project and create views manually with these settings:" >&2
echo "- By Milestone: filter=\"is:issue label:roadmap\"; Group by=Milestone" >&2
echo "- LLM by Milestone: filter=\"is:issue label:roadmap label:llm\"; Group by=Milestone" >&2
echo "- Protocol by Milestone: filter=\"is:issue label:roadmap label:protocol\"; Group by=Milestone" >&2
echo "- Infra by Milestone: filter=\"is:issue label:roadmap label:infra\"; Group by=Milestone" >&2
echo "Opening project in browser..." >&2
gh project view "$projNum" --owner "$OWNER_LOGIN" --web
