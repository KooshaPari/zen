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

echo "Locating project: $TITLE (owner=$OWNER)" >&2
if [[ "$OWNER" == "@me" ]]; then
  OWNER_LOGIN=$(gh api user --jq .login)
else
  OWNER_LOGIN="$OWNER"
fi

proj=$(gh project list --owner "$OWNER_LOGIN" --format json | jq -r --arg t "$TITLE" '.[]|select(.title==$t)')
if [[ -z "$proj" ]]; then
  echo "Project not found. Create it first (see scripts/create_projects_board.sh)." >&2
  exit 1
fi
projNum=$(echo "$proj" | jq -r '.number')

echo "Fetching project fields..." >&2
fields=$(gh project field-list "$projNum" --owner "$OWNER_LOGIN" --format json)
get_field_id(){ echo "$fields" | jq -r --arg n "$1" '.[]|select(.name==$n)|.id' ; }

milestoneId=$(get_field_id Milestone)
labelsId=$(get_field_id Labels)

if [[ -z "$milestoneId" || "$milestoneId" == "null" ]]; then
  echo "Milestone field not found on project. Ensure repo is linked and items include repo issues." >&2
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

echo "Creating saved views..." >&2
create_view "By Milestone" "is:issue label:roadmap" "$milestoneId"
create_view "LLM by Milestone" "is:issue label:roadmap label:llm" "$milestoneId"
create_view "Protocol by Milestone" "is:issue label:roadmap label:protocol" "$milestoneId"
create_view "Infra by Milestone" "is:issue label:roadmap label:infra" "$milestoneId"

echo "Done. Open the project to see saved views." >&2
