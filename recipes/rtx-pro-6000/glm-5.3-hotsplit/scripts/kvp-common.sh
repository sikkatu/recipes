#!/usr/bin/env bash
# kvp-common.sh -- the ONE shared shell helper set for this recipe.
# Source it from every script:  . "$(dirname "${BASH_SOURCE[0]}")/kvp-common.sh"
#
# This file is copied per recipe (recipes must stay standalone) and the copies
# are kept BYTE-IDENTICAL by tests/test_cross_repo_contract.py -- a fix here
# must land in every recipe at once.
#
# Provides:
#   kvp_curl [curl args...]      curl with the API key sent on a stdin pipe
#                                (printf is a bash builtin -- the key is never
#                                visible in ps and no key file is written;
#                                without a key the request is unauthenticated,
#                                the documented ALLOW_NO_AUTH loopback mode).
#   _kvp_seed_env_file <file>    seed shell variables from an env-style file
#                                (tp4.env contract: environment wins over the
#                                file; indented comments skipped; inline
#                                comments cut at " #"; $HOME expanded; only
#                                valid shell identifiers are exported).

kvp_curl(){
  if [ -n "${API_KEY:-}" ]; then
    # A curl config file parses " ..." quoted values -- an unescaped " or \
    # inside the key is silently mangled (abc"def -> abc). Escape both so ANY
    # key bytes reach the server intact.
    _kvp_esc="${API_KEY//\\/\\\\}"
    _kvp_esc="${_kvp_esc//\"/\\\"}"
    printf 'header = "Authorization: Bearer %s"\n' "$_kvp_esc" | curl -K - "$@"
  else
    curl "$@"
  fi
}

_kvp_seed_env_file(){
  _kvp_sf="$1"
  [ -f "$_kvp_sf" ] || return 0
  while IFS= read -r _kvp_line || [ -n "$_kvp_line" ]; do
    case "$_kvp_line" in
      ''|[[:space:]]*'#'*) continue ;;          # blanks + comment lines
    esac
    _kvp_line="${_kvp_line#export }"
    _kvp_name="${_kvp_line%%=*}"
    [ "$_kvp_name" = "$_kvp_line" ] && continue # not an assignment
    _kvp_name="$(printf '%s' "$_kvp_name" | tr -d ' \t')"
    case "$_kvp_name" in
      ''|'#'*) continue ;;
    esac
    printf '%s' "$_kvp_name" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]*$' || continue
    if [ -z "${!_kvp_name:-}" ]; then
      _kvp_val="${_kvp_line#*=}"
      _kvp_val="${_kvp_val%% #*}"                       # inline comment at " #"
      _kvp_val="${_kvp_val#"${_kvp_val%%[![:space:]]*}"}"  # leading whitespace
      _kvp_val="${_kvp_val%"${_kvp_val##*[![:space:]]}"}"  # trailing whitespace
      _kvp_val="${_kvp_val//\"/}"                       # surrounding double quotes
      case "$_kvp_val" in
        *'$HOME'*|*'${HOME}'*)
          _kvp_val="${_kvp_val//\$HOME/$HOME}"
          _kvp_val="${_kvp_val//\$\{HOME\}/$HOME}" ;;
      esac
      export "$_kvp_name=$_kvp_val"
    fi
  done < "$_kvp_sf"
  return 0
}
