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
#   _kvp_validate_posint NAME    die unless $NAME holds ^[1-9][0-9]*$ -- garbage
#                                must never reach a --flag of the engine.
#   _kvp_note_old_default NAME OLD NEW FILE
#                                one NOTE per launcher run when FILE pins NAME
#                                to OLD (the pre-change default) and NEW is the
#                                current default. The file is NEVER edited.

# The concurrency default lives here ONCE so every launcher reads the same
# value (review round 87). MiMo hotsplit launchers consume it; the glm /
# deepseek recipes keep their own literal MAX_NUM_SEQS=1 -- they were NOT
# checked at concurrency > 1 and must not silently inherit this number.
KVP_DEFAULT_MAX_NUM_SEQS=10

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
    _kvp_name="$(printf '%s' "$_kvp_name" | tr -d '[:space:]')"
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
      _kvp_val="$(printf '%s' "$_kvp_val" | tr -d "\'")"   # strip single quotes too (r88 L2), quote-safe
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


# -- round 87 (MAX_NUM_SEQS=10 default): refuse garbage before it reaches
#    the engine's --flag. Call AFTER seeding tp4.env so file values are checked.
_kvp_validate_posint(){
  _kvp_vn="$1"
  _kvp_vv="${!_kvp_vn:-}"
  printf '%s' "$_kvp_vv" | grep -Eq '^[1-9][0-9]*$' && return 0
  echo "ERROR: $_kvp_vn must be a positive integer, got '${_kvp_vv:-<unset>}'." >&2
  echo "  Set $_kvp_vn=<N> (1, 2, 3, ...) in tp4.env or the environment." >&2
  return 1
}

# -- round 87: a tp4.env copied from an OLD tp4.env.example pins the OLD
#    default (e.g. MAX_NUM_SEQS=2). A valid user choice must keep working --
#    so we only print ONE NOTE per launcher run and NEVER edit the file.
_kvp_note_old_default(){
  _kvp_nd_name="$1"; _kvp_nd_old="$2"; _kvp_nd_new="$3"; _kvp_nd_file="$4"
  [ -f "$_kvp_nd_file" ] || return 0
  # an uncommented assignment NAME=OLD, tolerating spaces and a trailing comment
  _kvp_nd_line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${_kvp_nd_name}[[:space:]]*=" "$_kvp_nd_file" | tail -1 || true)"
  [ -n "$_kvp_nd_line" ] || return 0
  _kvp_nd_val="${_kvp_nd_line#*=}"
  _kvp_nd_val="${_kvp_nd_val%% #*}"
  # Strip surrounding double/single quotes from the VALUE before trimming
  # whitespace (review r88 L2): `NAME="2"` must NOT miss the note, and
  # `NAME='2'` must match the same value instead of tripping the validator.
  # strip leading/trailing quote chars (double OR single), review r88 L2
  _kvp_nd_val="$(printf '%s' "$_kvp_nd_val" | sed -e "s/^[\"']//" -e "s/[\"']$//")"
  _kvp_nd_val="$(printf '%s' "$_kvp_nd_val" | tr -d '[:space:]')"
  [ "$_kvp_nd_val" = "$_kvp_nd_old" ] || return 0
  # r88 L2 cont.: if the EFFECTIVE value (env) already differs from the file's
  # pin, the operator has overridden it -- no NOTE (nothing stale is in play).
  _kvp_nd_eff="${!_kvp_nd_name:-}"
  [ -n "$_kvp_nd_eff" ] && [ "$_kvp_nd_eff" != "$_kvp_nd_val" ] && return 0
  echo "NOTE: tp4.env pins ${_kvp_nd_name}=${_kvp_nd_old} (the old default); the recipe default is now ${_kvp_nd_new}. Delete that line to use the new default. Nothing was changed." >&2
  return 0
}
