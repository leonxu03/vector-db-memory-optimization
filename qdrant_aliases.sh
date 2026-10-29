Q_URL="${QDRANT_URL:-http://localhost:6333}"

# parse --c <name> from args
_qc() {
  while [ $# -gt 0 ]; do
    [ "$1" = "--c" ] && { echo "$2"; return 0; }
    shift
  done
  echo "Usage: <cmd> --c <collection_name>" >&2
  return 1
}

qlist()   { curl -s "$Q_URL/collections" | jq; }
qver()    { curl -s "$Q_URL/" | jq; }
qmem()    { local c; c=$(_qc "$@") || return 1; curl -s "$Q_URL/collections/$c/memory" | jq; }
qinfo()   { local c; c=$(_qc "$@") || return 1; curl -s "$Q_URL/collections/$c" | jq; }
qcount()  { local c; c=$(_qc "$@") || return 1
            curl -s -X POST "$Q_URL/collections/$c/points/count" \
              -H 'Content-Type: application/json' -d '{"exact": true}' | jq; }
qdelete() { local c ans; c=$(_qc "$@") || return 1
            printf "Delete collection '%s'? [y/N] " "$c"; read -r ans
            [ "$ans" = "y" ] && curl -s -X DELETE "$Q_URL/collections/$c" | jq || echo "Cancelled."; }

# memory summary for every collection
qmemall() {
  for c in $(curl -s "$Q_URL/collections" | jq -r '.result.collections[].name'); do
    echo "== $c"; curl -s "$Q_URL/collections/$c/memory" | jq .result.total
  done
}