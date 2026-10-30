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

qmemtable() {
    local c json

    # Read JSON from pipe, or query Qdrant directly
    if [ -t 0 ]; then
        c=$(_qc "$@") || return 1
        json=$(curl -fsS "$Q_URL/collections/$c/memory") || return 1
    else
        json=$(cat) || return 1
    fi

    printf '%s\n' "$json" | jq -r '
        .result as $r |
        [
            ["Original Vectors",
                ($r.vectors[0].storage.disk_bytes // 0),
                ($r.vectors[0].storage.ram_bytes // 0),
                ($r.vectors[0].storage.cached_bytes // 0)],

            ["HNSW Index",
                ($r.vectors[0].index.disk_bytes // 0),
                ($r.vectors[0].index.ram_bytes // 0),
                ($r.vectors[0].index.cached_bytes // 0)],

            ["Quantized Vectors",
                ($r.vectors[0].quantized.disk_bytes // 0),
                ($r.vectors[0].quantized.ram_bytes // 0),
                ($r.vectors[0].quantized.cached_bytes // 0)],

            ["Payload",
                ($r.payload.disk_bytes // 0),
                ($r.payload.ram_bytes // 0),
                ($r.payload.cached_bytes // 0)],

            ["ID Tracker",
                ($r.other.id_tracker.disk_bytes // 0),
                ($r.other.id_tracker.ram_bytes // 0),
                ($r.other.id_tracker.cached_bytes // 0)],

            ["Total",
                ($r.total.disk_bytes // 0),
                ($r.total.ram_bytes // 0),
                ($r.total.cached_bytes // 0)]
        ] |
        .[] |
        [.[0], (.[1] / 1000000), (.[2] / 1000000), (.[3] / 1000000)] |
        @tsv
    ' | awk -F '\t' '
        BEGIN {
            printf "+----------------------+--------------+--------------+------------------+\n"
            printf "| %-20s | %12s | %12s | %16s |\n", \
                "Component", "Disk (MB)", "RAM (MB)", "Page cache (MB)"
            printf "+----------------------+--------------+--------------+------------------+\n"
        }

        {
            printf "| %-20s | %12.2f | %12.2f | %16.2f |\n", \
                $1, $2, $3, $4

            if ($1 == "Total") {
                printf "+----------------------+--------------+--------------+------------------+\n"
            }
        }
    '
}