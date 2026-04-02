#!/usr/bin/env bash
# Start/stop XMem server for AEL experiments.
#
# Usage:
#   ./scripts/start_xmem.sh start [experiment_name] [seed] [port]
#   ./scripts/start_xmem.sh stop [port]
#   ./scripts/start_xmem.sh status [port]
#
# Each experiment gets its own SQLite DB for isolation.

set -euo pipefail

XMEM_DIR="${XMEM_DIR:-xmem-server}"
DEFAULT_PORT=8200

ACTION="${1:-status}"
EXPERIMENT="${2:-default}"
SEED="${3:-42}"
PORT="${4:-$DEFAULT_PORT}"

DB_PATH="/tmp/ael_${EXPERIMENT}_${SEED}.db"
PID_FILE="/tmp/xmem_ael_${PORT}.pid"

case "$ACTION" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "XMem already running on port $PORT (PID $(cat "$PID_FILE"))"
            exit 0
        fi

        echo "Starting XMem server:"
        echo "  Port:       $PORT"
        echo "  DB:         $DB_PATH"
        echo "  Experiment: $EXPERIMENT (seed=$SEED)"

        python "$XMEM_DIR/server.py" \
            --port "$PORT" \
            --preset gpt4o-mini \
            --db-path "$DB_PATH" \
            --decay-half-life 14.0 \
            --consolidation-threshold 1500 \
            --bg-consolidation-interval 120 \
            &

        echo $! > "$PID_FILE"
        sleep 2

        # Health check
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "XMem server started (PID $(cat "$PID_FILE"))"
        else
            echo "WARNING: XMem server may not be ready yet, check logs"
        fi
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                echo "XMem server stopped (PID $PID)"
            else
                echo "PID $PID not running"
            fi
            rm -f "$PID_FILE"
        else
            echo "No PID file found for port $PORT"
        fi
        ;;

    status)
        if curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
            echo ""
            echo "XMem server is running on port $PORT"
        else
            echo "XMem server is NOT running on port $PORT"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|status} [experiment] [seed] [port]"
        exit 1
        ;;
esac
