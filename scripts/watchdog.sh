#!/usr/bin/env bash
# watchdog.sh — Lightweight training watchdog
#
# Usage:
#   sudo -v && export WATCHDOG_SHUTDOWN_TOKEN=$(openssl rand -hex 32)
#   bash watchdog.sh

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
TRAINING_PROCESS="src/train.py"
CHECK_HOST="8.8.8.8"
GRACE_SECONDS=150
CHECKPOINT_WAIT_SECONDS=30
SHUTDOWN_COUNTDOWN=5
# ══════════════════════════════════════════════════════════════

SHUTDOWN_ARMED=false
[[ -n "$WATCHDOG_SHUTDOWN_TOKEN" ]] && SHUTDOWN_ARMED=true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Verify sudo credentials if armed ─────────────────────────
if $SHUTDOWN_ARMED; then
    if ! sudo -n /sbin/shutdown --help &>/dev/null; then
        echo "✖  Token set but sudo credentials not cached. Run: sudo -v"
        exit 1
    fi
    echo "✔  Shutdown ARMED — sudo credentials verified."
else
    echo "⚠  Shutdown DISARMED. Will SIGINT training but NOT power off."
    echo "   To arm: sudo -v && export WATCHDOG_SHUTDOWN_TOKEN=\$(openssl rand -hex 32)"
fi
echo

# ── Run in its own process group ─────────────────────────────
if [[ "$$" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]]; then
    exec setsid bash "$0" "$@"
fi
OWN_PGID=$(ps -o pgid= -p $$ | tr -d ' ')

# ── Sudo keepalive ────────────────────────────────────────────
sudo_keepalive() {
    while true; do sleep 300; sudo -n -v 2>/dev/null; done
}
$SHUTDOWN_ARMED && sudo_keepalive &

# ── Temp files ────────────────────────────────────────────────
CANCEL_FILE=$(mktemp)
LOSS_LOCK_FILE=$(mktemp)
LINK_MON_PIDFILE=$(mktemp)
ROUTE_MON_PIDFILE=$(mktemp)

cancelled() { [[ -s "$CANCEL_FILE" ]]; }
cancel()    { echo "1" > "$CANCEL_FILE"; }

# ── Cleanup on Ctrl+C / SIGTERM ──────────────────────────────
CLEANING_UP=false
cleanup() {
    # Re-entrancy guard: step 4 below sends SIGTERM to our own process
    # group (which includes this very process), and the unconditional
    # `cleanup` call at the bottom of the script is a second call path.
    # Without this guard, either one re-enters cleanup() and recurses
    # forever, printing "Watchdog stopping." endlessly and never reaching
    # the actual shutdown.
    $CLEANING_UP && return
    CLEANING_UP=true
    trap '' SIGINT SIGTERM   # stop reacting to further/self-sent signals

    log "Watchdog stopping."

    # 1. Signal cancel so every loop stops at its next check
    cancel

    # 2. Cancel any pending shutdown
    sudo -n /sbin/shutdown -c 2>/dev/null || true

    # 3. Kill the ip monitor processes by PID — this immediately unblocks
    #    the `while read` loops in watchdog_link/route (they're blocked on
    #    the fifo waiting for the next network event, not on sleep, so
    #    SIGTERM to the group alone won't unblock them in time)
    local lm rm
    lm=$(cat "$LINK_MON_PIDFILE"  2>/dev/null); [[ -n "$lm" ]] && kill "$lm"  2>/dev/null || true
    rm=$(cat "$ROUTE_MON_PIDFILE" 2>/dev/null); [[ -n "$rm" ]] && kill "$rm"  2>/dev/null || true

    # 4. SIGTERM whole process group (handle_loss subshells, keepalive, etc.)
    #    Note: this includes this process itself, but the trap is disarmed
    #    above so it's a no-op for us.
    kill -TERM -"$OWN_PGID" 2>/dev/null || true
    sleep 1

    # Clean up temp files before the SIGKILL below, since SIGKILL can't be
    # trapped/ignored and may end this process before later lines run.
    rm -f "$CANCEL_FILE" "$LOSS_LOCK_FILE" "$LINK_MON_PIDFILE" "$ROUTE_MON_PIDFILE"

    # 5. Force-kill anything still alive (this likely terminates this
    #    process too, via SIGKILL, before `exit 0` below executes — that's
    #    fine, the outcome is the same).
    kill -9 -"$OWN_PGID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Network check ─────────────────────────────────────────────
internet_alive() {
    timeout 2 bash -c "echo >/dev/tcp/$CHECK_HOST/53" 2>/dev/null
}

# ── Clear Ptyxis notifications ────────────────────────────────
clear_ptyxis_notifications() {
    local notif_file="$HOME/.local/share/gnome-shell/notifications"
    [[ -f "$notif_file" ]] || return
    [[ -n "$DBUS_SESSION_BUS_ADDRESS" ]] || return
    grep -qi "Ptyxis" "$notif_file" || return
    local uuids
    uuids=$(grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$notif_file" | sort -u)
    [[ -z "$uuids" ]] && return
    while IFS= read -r uuid; do
        gdbus call --session \
            --dest org.gtk.Notifications \
            --object-path /org/gtk/Notifications \
            --method org.gtk.Notifications.RemoveNotification \
            "org.gnome.Ptyxis" "$uuid" &>/dev/null || true
    done <<< "$uuids"
    log "Cleared Ptyxis notifications."
}

# ── SIGINT training ───────────────────────────────────────────
sigint_training() {
    local pids
    pids=$(pgrep -f "$TRAINING_PROCESS" 2>/dev/null)
    if [[ -z "$pids" ]]; then
        log "No training process found — skipping SIGINT."
        return
    fi
    log "SIGINT → PID(s) $pids"
    kill -INT $pids 2>/dev/null || true
}

# ── Wait for checkpoint ───────────────────────────────────────
wait_for_checkpoint() {
    log "Waiting up to ${CHECKPOINT_WAIT_SECONDS}s for checkpoint save..."
    local deadline=$(( $(date +%s) + CHECKPOINT_WAIT_SECONDS ))
    while (( $(date +%s) < deadline )); do
        cancelled && return
        if ! pgrep -f "$TRAINING_PROCESS" &>/dev/null; then
            log "Training process exited — checkpoint saved."
            return
        fi
        sleep 2
    done
    log "Checkpoint wait timed out — proceeding to shutdown."
}

# ── Shutdown sequence ─────────────────────────────────────────
do_shutdown() {
    if ! $SHUTDOWN_ARMED; then
        log "Shutdown skipped — WATCHDOG_SHUTDOWN_TOKEN not set."
        return
    fi
    clear_ptyxis_notifications
    log "Shutdown in ${SHUTDOWN_COUNTDOWN}s — Ctrl+C to cancel."
    for (( i=SHUTDOWN_COUNTDOWN; i>0; i-- )); do
        cancelled && { sudo -n /sbin/shutdown -c 2>/dev/null; log "Shutdown cancelled."; return; }
        sleep 1
    done
    cancelled && { sudo -n /sbin/shutdown -c 2>/dev/null; return; }
    log "Shutting down NOW."
    sudo -n /sbin/shutdown -h now "Watchdog: internet lost"
}

# ── Loss handler ──────────────────────────────────────────────
handle_loss() {
    local source="$1"
    cancelled && return

    # Mutex: only one thread handles a loss event at a time
    { flock -n 9 || return; } 9>"$LOSS_LOCK_FILE"

    log "Network event ($source) — confirming in 5s..."
    # Sleep 1s at a time so cancel flag is checked every second
    for (( s=0; s<5; s++ )); do
        cancelled && return
        sleep 1
    done
    cancelled && return

    if internet_alive; then
        log "False alarm — internet still reachable."
        return
    fi
    log "Internet loss confirmed."

    local remaining=$GRACE_SECONDS
    while (( remaining > 0 )); do
        cancelled && return
        if (( remaining % 10 == 0 )); then
            if internet_alive; then
                log "Internet restored — countdown cancelled."
                return
            fi
            log "Still offline — ${remaining}s remaining."
        fi
        sleep 1
        (( remaining-- ))
    done

    cancelled && return
    sigint_training
    wait_for_checkpoint
    cancelled && return
    do_shutdown
}

# ── Watchdog threads ──────────────────────────────────────────
# Use a named pipe so we can record the ip monitor PID and kill it
# directly in cleanup(), unblocking the read loop instantly.

watchdog_link() {
    log "Link watchdog started (ip monitor link)."
    local fifo; fifo=$(mktemp -u); mkfifo "$fifo"
    ip monitor link 2>/dev/null > "$fifo" &
    echo $! > "$LINK_MON_PIDFILE"
    while IFS= read -r line; do
        cancelled && break
        [[ "$line" =~ state[[:space:]]+DOWN ]] || continue
        log "Link DOWN: ${line:0:80}"
        handle_loss "link"
    done < "$fifo"
    rm -f "$fifo"
}

watchdog_route() {
    log "Route watchdog started (ip monitor route)."
    local fifo; fifo=$(mktemp -u); mkfifo "$fifo"
    ip monitor route 2>/dev/null > "$fifo" &
    echo $! > "$ROUTE_MON_PIDFILE"
    while IFS= read -r line; do
        cancelled && break
        [[ "$line" =~ ^Deleted.*default ]] || continue
        log "Default route removed: ${line:0:80}"
        handle_loss "route"
    done < "$fifo"
    rm -f "$fifo"
}

# ── Launch ────────────────────────────────────────────────────
watchdog_link &
LINK_PID=$!
watchdog_route &
ROUTE_PID=$!

log "Watchdog running. Ctrl+C to stop."

# FIX: loop with `wait -n` instead of bare `wait`.
# Bare `wait` delays SIGINT delivery until a child exits. With the loop,
# the signal is delivered to this shell promptly and cleanup() runs.
while kill -0 "$LINK_PID" 2>/dev/null || kill -0 "$ROUTE_PID" 2>/dev/null; do
    wait -n 2>/dev/null || true
    cancelled && break
done

cleanup
