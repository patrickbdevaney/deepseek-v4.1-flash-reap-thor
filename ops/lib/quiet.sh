# Shared model-loading-process detection (from cuda/CHAIN_STEPS.sh, ops audit W5): matches the program name / python
# script argv, never a whole command line, so editors, pagers and claude shells that merely mention a stage do not count.
QUIET_SKIP_COMM=" bash sh dash zsh fish claude node less more vi vim nvim view nano emacs tail head cat grep egrep rg pgrep ps top htop watch sed awk tmux screen ssh sshd sudo timeout journalctl "
gpu_busy() { local anc=" " p=$$ d pid comm a0 a s m prev hits=""
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do anc+="$p "; p=$(awk '/^PPid:/{print $2}' /proc/$p/status 2>/dev/null); done
  for d in /proc/[0-9]*; do pid=${d#/proc/}
    case "$anc" in *" $pid "*) continue ;; esac
    [ "$pid" = "$BASHPID" ] && continue
    { read -r comm < $d/comm; } 2>/dev/null || continue
    case "$QUIET_SKIP_COMM" in *" $comm "*) continue ;; esac
    local -a argv=(); { mapfile -d '' -t argv < $d/cmdline; } 2>/dev/null || continue
    [ ${#argv[@]} -gt 0 ] || continue
    a0=${argv[0]##*/}; m=""
    case "$a0" in
      v41_decode|v41_server|gate_v41_*|gemv_format_bench|v41-teacher-kld|test-backend-ops|vllm|\
      llama-server|llama-cli|llama-perplexity|llama-completion|llama-imatrix) m=$a0 ;;
      python|python3|python3.*)
        prev=""
        for a in "${argv[@]:1}"; do
          if [ "$prev" = -m ]; then case "$a" in vllm|vllm.*|sglang|sglang.*) m="$a0 -m $a" ;; esac; break; fi
          [ "$prev" = -c ] && break
          case "$prev" in -X|-W) prev=""; continue ;; esac      # the value of -X opt / -W opt
          case "$a" in -*) prev=$a; continue ;; esac
          s=${a##*/}
          case "$s" in
            s0[1-9]_*.py|s0[1-9][a-z]_*.py|s1[0-1]_*.py|s1[0-1][a-z]_*.py|ref_dump.py|v41_unit_ref.py) m="$a0 $s" ;;
          esac
          break
        done ;;
    esac
    [ -n "$m" ] && hits+="${hits:+, }pid $pid ($m)"
  done
  echo "$hits"; }
leaf_runner_up() { local d; local -a argv
  for d in /proc/[0-9]*; do argv=(); { mapfile -d '' -t argv < $d/cmdline; } 2>/dev/null || continue
    [ ${#argv[@]} -ge 2 ] || continue
    case "${argv[0]##*/}" in bash|sh) case "${argv[1]:-}" in *pipeline/runner.sh) return 0 ;; esac ;; esac
  done; return 1; }
