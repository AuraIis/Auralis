#!/bin/bash
# =============================================================================
#  Auralis Fan Control — Engine (liest fan.conf, schreibt Live-Status fuer GUI)
#  SICHERHEIT: Default observe (schreibt nichts). live -> echte PWM.
#  Failsafe: Uebertemp/Sensorfehler -> 100%. Beim Stop -> BIOS-Modi zurueck.
#  Aufruf:  auralis_fan.sh   (Modus kommt aus fan.conf)
# =============================================================================
set -u

BASE="/boot/config/plugins/auralis-fan"
CONF="$BASE/fan.conf"
LOG="/var/log/auralis-fan.log"
STATUS="/tmp/auralis-fan.json"
PIDFILE="/var/run/auralis-fan.pid"

HW="/sys/class/hwmon/hwmon4"                       # it8689
CPU_INPUT="/sys/class/hwmon/hwmon1/temp1_input"   # k10temp Tctl
NVME_INPUT="/sys/class/hwmon/hwmon0/temp1_input"  # NVMe
DISKS_INI="/var/local/emhttp/disks.ini"           # Unraid Platten-Temps

declare -A SRC CURVE FLOOR LAST
declare -A ST NOW RPM WANT

load_conf() {
  MODE="observe"; INTERVAL=5; DEADBAND=6
  GPU_CRIT=87; CPU_CRIT=88; DISK_CRIT=48; NVME_CRIT=78
  local i
  for i in 1 2 3 4 5; do SRC[$i]="max"; CURVE[$i]="45:70 60:120 72:185 83:255"; FLOOR[$i]=60; done
  [ -r "$CONF" ] || return
  local raw line key val rest
  while IFS= read -r raw; do
    line="${raw%%#*}"
    line="$(printf '%s' "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$line" ] && continue
    key="${line%%=*}"; val="${line#*=}"
    case "$key" in
      MODE) MODE="$val";;
      INTERVAL) INTERVAL="$val";;
      DEADBAND) DEADBAND="$val";;
      GPU_CRIT) GPU_CRIT="$val";;
      CPU_CRIT) CPU_CRIT="$val";;
      DISK_CRIT) DISK_CRIT="$val";;
      NVME_CRIT) NVME_CRIT="$val";;
      CHAN1|CHAN2|CHAN3|CHAN4|CHAN5)
        i="${key#CHAN}"
        SRC[$i]="${val%%;*}"; rest="${val#*;}"
        CURVE[$i]="${rest%%;*}"; FLOOR[$i]="${rest##*;}";;
    esac
  done < "$CONF"
}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
abs() { local v=$1; [ "$v" -lt 0 ] && v=$(( -v )); echo "$v"; }

read_temps() {
  CPU=""; GPU=""; NVME=""; DISKMAX=""
  [ -r "$CPU_INPUT" ]  && CPU=$(( ($(cat "$CPU_INPUT") + 500) / 1000 ))
  [ -r "$NVME_INPUT" ] && NVME=$(( ($(cat "$NVME_INPUT") + 500) / 1000 ))
  GPU=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  [ -r "$DISKS_INI" ] && DISKMAX=$(awk -F'"' '/^temp=/{ if ($2 ~ /^[0-9]+$/ && $2+0>m) m=$2+0 } END{ if(m>0) print m }' "$DISKS_INI")
}

calc_pwm() {  # $1=temp $2="t:p t:p..." -> 0..255
  awk -v t="$1" -v curve="$2" 'BEGIN{
    n=split(curve,a," ");
    for(i=1;i<=n;i++){ split(a[i],b,":"); tt[i]=b[1]+0; pp[i]=b[2]+0 }
    if(n<1){ print 255; exit }
    if(t<=tt[1]){ printf "%d",pp[1]; exit }
    if(t>=tt[n]){ printf "%d",pp[n]; exit }
    for(i=1;i<n;i++){ if(t>=tt[i]&&t<=tt[i+1]){ f=(t-tt[i])/(tt[i+1]-tt[i]); v=pp[i]+f*(pp[i+1]-pp[i]); if(v<0)v=0; if(v>255)v=255; printf "%d",int(v+0.5); exit } }
    printf "%d",pp[n] }'
}

src_temp() {
  case "$1" in
    cpu)   echo "${CPU}";;
    gpu)   echo "${GPU}";;
    nvme)  echo "${NVME}";;
    disks) echo "${DISKMAX}";;
    max)   local m="${CPU:-0}"; [ -n "${GPU}" ] && [ "${GPU}" -gt "$m" ] && m="${GPU}"; echo "$m";;
    *)     echo "${CPU}";;
  esac
}

set_pwm() {  # $1=kanal $2=wert (nur live)
  [ "$MODE" != "live" ] && return 0
  echo 1    > "$HW/pwm${1}_enable" 2>/dev/null
  echo "$2" > "$HW/pwm${1}"        2>/dev/null
}

restore() {
  log "EXIT: BIOS-Lueftermodi wiederhergestellt (failsafe)"
  if [ "$MODE" = "live" ]; then
    echo 2 > "$HW/pwm1_enable" 2>/dev/null
    for i in 2 3 4 5; do echo 0 > "$HW/pwm${i}_enable" 2>/dev/null; done
  fi
  rm -f "$PIDFILE" 2>/dev/null
  exit 0
}
trap restore INT TERM

write_status() {
  { printf '{"ts":"%s","mode":"%s","failsafe":%s,"cpu":%s,"gpu":%s,"disk":%s,"nvme":%s,"fans":[' \
      "$(date '+%H:%M:%S')" "$MODE" "${FAILSAFE:-0}" "${CPU:-null}" "${GPU:-null}" "${DISKMAX:-null}" "${NVME:-null}"
    for i in 1 2 3 4 5; do
      sep=","; [ "$i" = 5 ] && sep=""
      printf '{"ch":%s,"src":"%s","srctemp":%s,"now":%s,"rpm":%s,"want":%s}%s' \
        "$i" "${SRC[$i]}" "${ST[$i]:-0}" "${NOW[$i]:-0}" "${RPM[$i]:-0}" "${WANT[$i]:-0}" "$sep"
    done
    printf ']}\n'
  } > "$STATUS.tmp" 2>/dev/null && mv "$STATUS.tmp" "$STATUS" 2>/dev/null
}

load_conf
echo $$ > "$PIDFILE" 2>/dev/null
log "=== gestartet | MODE=$MODE interval=${INTERVAL}s ==="

while true; do
  load_conf            # Config bei jedem Zyklus neu lesen -> GUI-Aenderungen wirken sofort
  read_temps

  FAILSAFE=0; REASON=""
  if [ -z "$GPU" ]; then FAILSAFE=1; REASON="GPU-Sensor fehlt"; GPU=99; fi
  [ -n "$GPU" ]     && [ "$GPU"     -ge "$GPU_CRIT"  ] && { FAILSAFE=1; REASON="GPU>=$GPU_CRIT"; }
  [ -n "$CPU" ]     && [ "$CPU"     -ge "$CPU_CRIT"  ] && { FAILSAFE=1; REASON="CPU>=$CPU_CRIT"; }
  [ -n "$DISKMAX" ] && [ "$DISKMAX" -ge "$DISK_CRIT" ] && { FAILSAFE=1; REASON="DISK>=$DISK_CRIT"; }
  [ -n "$NVME" ]    && [ "$NVME"    -ge "$NVME_CRIT" ] && { FAILSAFE=1; REASON="NVME>=$NVME_CRIT"; }
  [ -z "$CPU" ]     && { FAILSAFE=1; REASON="CPU-Sensor fehlt"; }

  for i in 1 2 3 4 5; do
    s="${SRC[$i]}"; st=$(src_temp "$s"); [ -z "$st" ] && st=0
    if [ "$FAILSAFE" = 1 ]; then tgt=255; else
      tgt=$(calc_pwm "$st" "${CURVE[$i]}")
      [ "$tgt" -lt "${FLOOR[$i]}" ] && tgt="${FLOOR[$i]}"
    fi
    ST[$i]="$st"; NOW[$i]="$(cat "$HW/pwm${i}" 2>/dev/null || echo 0)"
    RPM[$i]="$(cat "$HW/fan${i}_input" 2>/dev/null || echo 0)"; WANT[$i]="$tgt"
    if [ "$MODE" = "live" ]; then
      d=$(abs $(( tgt - ${LAST[$i]:-(-999)} )))
      if [ "$FAILSAFE" = 1 ] || [ -z "${LAST[$i]:-}" ] || [ "$d" -ge "$DEADBAND" ]; then
        set_pwm "$i" "$tgt"; LAST[$i]="$tgt"
      fi
    fi
  done

  [ "$FAILSAFE" = 1 ] && log "FAILSAFE 100% ($REASON) CPU=$CPU GPU=$GPU DISK=${DISKMAX:-NA}"
  write_status
  sleep "$INTERVAL"
done
