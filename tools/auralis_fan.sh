#!/bin/bash
# =============================================================================
#  Auralis Fan Control — Engine (Stufe 1)
#  Windows-artige Lüfterkurven für Unraid (ITE IT8689, Gigabyte B550)
#
#  SICHERHEIT:
#   - Default-Modus = "observe": rechnet + loggt, SCHREIBT NICHTS.
#   - "live" schreibt PWM. Bei Übertemperatur / Sensor-Fehler -> FAILSAFE 100%.
#   - Beim Beenden werden die BIOS-Lüftermodi wiederhergestellt (sicher laut).
#
#  Aufruf:  auralis_fan.sh [observe|live] [interval_seconds]
# =============================================================================
set -u

# ---- Modus / Intervall ------------------------------------------------------
MODE="${1:-observe}"          # observe | live
INTERVAL="${2:-5}"            # Sekunden pro Regel-Zyklus
DEADBAND=6                    # nur schreiben, wenn |Ziel-Letzt| >= DEADBAND
LOG="/var/log/auralis-fan.log"

# ---- Sensor-Pfade (auf diesem Board ermittelt) ------------------------------
HW="/sys/class/hwmon/hwmon4"             # it8689: pwm1..5, fan1..5_input
CPU_INPUT="/sys/class/hwmon/hwmon1/temp1_input"   # k10temp Tctl (milli-°C)
NVME_INPUT="/sys/class/hwmon/hwmon0/temp1_input"  # NVMe Composite (milli-°C)
DISKS_INI="/var/local/emhttp/disks.ini"           # Unraid Platten-Temps (ohne Spin-up)

# ---- Kritische Schwellen (°C) -> FAILSAFE 100% ------------------------------
GPU_CRIT=87
CPU_CRIT=88
DISK_CRIT=48
NVME_CRIT=78

# ---- Kanal-Konfiguration ----------------------------------------------------
#  PROVISORISCH: Zuordnung pwm<->Header noch NICHT verifiziert.
#  Quelle:  cpu | gpu | nvme | disks | max   (max = max(cpu,gpu))
#  Kurve:   "tempC:pwm  tempC:pwm ..."  (0..255), linear interpoliert
#  Floor:   Mindest-PWM (Lüfter nie ganz aus / Brummgrenze)
declare -A SRC CURVE FLOOR
SRC[1]="cpu";  CURVE[1]="40:70 55:110 70:190 80:255";  FLOOR[1]=60
SRC[2]="max";  CURVE[2]="45:70 60:120 72:185 83:255";  FLOOR[2]=70
SRC[3]="max";  CURVE[3]="45:70 60:120 72:185 83:255";  FLOOR[3]=60
SRC[4]="max";  CURVE[4]="45:70 60:120 72:185 83:255";  FLOOR[4]=70
SRC[5]="max";  CURVE[5]="45:70 60:120 72:185 83:255";  FLOOR[5]=70

declare -A LAST   # zuletzt geschriebener Wert je Kanal

# ---- Helfer -----------------------------------------------------------------
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
abs() { local v=$1; [ "$v" -lt 0 ] && v=$(( -v )); echo "$v"; }

read_temps() {
  CPU=""; GPU=""; NVME=""; DISKMAX=""
  # CPU (Tctl)
  if [ -r "$CPU_INPUT" ]; then CPU=$(( ($(cat "$CPU_INPUT") + 500) / 1000 )); fi
  # NVMe
  if [ -r "$NVME_INPUT" ]; then NVME=$(( ($(cat "$NVME_INPUT") + 500) / 1000 )); fi
  # GPU (nvidia-smi)
  GPU=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  # Platten (Unraid disks.ini, temp="NN" -> max, "*" = spun down/none ueberspringen)
  if [ -r "$DISKS_INI" ]; then
    DISKMAX=$(awk -F'"' '/^temp=/{ if ($2 ~ /^[0-9]+$/ && $2+0>m) m=$2+0 } END{ if(m>0) print m }' "$DISKS_INI")
  fi
}

# Kurve auswerten: $1=temp  $2="t:p t:p ..."  -> PWM 0..255
calc_pwm() {
  awk -v t="$1" -v curve="$2" 'BEGIN{
    n=split(curve,a," ");
    for(i=1;i<=n;i++){ split(a[i],b,":"); tt[i]=b[1]+0; pp[i]=b[2]+0 }
    if(t<=tt[1]){ printf "%d", pp[1]; exit }
    if(t>=tt[n]){ printf "%d", pp[n]; exit }
    for(i=1;i<n;i++){
      if(t>=tt[i] && t<=tt[i+1]){
        f=(t-tt[i])/(tt[i+1]-tt[i]); v=pp[i]+f*(pp[i+1]-pp[i]);
        if(v<0)v=0; if(v>255)v=255; printf "%d", int(v+0.5); exit
      }
    }
    printf "%d", pp[n]
  }'
}

src_temp() {  # gewählte Quelle -> Temperatur (°C, leer wenn unbekannt)
  case "$1" in
    cpu)   echo "${CPU}";;
    gpu)   echo "${GPU}";;
    nvme)  echo "${NVME}";;
    disks) echo "${DISKMAX}";;
    max)   local m="${CPU:-0}"; [ -n "${GPU}" ] && [ "${GPU}" -gt "$m" ] && m="${GPU}"; echo "$m";;
    *)     echo "${CPU}";;
  esac
}

set_pwm() {   # $1=kanal  $2=wert  (nur live)
  [ "$MODE" != "live" ] && return 0
  echo 1   > "$HW/pwm${1}_enable" 2>/dev/null   # 1 = manueller Modus
  echo "$2" > "$HW/pwm${1}"       2>/dev/null
}

restore() {   # beim Beenden: BIOS-Modi zurück (sicher = laut)
  log "EXIT: stelle BIOS-Lüftermodi wieder her (failsafe)"
  if [ "$MODE" = "live" ]; then
    echo 2 > "$HW/pwm1_enable" 2>/dev/null            # pwm1 -> Auto
    for i in 2 3 4 5; do echo 0 > "$HW/pwm${i}_enable" 2>/dev/null; done  # -> Vollgas
  fi
  exit 0
}
trap restore INT TERM

# ---- Hauptschleife ----------------------------------------------------------
log "=== Auralis Fan Control gestartet | MODE=$MODE interval=${INTERVAL}s ==="
[ "$MODE" = "observe" ] && log "OBSERVE: es wird NICHTS geschrieben — nur Anzeige, was getan würde."

while true; do
  read_temps

  # ---- Failsafe-Prüfung ----
  FAILSAFE=0; REASON=""
  if [ -z "$GPU" ]; then FAILSAFE=1; REASON="GPU-Sensor fehlt"; GPU=99; fi
  [ -n "$GPU" ]     && [ "$GPU"     -ge "$GPU_CRIT"  ] && { FAILSAFE=1; REASON="GPU>=$GPU_CRIT"; }
  [ -n "$CPU" ]     && [ "$CPU"     -ge "$CPU_CRIT"  ] && { FAILSAFE=1; REASON="CPU>=$CPU_CRIT"; }
  [ -n "$DISKMAX" ] && [ "$DISKMAX" -ge "$DISK_CRIT" ] && { FAILSAFE=1; REASON="DISK>=$DISK_CRIT"; }
  [ -n "$NVME" ]    && [ "$NVME"    -ge "$NVME_CRIT" ] && { FAILSAFE=1; REASON="NVME>=$NVME_CRIT"; }
  [ -z "$CPU" ]     && { FAILSAFE=1; REASON="CPU-Sensor fehlt"; }

  line="CPU=${CPU:-NA} GPU=${GPU:-NA} DISK=${DISKMAX:-NA} NVME=${NVME:-NA}"

  for i in 1 2 3 4 5; do
    s="${SRC[$i]}"; st=$(src_temp "$s"); [ -z "$st" ] && st=0
    if [ "$FAILSAFE" = 1 ]; then
      tgt=255
    else
      tgt=$(calc_pwm "$st" "${CURVE[$i]}")
      [ "$tgt" -lt "${FLOOR[$i]}" ] && tgt="${FLOOR[$i]}"
    fi
    cur=$(cat "$HW/pwm${i}" 2>/dev/null)
    rpm=$(cat "$HW/fan${i}_input" 2>/dev/null)

    if [ "$MODE" = "live" ]; then
      d=$(abs $(( tgt - ${LAST[$i]:-(-999)} )))
      if [ "$FAILSAFE" = 1 ] || [ -z "${LAST[$i]:-}" ] || [ "$d" -ge "$DEADBAND" ]; then
        set_pwm "$i" "$tgt"; LAST[$i]="$tgt"
      fi
    fi
    line="$line | pwm$i($s ${st}C) now=${cur:-?} rpm=${rpm:-?} ->${tgt}"
  done

  [ "$FAILSAFE" = 1 ] && line="$line  *** FAILSAFE 100% ($REASON) ***"
  log "$line"
  sleep "$INTERVAL"
done
