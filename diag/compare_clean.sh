#!/usr/bin/env bash
cd /workspace/v2data
JUNK='MwSt|Cookie|Impressum|Datenschutz|Newsletter|Postfach|Versand|Warenkorb|Tel\.:'
CRUMB='^(Home \||Sie befinden sich hier|Aktuelle Seite|Startseite [A-Z])'
echo "===== RP CLEANED SAMPLES ====="
sed -n '20p;400p;4000p' cleaned/_chunks/rp_de_00.clean | cut -c1-200
echo
echo "===== HPLT CLEANED SAMPLES ====="
sed -n '20p;400p;4000p' cleaned/_chunks/hplt_de_00.clean | cut -c1-200
echo
echo "===== JUNK-WORD LINES (per 100k lines, raw vs cleaned) ====="
printf 'RP    raw=%-6s cleaned=%s\n' "$(head -n100000 raw/german/redpajama_de.txt|grep -cE "$JUNK")" "$(head -n100000 cleaned/_chunks/rp_de_00.clean|grep -cE "$JUNK")"
printf 'HPLT  raw=%-6s cleaned=%s\n' "$(head -n100000 raw/german/hplt_de.txt|grep -cE "$JUNK")" "$(head -n100000 cleaned/_chunks/hplt_de_00.clean|grep -cE "$JUNK")"
echo "===== BREADCRUMB-START LINES (per 100k, raw vs cleaned) ====="
printf 'RP    raw=%-6s cleaned=%s\n' "$(head -n100000 raw/german/redpajama_de.txt|grep -cE "$CRUMB")" "$(head -n100000 cleaned/_chunks/rp_de_00.clean|grep -cE "$CRUMB")"
printf 'HPLT  raw=%-6s cleaned=%s\n' "$(head -n100000 raw/german/hplt_de.txt|grep -cE "$CRUMB")" "$(head -n100000 cleaned/_chunks/hplt_de_00.clean|grep -cE "$CRUMB")"
