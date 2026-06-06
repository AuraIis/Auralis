<?php
/* Auralis Fan Control — Backend (AJAX)
   Aktionen: status | save | start | stop                      */

$BASE   = "/boot/config/plugins/auralis-fan";
$CONF   = "$BASE/fan.conf";
$DAEMON = "$BASE/auralis_fan.sh";
$STATUS = "/tmp/auralis-fan.json";
$PIDF   = "/var/run/auralis-fan.pid";
$HW     = "/sys/class/hwmon/hwmon4";

header('Content-Type: application/json');

function rd($f){ return is_readable($f) ? trim(@file_get_contents($f)) : null; }
function mC($f){ $v = rd($f); return $v===null ? null : (int)round($v/1000); }

function read_disks(){
  $ini = "/var/local/emhttp/disks.ini"; if(!is_readable($ini)) return null;
  $max = null;
  foreach(file($ini) as $l){
    if(preg_match('/^temp="(\d+)"/',$l,$m)){ $t=(int)$m[1]; if($max===null||$t>$max)$max=$t; }
  }
  return $max;
}

function daemon_running($PIDF){
  if(!is_readable($PIDF)) return false;
  $pid = (int)trim(@file_get_contents($PIDF));
  return $pid>0 && file_exists("/proc/$pid");
}

$action = $_REQUEST['action'] ?? 'status';

if($action === 'status'){
  global $HW;
  $cpu  = mC("/sys/class/hwmon/hwmon1/temp1_input");
  $nvme = mC("/sys/class/hwmon/hwmon0/temp1_input");
  $disk = read_disks();
  $gpu  = null;
  $g = @shell_exec("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null");
  if($g!==null){ $g=trim($g); if(is_numeric($g)) $gpu=(int)$g; }

  $fans = [];
  for($i=1;$i<=5;$i++){
    $pwm = rd("$HW/pwm$i"); $rpm = rd("$HW/fan{$i}_input");
    $pct = $pwm===null ? null : (int)round($pwm*100/255);
    $fans[] = ["ch"=>$i,"pwm"=>($pwm===null?null:(int)$pwm),"pct"=>$pct,"rpm"=>($rpm===null?null:(int)$rpm)];
  }
  $live = is_readable($STATUS) ? json_decode(@file_get_contents($STATUS),true) : null;
  echo json_encode([
    "ok"=>true,
    "running"=>daemon_running($PIDF),
    "temps"=>["cpu"=>$cpu,"gpu"=>$gpu,"disk"=>$disk,"nvme"=>$nvme],
    "fans"=>$fans,
    "daemon"=>$live
  ]);
  exit;
}

if($action === 'save'){
  $p = $_POST;
  $modeOk = in_array($p['mode']??'', ['off','observe','live']) ? $p['mode'] : 'observe';
  $intI = max(2,(int)($p['interval']??5));
  $dead = max(1,(int)($p['deadband']??6));
  $gc=(int)($p['gpu_crit']??87); $cc=(int)($p['cpu_crit']??88);
  $dc=(int)($p['disk_crit']??48); $nc=(int)($p['nvme_crit']??78);

  $out  = "# Auralis Fan Control — Konfiguration (von GUI geschrieben)\n";
  $out .= "MODE=$modeOk\nINTERVAL=$intI\nDEADBAND=$dead\n";
  $out .= "GPU_CRIT=$gc\nCPU_CRIT=$cc\nDISK_CRIT=$dc\nNVME_CRIT=$nc\n";
  for($i=1;$i<=5;$i++){
    $src = in_array($p["src$i"]??'', ['cpu','gpu','nvme','disks','max']) ? $p["src$i"] : 'max';
    $cur = trim(preg_replace('/[^0-9: ]/','', $p["curve$i"]??'45:70 60:120 72:185 83:255'));
    if($cur==='') $cur='45:70 60:120 72:185 83:255';
    $flo = max(0,min(255,(int)($p["floor$i"]??60)));
    $out .= "CHAN$i=$src;$cur;$flo\n";
  }
  @file_put_contents($CONF.".tmp",$out); @rename($CONF.".tmp",$CONF);
  echo json_encode(["ok"=>is_readable($CONF),"saved"=>$modeOk]);
  exit;
}

if($action === 'start'){
  @shell_exec("nohup bash ".escapeshellarg($DAEMON)." >/dev/null 2>&1 &");
  usleep(400000);
  echo json_encode(["ok"=>daemon_running($PIDF)]);
  exit;
}

if($action === 'stop'){
  if(is_readable($PIDF)){
    $pid=(int)trim(@file_get_contents($PIDF));
    if($pid>0) @shell_exec("kill -TERM $pid 2>/dev/null");
  }
  usleep(500000);
  echo json_encode(["ok"=>!daemon_running($PIDF)]);
  exit;
}

echo json_encode(["ok"=>false,"error"=>"unknown action"]);
