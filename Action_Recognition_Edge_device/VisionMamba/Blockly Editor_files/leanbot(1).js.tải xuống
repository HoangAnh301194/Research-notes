
"use strict";

goog.provide("Blockly.Arduino.leanbot");
goog.require("Blockly.Arduino");


/*==================================================================================================
                                      Variables
==================================================================================================*/

var LeanbotArduino = {};


/*==================================================================================================
                                      Leanbot
==================================================================================================*/

const LeanbotArduino_addHeader = function () {
  // workaround: avoid error in case this function is call before "Blockly.Arduino.init()" due to Async execution
  if (Blockly.Arduino.includes_ === undefined) { return; }
  if (Blockly.Arduino.setups_   === undefined) { return; }

  Blockly.Arduino.includes_["includes_leanbot"] = "#include <Leanbot.h>";
  Blockly.Arduino.setups_["setup_leanbot"]      = "Leanbot.begin();";
};


const LeanbotArduino_begin = function() {
  for (const [id, block] of Object.entries(LeanbotArduino)) {    // loop through each pair [key, value]
    // console.log(id, block);
    // init the block
    Blockly.Arduino[id]         = block;
    Blockly.Arduino[id + "_vi"] = block;
  }
};


/*==================================================================================================
                                      General
==================================================================================================*/

LeanbotArduino["Leanbot.commentLine"] = function(block) {
  LeanbotArduino_addHeader();

  var commentString = block.getFieldValue("commentString");

  var code = "\n" + "// " + commentString + "\n";
  return code;
};


LeanbotArduino["Leanbot.section"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");

  var commentString = block.getFieldValue("commentString");

  if (commentString.length > 0) {
    commentString = " // " + commentString;
  }

  var code = "";

  code += "\n";
  code += "{" + commentString + "\n";
  code += body;
  code += "}\n";
  code += "\n";

  return code;
};


LeanbotArduino["Leanbot.whenStarted"] = function(block) {
  return "";
};


LeanbotArduino["Leanbot.delay"] = function(block) {
  LeanbotArduino_addHeader();

  // default as "ms"
  var multiplier = "";
  var order      = Blockly.Arduino.ORDER_NONE;

  // multiply by 1000: "ms" to "sec"
  var dropdown_unit = block.getFieldValue("Unit");
  if (dropdown_unit === "seconds") {
    multiplier = " * 1000";
    order      = Blockly.Arduino.ORDER_MULTIPLICATIVE;
  }

  var value_delay = Blockly.Arduino.valueToCode(block, "Delay", order);

  var code = "LbDelay(" + value_delay + multiplier + ");\n";
  return code;
};


LeanbotArduino["Leanbot.DCMotor.setPower"] = function(block) {
  LeanbotArduino_addHeader();

  var value_power = Blockly.Arduino.valueToCode(block, "Power",  Blockly.Arduino.ORDER_MULTIPLICATIVE);

  if (value_power != 0) {
    var value_direction = block.getFieldValue("Direction");
    if (value_direction === "Backward") {
      value_power = "-" + value_power;
    }

    // convert from ±100 to ±255
    value_power = value_power + " * 255 / 100"
  }

  var code = "Leanbot.DCMotor.setPower(" + value_power + ");\n";
  return code;
};


/*==================================================================================================
                                      LbMission
==================================================================================================*/

LeanbotArduino["LbMission.beginByTeacher_end"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");
  // remove one level of indentation per line
  body = body.substring(2);
  body = body.replaceAll("\n  ", "\n");

  var comboTouch = block.getFieldValue("comboTouch");

  var code = "";

  code += "LbMission.begin(" + comboTouch + ");\n";
  // code += "\n";
  code += body;
  // code += "\n";
  code += "LbMission.end();\n";

  return code;
};


LeanbotArduino["LbMission.beginByServer_end"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");
  // remove one level of indentation per line
  body = body.substring(2);
  body = body.replaceAll("\n  ", "\n");

  var value_missionId = block.getFieldValue("missionId");
  value_missionId = `F("` + value_missionId + `")`;

  var code = "";

  code += "LbMission.beginDigitalTwin(" + value_missionId + ");\n";
  code += body;
  code += "LbMission.end();\n";

  return code;
};


LeanbotArduino["LbMission.beginByName_end"] = LeanbotArduino["LbMission.beginByServer_end"];  // re-use identical code


LeanbotArduino["LbMission.beginBySelect_end"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");
  // // remove one level of indentation per line
  body = body.substring(2);
  body = body.replaceAll("\n  ", "\n");

  var value_missionId = Blockly.Arduino.valueToCode(block, "missionId", Blockly.Arduino.ORDER_NONE) || `""`;
  value_missionId = `F(` + value_missionId + `)`;

  var code = "";

  code += "LbMission.beginDigitalTwin(" + value_missionId + ");\n";
  code += body;
  code += "LbMission.end();\n";

  return code;
};


LeanbotArduino["LbMission.missionSelect"] = function(block) {
  var value_missionSelect = block.getFieldValue("missionSelect");
  value_missionSelect = Blockly.Arduino.quote_(value_missionSelect);  // align with: Blockly.Arduino['text']
  return [value_missionSelect, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbMission.beginImmediately_end"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");
  // remove one level of indentation per line
  body = body.substring(2);
  body = body.replaceAll("\n  ", "\n");

  var code = "";

  code += "LbMission.begin(0);\n";
  // code += "\n";
  code += body;
  // code += "\n";
  code += "LbMission.end();\n";

  return code;
};


LeanbotArduino["LbMission.end"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbMission.end();\n";
  return code;
};


LeanbotArduino["LbMission.elapsedMillis"] = function(block) {
  LeanbotArduino_addHeader();

  var code  = "LbMission.elapsedMillis()";
  var order = Blockly.Arduino.ORDER_ATOMIC;

  var dropdown_unit = block.getFieldValue("Unit");
  if (dropdown_unit === "seconds") {
    code += " / 1000";
    order = Blockly.Arduino.ORDER_MULTIPLICATIVE;
  }

  return [code, order];
};


/*==================================================================================================
                                      LbMotion
==================================================================================================*/

LeanbotArduino["LbMotion.run"] = function(block) {
  LeanbotArduino_addHeader();

  var runFunction = "LbMotion.runLR";

  var dropdown_unitspeed = block.getFieldValue("UnitSpeed");
  if (dropdown_unitspeed === "rpm") {
    runFunction = "LbMotion.runLRrpm";
  }

  var value_left  = Blockly.Arduino.valueToCode(block, "Left",  Blockly.Arduino.ORDER_NONE);
  var value_right = Blockly.Arduino.valueToCode(block, "Right", Blockly.Arduino.ORDER_NONE);

  var code = runFunction + "(" + value_left + ", " + value_right + ");\n";
  return code;
};


LeanbotArduino["LbMotion.waitDistanceMm"] = function(block) {
  LeanbotArduino_addHeader();

  // default as "mm"
  var multiplier = "";
  var order      = Blockly.Arduino.ORDER_NONE;

  // multiply by 10: "mm" to "cm"
  var dropdown_unitdistance = block.getFieldValue("UnitDistance");
  if (dropdown_unitdistance === "cm") {
    multiplier = " * 10";
    order      = Blockly.Arduino.ORDER_MULTIPLICATIVE;
  }

  var value_distanceMm = Blockly.Arduino.valueToCode(block, "distanceMm", order);

  var code = "LbMotion.waitDistanceMm(" + value_distanceMm + multiplier + ");\n";
  return code;
};


LeanbotArduino["LbMotion.waitRotationDeg"] = function(block) {
  LeanbotArduino_addHeader();

  var value_rotationDeg = Blockly.Arduino.valueToCode(block, "rotationDeg",  Blockly.Arduino.ORDER_NONE);

  var code = "LbMotion.waitRotationDeg(" + value_rotationDeg + ");\n";
  return code;
};


LeanbotArduino["LbMotion.wait"] = LeanbotArduino["Leanbot.delay"];  // re-ude identical block


LeanbotArduino["LbMotion.stop"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbMotion.stopAndWait();\n";
  return code;
};


LeanbotArduino["LbMotion.setZeroOrigin"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbMotion.setZeroOrigin();\n";
  return code;
};


LeanbotArduino["LbMotion.approxDistance"] = function(block) {
  LeanbotArduino_addHeader();

  var code  = "LbMotion.getDistanceMm()";
  var order = Blockly.Arduino.ORDER_ATOMIC;

  var dropdown_unitdistance = block.getFieldValue("UnitDistance");
  if (dropdown_unitdistance === "cm") {
    code += " / 10";
    order = Blockly.Arduino.ORDER_MULTIPLICATIVE;
  }

  return [code, order];
};


LeanbotArduino["LbMotion.approxHeading"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbMotion.getRotationDeg()";
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


/*==================================================================================================
                                      LbGripper
==================================================================================================*/

LeanbotArduino["LbGripper.help"] = function(block) {
  return "";
};


LeanbotArduino["LbGripper.open"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbGripper.open();\n";
  return code;
};


LeanbotArduino["LbGripper.close"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbGripper.close();\n";
  return code;
};


LeanbotArduino["LbGripper.moveTo"] = function(block) {
  LeanbotArduino_addHeader();

  var value_angleDeg = Blockly.Arduino.valueToCode(block, "angleDeg",  Blockly.Arduino.ORDER_NONE);

  var code = "LbGripper.moveTo(" + value_angleDeg + ");\n";
  return code;
};


LeanbotArduino["LbGripper.moveToLR"] = function(block) {
  LeanbotArduino_addHeader();

  var value_angleDegL = Blockly.Arduino.valueToCode(block, "angleDegL",  Blockly.Arduino.ORDER_NONE);
  var value_angleDegR = Blockly.Arduino.valueToCode(block, "angleDegR",  Blockly.Arduino.ORDER_NONE);
  var value_timeMs    = Blockly.Arduino.valueToCode(block,    "timeMs",  Blockly.Arduino.ORDER_NONE);

  var code = "";

  code += "LbGripper.moveToLR(";

  code += value_angleDegL;
  code += ", ";

  code += value_angleDegR;
  code += ", ";

  code += value_timeMs;
  code += ");\n";

  return code;
};


/*==================================================================================================
                                      LbRGB
==================================================================================================*/

LeanbotArduino["LbRGB.help"] = function(block) {
  return "";
};


LeanbotArduino["LbRGB.colour_picker"] = function(block) {
  // LeanbotArduino_addHeader();

  var code = block.getFieldValue('Color').replace("#", "0x");
  return [code, Blockly.Arduino.ORDER_NONE];
};


LeanbotArduino["LbRGB.colour_random"] = function (block) {
  LeanbotArduino_addHeader();

  // Generate a random colour
  var code = "CRGB(random8(), random8(), random8())";
  return [code, Blockly.Arduino.ORDER_NONE];
};


LeanbotArduino["LbRGB.colour_rgb"] = function (block) {
  LeanbotArduino_addHeader();

  var red   = Blockly.Arduino.valueToCode(block, "Red",   Blockly.Arduino.ORDER_COMMA);
  var green = Blockly.Arduino.valueToCode(block, "Green", Blockly.Arduino.ORDER_COMMA);
  var blue  = Blockly.Arduino.valueToCode(block, "Blue",  Blockly.Arduino.ORDER_COMMA);

  var rgb = (Number(red) << 16) | (Number(green) << 8) | (Number(blue) << 0);
  var code = "0x" + ("000000" + rgb.toString(16)).substr(-6);  // 6-digit

  // var code = "CRGB(" + red + ", " + green + ", " + blue + ")";
  return [code, Blockly.Arduino.ORDER_NONE];
};


LeanbotArduino["LbRGB.colour_blend"] = function (block) {
  // Blend two colours together.
  var c1    = block.getFieldValue("Color1").replace("#", "0x");
  var c2    = block.getFieldValue("Color2").replace("#", "0x");
  var ratio = block.getFieldValue("Ratio");

  var rgb1 = Number(c1);
  var rgb2 = Number(c2);

  var r1 = (rgb1 >>> 16) & 0xFF;
  var g1 = (rgb1 >>>  8) & 0xFF;
  var b1 = (rgb1 >>>  0) & 0xFF;

  var r2 = (rgb2 >>> 16) & 0xFF;
  var g2 = (rgb2 >>>  8) & 0xFF;
  var b2 = (rgb2 >>>  0) & 0xFF;

  var r  = Math.round( r1 * (1 - ratio) + r2 * ratio );
  var g  = Math.round( g1 * (1 - ratio) + g2 * ratio );
  var b  = Math.round( b1 * (1 - ratio) + b2 * ratio );

  var c  = (r << 16) | (g << 8) | (b << 0);

  var code = "0x" + ("000000" + c.toString(16)).substr(-6);  // 6-digit

  return [code, Blockly.Arduino.ORDER_NONE];
};


LeanbotArduino["LbRGB.setColor"] = function(block) {
  LeanbotArduino_addHeader();

  var value_ledX = block.getFieldValue("ledX");
  var colour_rgb = Blockly.Arduino.valueToCode(block, 'RGBColor', Blockly.Arduino.ORDER_NONE);
  colour_rgb = colour_rgb.replace("#", "0x");

  var code = "LbRGB[" + value_ledX + "] = " + colour_rgb + ";\n";
  return code;
};


LeanbotArduino["LbRGB.setColorText"] = function(block) {
  LeanbotArduino_addHeader();

  var value_ledX  = block.getFieldValue("ledX");
  var value_color = block.getFieldValue("RGBColor");

  var code = "LbRGB[" + value_ledX + "] = CRGB::" + value_color + ";\n";
  return code;
};


LeanbotArduino["LbRGB.shape"] = function(block) {
  LeanbotArduino_addHeader();

  const bitMap = [
    "ledO", "ledA", "ledB", "ledC", "ledD", "ledE", "ledF"
  ];

  var code = "BITMAP(";

  for (var i = 0; i < bitMap.length; i++) {
    if (block.getFieldValue( bitMap[i] ) == "TRUE") {
      code += bitMap[i] + ", "
    }
  }

  code = code.substring(0, code.length-2);  // remove last ", "

  code += ")"

  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbRGB.fillColor"] = function(block) {
  LeanbotArduino_addHeader();

  var colour_rgb = Blockly.Arduino.valueToCode(block, 'RGBColor', Blockly.Arduino.ORDER_NONE);
  colour_rgb = colour_rgb.replace("#", "0x");

  var value_bitmask = Blockly.Arduino.valueToCode(block, 'bitmask', Blockly.Arduino.ORDER_NONE);

  var code = "LbRGB.fillColor(" + colour_rgb + ", " + value_bitmask + ");\n";
  return code;
};


LeanbotArduino["LbRGB.clear"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbRGB.clear();\n";
  return code;
};


LeanbotArduino["LbRGB.show"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbRGB.show();\n";
  return code;
};


LeanbotArduino["LbRGB.clear_show"] = function(block) {
  LeanbotArduino_addHeader();

  var body = Blockly.Arduino.statementToCode(this, "Body");
  // remove one level of indentation per line
  body = body.substring(2);
  body = body.replaceAll("\n  ", "\n");

  var code = "";

  code += "LbRGB.clear();\n";
  code += body;
  code += "LbRGB.show();\n";

  return code;
};


/*==================================================================================================
                                      LbIRLine
==================================================================================================*/

LeanbotArduino["LbIRLine.read"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbIRLine.read();\n";
  return code;
};


LeanbotArduino["LbIRLine.value"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbIRLine.value()";
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbIRLine.isBlackDetected"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "LbIRLine.isBlackDetected()";
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbIRLine.state"] = function(block) {
  LeanbotArduino_addHeader();

  var state = 0;
  for (var i = 0; i <= 3; i++) {
    if (block.getFieldValue("state" + i) == "#000000") {
      state |= (1 << (3-i));
    }
  }

  var code = String(state);
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbIRLine.displayOnRGB"] = function(block) {
  LeanbotArduino_addHeader();

  var colour_rgb = Blockly.Arduino.valueToCode(block, "RGBColor", Blockly.Arduino.ORDER_NONE)
  colour_rgb = colour_rgb.replace("#", "0x");

  var code = "LbIRLine.displayOnRGB(" + colour_rgb + ");\n";
  return code;
};


LeanbotArduino["LbIRLine.doManualCalibration"] = function(block) {
  LeanbotArduino_addHeader();

  var value_touchX = block.getFieldValue("touchX");

  var code = "LbIRLine.doManualCalibration(" + value_touchX + ");";

  Blockly.Arduino.setups_["setup_leanbot_lineCalibration"] = code;
  return "";
};


LeanbotArduino["LbIRLine.setThreshold"] = function(block) {
  LeanbotArduino_addHeader();

  var value_th2L  = Blockly.Arduino.valueToCode(block, "th2L",  Blockly.Arduino.ORDER_NONE);
  var value_th0L  = Blockly.Arduino.valueToCode(block, "th0L",  Blockly.Arduino.ORDER_NONE);
  var value_th1R  = Blockly.Arduino.valueToCode(block, "th1R",  Blockly.Arduino.ORDER_NONE);
  var value_th3R  = Blockly.Arduino.valueToCode(block, "th3R",  Blockly.Arduino.ORDER_NONE);

  var code = "LbIRLine.setThreshold(";
  code += value_th2L + ", ";
  code += value_th0L + ", ";
  code += value_th1R + ", ";
  code += value_th3R;
  code += ");";

  Blockly.Arduino.setups_["setup_leanbot_setThreshold"] = code;
  return "";
};


LeanbotArduino["LbIRLine.help"] = function(block) {
  return "";
};


/*==================================================================================================
                                      LbIRLine
==================================================================================================*/

LeanbotArduino["LbTouch.read"] = function(block) {
  LeanbotArduino_addHeader();

  var value_touchX = block.getFieldValue("touchX");

  var code = "(LbTouch.read(" + value_touchX + ") == HIGH)";
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["LbTouch.waitUntil"] = function(block) {
  LeanbotArduino_addHeader();

  var value_touchX = block.getFieldValue("touchX");

  var code = "while (LbTouch.read(" + value_touchX + ") == LOW);\n";   // single toucn
//var code = "while (LbTouch.readBits() != " + value_touchX + ");\n";  // combo touch

  return code;
};


// Blockly.Arduino["LbTouch.onPress"] = function(block) {
//   LeanbotArduino_addHeader();

//   var value_touchX = block.getFieldValue("touchX");

//   var code = "LbTouch.onPress(" + value_touchX + ")";
//   return [code, Blockly.Arduino.ORDER_ATOMIC];
// };


/*==================================================================================================
                                      LbSRF04
==================================================================================================*/

LeanbotArduino["Leanbot.ping"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "Leanbot.pingCm()";

  var dropdown_unitping = block.getFieldValue("UnitPing");
  if (dropdown_unitping === "mm") {
    code = "Leanbot.pingMm()";
  }

  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


LeanbotArduino["Leanbot.objectInFront"] = function(block) {
  LeanbotArduino_addHeader();

  var functionName = "Leanbot.pingCm()";

  var value_distance = Blockly.Arduino.valueToCode(block, "Distance",  Blockly.Arduino.ORDER_NONE);

  // var dropdown_unitping = block.getFieldValue("UnitPing");
  // if (dropdown_unitping === "mm") {
  //   value_distance = value_distance * 10;
  //   functionName = "Leanbot.pingMm()";
  // }

  if ( !isNaN(value_distance) ) {   // input is number
    value_distance = Math.round(value_distance);
  }

  var code = functionName + " <= " + value_distance;
  return [code, Blockly.Arduino.ORDER_RELATIONAL];
};


/*==================================================================================================
                                      LbBuzzer
==================================================================================================*/

LeanbotArduino["LbBuzzer.tone"] = function(block) {
  LeanbotArduino_addHeader();

  var value_freq = Blockly.Arduino.valueToCode(block, "Freq",  Blockly.Arduino.ORDER_NONE);

  var code = "Leanbot.tone(" + value_freq + ");\n";
  return code;
};


LeanbotArduino["Leanbot.toneDuration"] = function(block) {
  LeanbotArduino_addHeader();

  // default as "Hz"
  var multiplier = "";
  var order_freq = Blockly.Arduino.ORDER_NONE

  // multiply by 1000: "Hz" to "kHz"
  var dropdown_unitfreq = block.getFieldValue("UnitFreq");
  if (dropdown_unitfreq === "kHz") {
    multiplier = " * 1000";
    order_freq = Blockly.Arduino.ORDER_MULTIPLICATIVE;
  }

  var value_freq     = Blockly.Arduino.valueToCode(block, "Freq",     order_freq);
  var value_duration = Blockly.Arduino.valueToCode(block, "Duration", Blockly.Arduino.ORDER_NONE);

  var code = "Leanbot.tone(" + value_freq + multiplier + ", " + value_duration + ");\n";
  code += "LbDelay(" + value_duration + ");\n";
  return code;
};


LeanbotArduino["LbBuzzer.noTone"] = function(block) {
  LeanbotArduino_addHeader();

  var code = "Leanbot.noTone();\n";
  return code;
};


/*==================================================================================================
                                      Init
==================================================================================================*/

LeanbotArduino_begin();
