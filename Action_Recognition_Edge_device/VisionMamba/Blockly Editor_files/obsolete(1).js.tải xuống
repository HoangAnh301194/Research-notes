
"use strict";

goog.provide("Blockly.Arduino.obsolete");
goog.require("Blockly.Arduino");


/*==================================================================================================
                                      Code generation
==================================================================================================*/

Blockly.Arduino["LbMission.begin"] = function(block) {
  LeanbotArduino_addHeader();

  var value_missionId = Blockly.Arduino.valueToCode(block, "missionId",  Blockly.Arduino.ORDER_NONE);

  if (value_missionId.length > 2) {  // empty string is `""`
    value_missionId = "F(" + value_missionId + ")";
  } else {
    value_missionId = "";
  }

  var code = "LbMission.begin(" + value_missionId + ");\n";
  return code;
};


Blockly.Arduino["LbSRF04.objectInFront"] = function(block) {
  LeanbotArduino_addHeader();

  var functionName = "Leanbot.pingCm()";

  var value_distance = Blockly.Arduino.valueToCode(block, "Distance",  Blockly.Arduino.ORDER_NONE);

  // var dropdown_unitping = block.getFieldValue("UnitPing");
  // if (dropdown_unitping === "mm") {
  //   value_distance = value_distance * 10;
  //   functionName = "Leanbot.pingMm()";
  // }

  value_distance = Math.round(value_distance);

  var code = "(" + functionName + " <= " + value_distance + ")";
  return [code, Blockly.Arduino.ORDER_ATOMIC];
};


Blockly.Arduino["LbBuzzer.toneDuration"] = function(block) {
  LeanbotArduino_addHeader();

  var value_freq     = Blockly.Arduino.valueToCode(block, "Freq",     Blockly.Arduino.ORDER_NONE);
  var value_duration = Blockly.Arduino.valueToCode(block, "Duration", Blockly.Arduino.ORDER_NONE);

  var dropdown_unitfreq = block.getFieldValue("UnitFreq");
  if (dropdown_unitfreq === "kHz") {
    value_freq = value_freq * 1000;
  }
  value_freq = Math.round(value_freq);

  var code = "Leanbot.tone(" + value_freq + ", " + value_duration + ");\n";
  code += "LbDelay(" + value_duration + ");\n";
  return code;
};


Blockly.Arduino["Unknown_block"] = function(block) {
  return "Unknown\n";
};
