/**
 * @license
 * Copyright 2022 Obsolete
 *
 */

/**
 * @fileoverview Obsolete blocks for Blockly.
 * @author leanbot
 */

"use strict";

goog.provide("Blockly.Constants.obsolete");

goog.require("Blockly.Blocks");
goog.require("Blockly");


/*==================================================================================================
                                      Variables
==================================================================================================*/

var _leanbot_outOfRange = function(value, min, max) {
  return ( (value < min) || (value > max) );
};


/*==================================================================================================
                                      Functions
==================================================================================================*/

Blockly.Blocks["LbMission.begin"] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Mission:");

    this.appendValueInput("missionId")
        .setCheck("String");

    this.appendDummyInput()
        .appendField("Begin");

    // this.appendDummyInput()
    //     .appendField(new Blockly.FieldColour("#0x663366"), "missionColor");

    // this.appendValueInput("missionColor")
    //   .appendField(new Blockly.FieldColour("#00ff00"), "missionColor");

    this.setInputsInline(true);
    this.setPreviousStatement(true);
    this.setNextStatement(true);
    this.setStyle("leanbot_obsolete_blocks");
    this.setTooltip("Start the mission");
    this.setHelpUrl("");
  }
};


Blockly.Blocks["LbSRF04.objectInFront"] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Ping: Object in front is within");

    this.appendValueInput("Distance")
        .setCheck(intCompatibility);

    this.appendDummyInput()
        .appendField(new Blockly.FieldDropdown(
            [
              ["cm", "cm"],
              // ["mm", "mm"],
            ]
          ),
          "UnitPing"
        );

    this.appendDummyInput()
        .appendField("distance ?");

    this.setInputsInline(true);
    // this.setOutput(true, ["Boolean", "Number"]);
    this.setOutput(true, "Boolean");
    this.setStyle("leanbot_obsolete_blocks");
    this.setTooltip("Check if an object is in front?");
    this.setHelpUrl("");
  }
};


Blockly.Blocks["LbBuzzer.toneDuration"] = {
  init: function() {

    this.appendValueInput("Freq")
        .appendField("Sound: Play")
        .setCheck(intCompatibility);

    this.appendDummyInput()
        .appendField(new Blockly.FieldDropdown(
            [
              ["Hz", "Hz"],
              ["kHz", "kHz"],
            ]
          ),
          "UnitFreq"
        );

    this.appendDummyInput()
        .appendField("tone for");

    this.appendValueInput("Duration")
        .setCheck(intCompatibility);

    this.appendDummyInput()
        .appendField(new Blockly.FieldDropdown(
            [
              // ["second", "second"],
              ["milliseconds", "milliseconds"],
            ]
          ),
          "UnitDuration"
        );

    this.setInputsInline(true);
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setStyle("leanbot_obsolete_blocks");
    this.setTooltip("Play sound with frequency Hz in a duration of time");
    this.setHelpUrl("");
  },

  /**
   * Called whenever anything on the workspace changes.
   * It checks/warns if the selected stepper instance has a config block.
   * @this Blockly.Block
   */
  onchange: function() {
    if (!this.workspace) return;  // Block has been deleted.

    var value_freq     = Blockly.Arduino.valueToCode(this, "Freq",     Blockly.Arduino.ORDER_NONE);
    var value_duration = Blockly.Arduino.valueToCode(this, "Duration", Blockly.Arduino.ORDER_NONE);

    var dropdown_unitfreq = this.getFieldValue("UnitFreq");
    if (dropdown_unitfreq === "kHz") {
      value_freq = value_freq * 1000;
    }
    value_freq = Math.round(value_freq);

    if ( _leanbot_outOfRange(value_freq, 0, 65535) ) {
      this.setWarningText("Out of range: 0 ... 65535 (Hz)");
    }
    else if ( _leanbot_outOfRange(value_duration, 0, 65535) ) {
      this.setWarningText("Out of range: 0 ... 65535 (ms)");
    }
    else {
      this.setWarningText(null);    // clear warning
    }
  }
};


Blockly.Blocks["Unknown_block"] = {
  init: function() {
    this.appendDummyInput()
        .appendField("UnknownBlock");

    this.appendValueInput("Unknown_value");

    this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput(""), "Unknown_field");

    this.appendStatementInput("Unknown_statement");

    this.setInputsInline(true);
    this.setPreviousStatement(true);
    this.setNextStatement(true);
  }
};
