/**
 * @license
 * Copyright 2020 Sébastien CANET
 * SPDX-License-Identifier: BSD-3-Clause
 */

'use strict';

goog.provide('Blockly.Msg.leanbot_en');

goog.require('Blockly.Msg');


var LeanbotMsg = {};

/*==================================================================================================
                                      Categories
==================================================================================================*/

LeanbotMsg["Caregory"] = {
  Name       :    "Leanbot",
  Motion     :    "Motion",
  Gripper    :    "Gripper",
  RGBLed     :    "RGB Leds",
  Sound      :    "Sound",
  Sensors    :    "Sensors",
  IRLine     :    "IRLine",
};


/*==================================================================================================
                                      Blocks
==================================================================================================*/

LeanbotMsg["Warning"] = {
  OutOfRange             : "Out of range",
};


LeanbotMsg["Unit"] = {
  Second                 : "seconds",
  Millisecond            : "milliseconds",
  Centimeter             : "cm",
  Millimeter             : "mm",
  Degree                 : "degrees",
  Hertz                  : "Hz",
  Kilohertz              : "kHz",
  StepPerSec             : "steps/sec",
  RevPerMin              : "rpm",
};


LeanbotMsg["Block"] = {

  "Leanbot.commentLine" : {
    message0 : "\u2800 %1",
    tooltip  : "Add a comment line",
    defaults : {
      commentString : "Comment",
    },
  },

  "Leanbot.section" : {
    message0 : "\u2800 %1 %2 %3",
    tooltip  : "A group of codes",
    defaults : {
      commentString : "Comment",
    },
  },

  "Leanbot.whenStarted" : {
    message0 : "When Leanbot started",
    tooltip  : "Start block",
  },

  "Leanbot.delay" : {
    message0 : "Wait for %1 %2",
    tooltip  : "Keep running at current velocities for a given duration of time",
    text     : {"Running": "running", "Turning": "turning"},
  },


  /*---------------------------------------------------------------*/
  "LbMission.beginByTeacher_end" : {
    message0 : "Mission: Begin when %1 combo touched %2 %3 Mission: End",
    tooltip  : "Start the mission, do tasks, then finish",
  },

  "LbMission.beginByServer_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Start the mission, do tasks, then finish",
  },

  "LbMission.beginByName_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Start the mission, do tasks, then finish",
  },

  "LbMission.beginBySelect_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Start the mission, do tasks, then finish",
  },

  "LbMission.missionSelect" : {
    message0 : "%1 %2 %3",
    tooltip  : "Return the selected Mission name",
  },

  "LbMission.beginImmediately_end" : {
    message0 : "Mission: Begin immediately %1 %2 Mission: End",
    tooltip  : "Start the mission immediately without countdown, do tasks, then finish",
  },

  "LbMission.end" : {
    message0 : "Mission: End",
    tooltip  : "Finish the mission",
  },

  "LbMission.elapsedMillis" : {
    message0 : "Mission: elapsed Time in %1",
    tooltip  : "Return the elapsed time since Leanbot started the Mission",
  },

  "Leanbot.DCMotor.setPower" : {
    message0 : "DC Motor: Run %1 at power level %2 %%",
    tooltip  : "Run DC Motor",
    text     : {"Forward": "forward", "Backward": "backward"},
  },


  /*---------------------------------------------------------------*/
  "LbMotion.run": {
    message0 : "Motion: %1 with velocity Left %2 Right %3 %4",
    tooltip  : "Set rotation velocities of both wheels",
    text     : {"Run": "Run", "Turn": "Turn"},
  },

  "LbMotion.waitDistanceMm" : {
    message0 : "Motion: Keep running for %1 %2",
    tooltip  : "Wait until Leanbot has moved by (approximately) distance",
  },

  "LbMotion.waitRotationDeg" : {
    message0 : "Motion: Keep turning for %1 %2",
    tooltip  : "Wait until Leanbot has rotated by (approximately) rotationDeg degrees",
  },

  "LbMotion.wait" : {
    message0 : "Motion: Keep %1 for %2 %3",
    tooltip  : "Keep running at current velocities for a given duration of time",
    text     : {"Running": "running", "Turning": "turning"},
  },

  "LbMotion.stop" : {
    message0 : "Motion: Stop",
    tooltip  : "Stop Leanbot and wait until Leanbot has decelerated to a complete stop",
  },

  "LbMotion.setZeroOrigin" : {
    message0 : "Motion: Reset Distance and Heading",
    tooltip  : "Set current position as origin",
  },

  "LbMotion.approxDistance" : {
    message0 : "Motion: approx. Distance in %1",
    tooltip  : "Get approximate distance from origin",
  },

  "LbMotion.approxHeading" : {
    message0 : "Motion: approx. Heading in %1",
    tooltip  : "Get approximate heading from origin",
  },


  /*---------------------------------------------------------------*/
  "LbGripper.open": {
    message0 : "Gripper: Open",
    tooltip  : "Move gripper arms to open position (both arms at 0 degree position)",
  },

  "LbGripper.close" : {
    message0 : "Gripper: Close",
    tooltip  : "Move gripper arms to close position (both arms at 90 degree position)",
  },

  "LbGripper.moveTo" : {
    message0 : "Gripper: Rotate to %1 %2",
    tooltip  : "Move both gripper arms to angleDeg angle",
  },

  "LbGripper.moveToLR" : {
    message0 : "Gripper: Rotate Left to %1 and Right to %2 %3 in %4 %5",
    tooltip  : "Move both gripper arms to angleDeg angle",
  },


  /*---------------------------------------------------------------*/
  "LbRGB.colour_picker" : {
    message0 : "Color: %1",
    tooltip  : "Choose a color from the palette",
  },

  "LbRGB.colour_random" : {
    message0 : "Color: Random",
    tooltip  : "Choose a color at random",
  },

  "LbRGB.colour_rgb" : {
    message0 : "Color: R %1 G %2 B %3",
    tooltip  : "Create a color with the specified amount of red, green, and blue. All values must be between 0 and 255",
  },

  "LbRGB.colour_blend" : {
    message0 : "Color: Blend %1 and %2 with ratio %3",
    tooltip  : "Blends two colours together with a given ratio [0.0, 1.0]",
  },

  "LbRGB.setColor" : {
    message0 : "RGB Leds: Set %1 to %2",
    tooltip  : "Set ledX to RGBColor color",
    warning  : "Missing color block",
  },

  "LbRGB.setColorText" : {
    message0 : "RGB Leds: Set %1 to %2",
    tooltip  : "Set ledX to RGBColor color",
  },

  "LbRGB.shape" : {
    message0 : "Shape: O %1 A %2 B %3 C %4 D %5 E %6 F %7",
    tooltip  : "Set a shape (set of Leds) to a color",
    warning  : "At least 2 leds must be selected",
  },

  "LbRGB.fillColor" : {
    message0 : "RGB Leds: Fill %1 with %2",
    tooltip  : "Fill a shape (set of Leds) with color",
    warning  : "Missing color block",
  },

  "LbRGB.clear" : {
    message0 : "RGB Leds: Clear",
    tooltip  : "Clear all Leds to black",
  },

  "LbRGB.show" : {
    message0 : "RGB Leds: Display",
    tooltip  : "Showing all Leds to diplay",
  },

  "LbRGB.clear_show" : {
    message0 : "RGB Leds: Clear %1 %2 RGB Leds: Display",
    tooltip  : "Clear all Leds to black and showing all Leds to diplay",
  },


  /*---------------------------------------------------------------*/
  "Leanbot.toneDuration" : {
    message0 : "Sound: Play %1 %2 tone for %3 %4",
    tooltip  : "Play sound with frequency Hz in a duration of time",
  },


  /*---------------------------------------------------------------*/
  "LbTouch.read" : {
    message0 : "Touch: %1 is touched ?",
    tooltip  : "Read the binary state of 4 touch sensors",
  },

  "LbTouch.waitUntil" : {
    message0 : "Touch: wait until %1 touched",
    tooltip  : "Wait until the desired sensor is touched",
  },

  "Leanbot.ping" : {
    message0 : "Ping: Front distance in %1",
    tooltip  : "Measure the front distance",
  },

  "Leanbot.objectInFront" : {
    message0 : "Ping: Object in front is within %1 %2 distance ?",
    tooltip  : "Measure the front distance",
  },


  /*---------------------------------------------------------------*/
  "LbIRLine.doManualCalibration" : {
    message0 : "IRLine: Do manual calibration with %1 button",
    tooltip  : "Do 3-step light level calibration with touchButtonX button",
  },

  "LbIRLine.setThreshold" : {
    message0 : "IRLine: Set threshold  2L %1 0L %2 1R %3 3R %4",
    tooltip  : "Set light level threshold value",
  },

  "LbIRLine.read" : {
    message0 : "IRLine: Read sensors",
    tooltip  : "Read the binary state of 4 IR line sensors",
  },

  "LbIRLine.value" : {
    message0 : "IRLine: Sensor values (2L, 0L, 1R, 3R)",
    tooltip  : "Return the value of 4 IR line sensors which are read before",
  },

  "LbIRLine.isBlackDetected" : {
    message0 : "IRLine: Black is detected ?",
    tooltip  : "Return true if line sensor is on the line",
  },

  "LbIRLine.state" : {
    message0 : "IRLine: %1 %2 %3 %4",
    tooltip  : "Represent the binary state of 4 IR line sensors",
  },

  "LbIRLine.displayOnRGB" : {
    message0 : "IRLine: Display on RGB with %1",
    tooltip  : "Display the line sensors result on RGB Leds with color",
  },

};


/*==================================================================================================
                                      Definition
==================================================================================================*/

Blockly.Msg.Leanbot["en"] = {
  Category      :      LeanbotMsg["Caregory"],
  Warning       :      LeanbotMsg["Warning"],
  Unit          :      LeanbotMsg["Unit"],

  ...LeanbotMsg["Block"],
};
