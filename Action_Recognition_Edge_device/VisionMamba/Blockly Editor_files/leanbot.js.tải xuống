/**
 * @license
 * Copyright 2022 Leanbot
 *
 */

/**
 * @fileoverview LEANBOT blocks for Blockly.
 * @author leanbot
 */

"use strict";

goog.provide("Blockly.Constants.leanbot");

goog.require("Blockly.Blocks");
goog.require("Blockly");


/*==================================================================================================
                                      Variables
==================================================================================================*/

var LeanbotBlocks = {};


const LeanbotMediaFolder = "./blocklyduino/blocks/leanbot/media/";


const Leanbot_dropDown = {
  "mission" : [
    ["msNumberGrid", "msNumberGrid"],
    ["SWRP3.M04",    "SWRP3.M04"],
    ["ms1.1",        "ms1.1"],
    ["ms11.1",       "ms11.1"],
    ["ms11.2",       "ms11.2"],
    ["ms12.1",       "ms12.1"],
  ],

  "touch" : [
    ["TB1A", "TB1A"],
    ["TB1B", "TB1B"],
    ["TB2A", "TB2A"],
    ["TB2B", "TB2B"],
  ],

  "ledRGB" : [
    ["ledO", "ledO"],
    ["ledA", "ledA"],
    ["ledB", "ledB"],
    ["ledC", "ledC"],
    ["ledD", "ledD"],
    ["ledE", "ledE"],
    ["ledF", "ledF"],
  ],

};


/*==================================================================================================
                                      Functions
==================================================================================================*/

var Leanbot_checkOutOfRange_language = function(block, language, ...checks) {
  for (const c of checks) {
    const value = c[0];
    const min   = c[1];
    const max   = c[2];
    if ( (value < min) || (value > max) ) {
      block.setWarningText(Blockly.Msg.Leanbot[language].Warning.OutOfRange + ": [" + min + ", " + max + "]");
      return;
    }
  }

  block.setWarningText(null);    // clear warning
};


var Leanbot_getMsgLanguage = function(language) {
  return Blockly.Msg.Leanbot[language];
};


var LeanbotBlocks_begin = function() {
  for (const [id, block] of Object.entries(LeanbotBlocks)) {    // loop through each pair [key, value]
    // console.log(id, block);
    // Generate block for this id (only add suffix for non-English blocks)
    Blockly.Blocks[id]         = block(id, Code.LANG);
    Blockly.Blocks[id + "_vi"] = block(id, "vi");
  }
};


/*==================================================================================================
                                      General
==================================================================================================*/

LeanbotBlocks["Leanbot.commentLine"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_input", name: "commentString", text: msgCfg.defaults.commentString},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_comment_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["Leanbot.section"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_input", name: "commentString", text: msgCfg.defaults.commentString},
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_comment_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["Leanbot.whenStarted"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        nextStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["Leanbot.delay"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "Delay", check: intCompatibility},
          {type: "field_dropdown", name: "Unit", options: [ [msgLang.Unit.Second, "seconds"], [msgLang.Unit.Millisecond, "milliseconds"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_delay = Blockly.Arduino.valueToCode(this, "Delay",  Blockly.Arduino.ORDER_NONE);

      var dropdown_unit = this.getFieldValue("Unit");
      if (dropdown_unit === "seconds") {
        value_delay = value_delay * 1000;
      }

      Leanbot_checkOutOfRange_language(this, lang, [value_delay, 0, 65535]);
    }
  };

  return block;
};


LeanbotBlocks["Leanbot.DCMotor.setPower"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "Direction", options: [ [msgCfg.text["Forward"], "Forward"], [msgCfg.text["Backward"], "Backward"] ] },
          {type: "input_value", name: "Power", check: intCompatibility},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_power = Blockly.Arduino.valueToCode(this, "Power",  Blockly.Arduino.ORDER_NONE);
      Leanbot_checkOutOfRange_language(this, lang, [value_power, 0, 100]);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbMission
==================================================================================================*/

LeanbotBlocks["LbMission.beginByTeacher_end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "comboTouch", options: [ ["TB1A + TB1B", "TB1A + TB1B"], ["TB1A + TB2B", "TB1A + TB2B"], ["TB2A + TB1B", "TB2A + TB1B"] ] },
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.beginByServer_end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "missionId", options: Leanbot_dropDown.mission },
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.beginByName_end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_input", name: "missionId", text: "name", spellcheck: false},
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.beginBySelect_end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value", name: "missionId", check: stringCompatibility},
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.missionSelect"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const quoteImgCfg = Blockly.Constants.Text.QUOTE_IMAGE_MIXIN;

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_image", src: quoteImgCfg.QUOTE_IMAGE_LEFT_DATAURI, width: quoteImgCfg.QUOTE_IMAGE_WIDTH, height: quoteImgCfg.QUOTE_IMAGE_HEIGHT, alt: "\u201c"},
          {type: "field_dropdown", name: "missionSelect", options: Leanbot_dropDown.mission },
          {type: "field_image", src: quoteImgCfg.QUOTE_IMAGE_RIGHT_DATAURI, width: quoteImgCfg.QUOTE_IMAGE_WIDTH, height: quoteImgCfg.QUOTE_IMAGE_HEIGHT, alt: "\u201d"},
        ],
        output: stringCompatibility,
        style: "leanbot_general_expression_blocks",
        // extensions: ["text_quotes"],  // display quotes
      };

      this.jsonInit(jsonCfg);
    },
  };

  return block;
};


LeanbotBlocks["LbMission.beginImmediately_end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.end"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        style: "leanbot_general_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMission.elapsedMillis"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "Unit", options: [ [msgLang.Unit.Second, "seconds"], [msgLang.Unit.Millisecond, "milliseconds"] ] }
        ],
        inputsInline: true,
        output: intCompatibility,
        style: "leanbot_general_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    },
  };

  return block;
};


/*==================================================================================================
                                      LbMotion
==================================================================================================*/

LeanbotBlocks["LbMotion.run"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {

      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "MotionType", options: [ [msgCfg.text["Run"], "Run"], [msgCfg.text["Turn"], "Turn"] ] },
          {type: "input_value",    name: "Left",  check: intCompatibility},
          {type: "input_value",    name: "Right", check: intCompatibility},
          {type: "field_dropdown", name: "UnitSpeed", options: [ [msgLang.Unit.RevPerMin, "rpm"], [msgLang.Unit.StepPerSec, "sps"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {  // Called whenever anything on the workspace changes.
      if (!this.workspace) return;  // Block has been deleted.

      const value_left  = Blockly.Arduino.valueToCode(this, "Left",  Blockly.Arduino.ORDER_NONE);
      const value_right = Blockly.Arduino.valueToCode(this, "Right", Blockly.Arduino.ORDER_NONE);

      let maxRange = 2000;  // steps
      if (this.getFieldValue("UnitSpeed") === "rpm") {
        maxRange = Math.round(maxRange / 2037.8864 * 60);
      }

      Leanbot_checkOutOfRange_language(this, lang,
        [value_left,  -maxRange, +maxRange],
        [value_right, -maxRange, +maxRange]
      );
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.waitDistanceMm"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "distanceMm", check: intCompatibility},
          {type: "field_dropdown", name: "UnitDistance", options: [ [msgLang.Unit.Centimeter, "cm"], [msgLang.Unit.Millimeter, "mm"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {  // Called whenever anything on the workspace changes.
      if (!this.workspace) return;  // Block has been deleted.

      let value_distanceMm = Blockly.Arduino.valueToCode(this, "distanceMm", Blockly.Arduino.ORDER_NONE);
      if (this.getFieldValue("UnitDistance") === "cm") {
        value_distanceMm *= 10;
      }
      value_distanceMm = Math.round(value_distanceMm);

      Leanbot_checkOutOfRange_language(this, lang, [value_distanceMm, -32768, +32768]);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.waitRotationDeg"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "rotationDeg", check: intCompatibility},
          {type: "field_dropdown", name: "UnitDegree", options: [[msgLang.Unit.Degree, "degrees"]]}
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {  // Called whenever anything on the workspace changes.
      if (!this.workspace) return;  // Block has been deleted.

      var value_rotationDeg = Blockly.Arduino.valueToCode(this, "rotationDeg", Blockly.Arduino.ORDER_NONE);
      Leanbot_checkOutOfRange_language(this, lang, [value_rotationDeg, -3600, +3600]);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.wait"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {

      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "MotionType", options: [ [msgCfg.text["Running"], "Running"], [msgCfg.text["Turning"], "Turning"] ] },
          {type: "input_value",    name: "Delay", check: intCompatibility},
          {type: "field_dropdown", name: "Unit", options: [ [msgLang.Unit.Second, "seconds"], [msgLang.Unit.Millisecond, "milliseconds"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {  // Called whenever anything on the workspace changes.
      if (!this.workspace) return;  // Block has been deleted.

      var value_delay = Blockly.Arduino.valueToCode(this, "Delay",  Blockly.Arduino.ORDER_NONE);

      var dropdown_unit = this.getFieldValue("Unit");
      if (dropdown_unit === "seconds") {
        value_delay = value_delay * 1000;
      }

      Leanbot_checkOutOfRange_language(this, lang, [value_delay, 0, 65535]);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.stop"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.setZeroOrigin"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_motion_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.approxDistance"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "UnitDistance", options: [ [msgLang.Unit.Millimeter, "mm"], [msgLang.Unit.Centimeter, "cm"] ] }
        ],
        inputsInline: true,
        output: "Number",
        style: "leanbot_motion_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbMotion.approxHeading"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "UnitDegree", options: [[msgLang.Unit.Degree, "degrees"]]}
        ],
        inputsInline: true,
        output: "Number",
        style: "leanbot_motion_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbGripper
==================================================================================================*/

LeanbotBlocks["LbGripper.help"] = function(id, lang) {
  const block = {
    init: function() {
      const jsonCfg = {
        message0: "%1",
        args0: [
          {type: "field_image", src: LeanbotMediaFolder + "LbGripper.svg", width: 500, height: 200, alt: "*"}
        ],
        inputsInline: true,
        style: "leanbot_comment_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbGripper.open"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_gripper_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbGripper.close"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_gripper_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbGripper.moveTo"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "angleDeg", check: intCompatibility},
          {type: "field_dropdown", name: "UnitDegree", options: [ [msgLang.Unit.Degree, "degrees"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_gripper_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_angleDeg = Blockly.Arduino.valueToCode(this, "angleDeg", Blockly.Arduino.ORDER_NONE);
      Leanbot_checkOutOfRange_language(this, lang, [value_angleDeg, -30, +120]);
    }
  };

  return block;
};


LeanbotBlocks["LbGripper.moveToLR"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "angleDegL", check: intCompatibility},
          {type: "input_value",    name: "angleDegR", check: intCompatibility},
          {type: "field_dropdown", name: "UnitDegree", options: [ [msgLang.Unit.Degree, "degrees"] ] },
          {type: "input_value",    name: "timeMs",    check: intCompatibility},
          {type: "field_dropdown", name: "UnitDuration", options: [ [msgLang.Unit.Millisecond, "milliseconds"] ] }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_gripper_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_angleDeg = Blockly.Arduino.valueToCode(this, "angleDeg", Blockly.Arduino.ORDER_NONE);
      Leanbot_checkOutOfRange_language(this, lang, [value_angleDeg, -30, +120]);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbRGB
==================================================================================================*/

LeanbotBlocks["LbRGB.help"] = function(id, lang) {
  const block = {
    init: function() {
      const jsonCfg = {
        message0: "%1",
        args0: [
          {type: "field_image", src: LeanbotMediaFolder + "LbRGB.png", width: 128, height: 128, alt: "*"}
        ],
        inputsInline: true,
        style: "leanbot_comment_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.colour_picker"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_colour", name: "Color", colour: "#ff0000"},
        ],
        inputsInline: true,
        output: null,
        style: "leanbot_rgbled_expression_blocks",
        helpUrl: "https://en.wikipedia.org/wiki/Color",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.colour_random"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        output: null,
        style: "leanbot_rgbled_expression_blocks",
        helpUrl: "http://randomcolour.com",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.colour_rgb"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value", name: "Red",   check: intCompatibility},
          {type: "input_value", name: "Green", check: intCompatibility},
          {type: "input_value", name: "Blue",  check: intCompatibility},
        ],
        inputsInline: true,
        output: null,
        style: "leanbot_rgbled_expression_blocks",
        helpUrl: "https://www.december.com/html/spec/colorpercompact.html",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_red   = Blockly.Arduino.valueToCode(this, "Red",   Blockly.Arduino.ORDER_NONE);
      var value_green = Blockly.Arduino.valueToCode(this, "Green", Blockly.Arduino.ORDER_NONE);
      var value_blue  = Blockly.Arduino.valueToCode(this, "Blue",  Blockly.Arduino.ORDER_NONE);

      Leanbot_checkOutOfRange_language(this, lang,
        [value_red,   0, 255],
        [value_green, 0, 255],
        [value_blue,  0, 255]
      );
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.colour_blend"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_colour", name: "Color1", colour: "#00ff00"},
          {type: "field_colour", name: "Color2", colour: "#0000ff"},
          {type: "field_number", name: "Ratio", value: 0.5, min: 0.0, max: 1.0}
        ],
        inputsInline: true,
        output: null,
        style: "leanbot_rgbled_expression_blocks",
        helpUrl: "https://meyerweb.com/eric/tools/color-blend/#:::rgbp",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.setColor"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "ledX", options: Leanbot_dropDown.ledRGB},
          {type: "input_value",    name: "RGBColor", check: "Colour"},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var colour_rgb = Blockly.Arduino.valueToCode(this, 'RGBColor', Blockly.Arduino.ORDER_NONE);
      if (colour_rgb === "") {
        this.setWarningText( msgCfg.warning );
      } else {
        this.setWarningText(null);    // clear warning
      }
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.setColorText"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];
  const helpUrl = "https://github.com/FastLED/FastLED/wiki/Pixel-reference#predefined-colors-list";

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "ledX", options: Leanbot_dropDown.ledRGB},
          {type: "field_input",    name: "RGBColor", text: "Yellow", spellcheck: false},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        helpUrl: helpUrl,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);

      var fieldHelper = new Blockly.FieldDropdown([ ["?", "Help"], [helpUrl, "HelpUrl"] ], this.validate);
      this.appendDummyInput()
        .appendField(fieldHelper, 'Helper');
    },

    validate: function(newValue) {
      if (newValue == 'HelpUrl') {
        window.open(helpUrl);
      }
      return 'Help';  // always keep Help option
    },
  };

  return block;
};


LeanbotBlocks["LbRGB.shape"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_checkbox", name: "ledO", checked: true},
          {type: "field_checkbox", name: "ledA", checked: true},
          {type: "field_checkbox", name: "ledB", checked: true},
          {type: "field_checkbox", name: "ledC", checked: true},
          {type: "field_checkbox", name: "ledD", checked: true},
          {type: "field_checkbox", name: "ledE", checked: true},
          {type: "field_checkbox", name: "ledF", checked: true},
        ],
        inputsInline: true,
        output: null,
        style: "leanbot_rgbled_expression_blocks",
        helpUrl: "",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      const bitMap = [
        "ledO", "ledA", "ledB", "ledC", "ledD", "ledE", "ledF"
      ];

      var ledCounter = 0;

      for (var i = 0; i < bitMap.length; i++) {
        if (this.getFieldValue( bitMap[i] ) == "TRUE") {
          ledCounter++;
        }
      }

      if (ledCounter < 2) {
        this.setWarningText( msgCfg.warning );
      } else {
        this.setWarningText(null);    // clear warning
      }
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.fillColor"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value", name: "bitmask",  check: intCompatibility},
          {type: "input_value", name: "RGBColor", check: "Colour"},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var colour_rgb = Blockly.Arduino.valueToCode(this, 'RGBColor', Blockly.Arduino.ORDER_NONE);
      if (colour_rgb === "") {
        this.setWarningText( msgCfg.warning );
      } else {
        this.setWarningText(null);    // clear warning
      }
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.clear"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.show"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbRGB.clear_show"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_dummy"},
          {type: "input_statement", name: "Body"},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_rgbled_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbIRLine
==================================================================================================*/

LeanbotBlocks["LbIRLine.doManualCalibration"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "touchX", options: Leanbot_dropDown.touch }
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_irline_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.setThreshold"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value", name: "th2L", check: intCompatibility},
          {type: "input_value", name: "th0L", check: intCompatibility},
          {type: "input_value", name: "th1R", check: intCompatibility},
          {type: "input_value", name: "th3R", check: intCompatibility},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_irline_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.read"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_irline_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.value"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        output: "Number",
        style: "leanbot_irline_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.isBlackDetected"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [],
        inputsInline: true,
        output: "Boolean",
        style: "leanbot_irline_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.state"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const lineColorCfg = {
        type          : "field_colour",
        colourOptions : ["#ffffff",      "#000000"   ],
        colourTitles  : ["White ground", "Black line"],
        columns       : 2
      };

      const jsonCfg = {
        ...msgCfg,
        args0: [
          {name: "state0", ...lineColorCfg},
          {name: "state1", ...lineColorCfg},
          {name: "state2", ...lineColorCfg},
          {name: "state3", ...lineColorCfg},
        ],
        inputsInline: true,
        output: "Number",
        style: "leanbot_irline_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.displayOnRGB"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value", name: "RGBColor", check:"Colour"},
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_irline_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbIRLine.help"] = function(id, lang) {
  const block = {
    init: function() {
      const jsonCfg = {
        message0: "%1",
        args0: [
          {type: "field_image", src: LeanbotMediaFolder + "LbIRLine.svg", width: 500, height: 200, alt: "*"}
        ],
        inputsInline: true,
        style: "leanbot_comment_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbIRLine
==================================================================================================*/

LeanbotBlocks["LbTouch.read"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "touchX", options: Leanbot_dropDown.touch}
        ],
        inputsInline: true,
        output: ["Boolean", "Number"],
        style: "leanbot_sensors_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["LbTouch.waitUntil"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "touchX", options: Leanbot_dropDown.touch}
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_sensors_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


// Blockly.Blocks["LbTouch.onPress"] = {
//   init: function() {
//     this.appendDummyInput()
//         .appendField("On press");

//     this.appendDummyInput()
//         .appendField(
//           new Blockly.FieldDropdown(
//             Leanbot_dropDown.touch
//           ),
//           "touchX"
//         );

//     this.setInputsInline(true);
//     this.setOutput(true, ["Boolean", "Number"]);
//     this.setColour(0);
//     this.setTooltip("Check the on-press event of 4 touch sensors");
//     this.setHelpUrl("");
//   }
// };


/*==================================================================================================
                                      LbSRF04
==================================================================================================*/

LeanbotBlocks["Leanbot.ping"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "field_dropdown", name: "UnitPing", options: [ [msgLang.Unit.Centimeter, "cm"], [msgLang.Unit.Millimeter, "mm"] ] }
        ],
        inputsInline: true,
        output: "Number",
        style: "leanbot_sensors_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


LeanbotBlocks["Leanbot.objectInFront"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "Distance", check: intCompatibility},
          {type: "field_dropdown", name: "UnitPing", options: [ [msgLang.Unit.Centimeter, "cm"] ] }
        ],
        inputsInline: true,
        output: ["Boolean", "Number"],
        style: "leanbot_sensors_expression_blocks",
      };

      this.jsonInit(jsonCfg);
    }
  };

  return block;
};


/*==================================================================================================
                                      LbBuzzer
==================================================================================================*/

// Blockly.Blocks["LbBuzzer.tone"] = {
//   init: function() {

//     this.appendValueInput("Freq")
//         .appendField("Sound: Play")
//         .setCheck(intCompatibility);

//     this.appendDummyInput()
//         .appendField("Hz tone for");

//     this.setInputsInline(true);
//     this.setPreviousStatement(true, null);
//     this.setNextStatement(true, null);
//     this.setColour(0);
//     this.setTooltip("Play sound with frequency Hz");
//     this.setHelpUrl("");
//   },

//   onchange: function() {
//     if (!this.workspace) return;  // Block has been deleted.

//     var value_freq = Blockly.Arduino.valueToCode(this, "Freq", Blockly.Arduino.ORDER_NONE);
//     Leanbot_checkOutOfRange(this, [value_freq, 0, 65535]);
//   }
// };


LeanbotBlocks["Leanbot.toneDuration"] = function(id, lang) {
  const msgLang = Leanbot_getMsgLanguage(lang);
  const msgCfg  = msgLang[id];

  const block = {
    init: function() {
      const jsonCfg = {
        ...msgCfg,
        args0: [
          {type: "input_value",    name: "Freq", check: intCompatibility},
          {type: "field_dropdown", name: "UnitFreq", options: [ [msgLang.Unit.Hertz, "Hz"], [msgLang.Unit.Kilohertz, "kHz"] ] },
          {type: "input_value",    name: "Duration", check: intCompatibility},
          {type: "field_dropdown", name: "UnitDuration", options: [ [msgLang.Unit.Millisecond, "milliseconds"] ] },
        ],
        inputsInline: true,
        previousStatement: null,
        nextStatement: null,
        style: "leanbot_sound_blocks",
      };

      this.jsonInit(jsonCfg);
    },

    onchange: function() {
      if (!this.workspace) return;  // Block has been deleted.

      var value_freq     = Blockly.Arduino.valueToCode(this, "Freq",     Blockly.Arduino.ORDER_NONE);
      var value_duration = Blockly.Arduino.valueToCode(this, "Duration", Blockly.Arduino.ORDER_NONE);

      var dropdown_unitfreq = this.getFieldValue("UnitFreq");
      if (dropdown_unitfreq === "kHz") {
        value_freq = value_freq * 1000;
      }
      value_freq = Math.round(value_freq);

      Leanbot_checkOutOfRange_language(this, lang,
        [value_freq,     0, 65535],
        [value_duration, 0, 65535]
      );
    }
  };

  return block;
};


// Blockly.Blocks["LbBuzzer.noTone"] = {
//   init: function() {
//     this.appendDummyInput()
//         .appendField("Stop playing sound");

//     this.setInputsInline(true);
//     this.setPreviousStatement(true, null);
//     this.setNextStatement(true, null);
//     this.setColour(0);
//     this.setTooltip("Stop playing sound");
//     this.setHelpUrl("");
//   }
// };


/*==================================================================================================
                                      Init
==================================================================================================*/

LeanbotBlocks_begin();
