/**
 * @license
 * Copyright 2012 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Math blocks for Blockly.
 *
 * This file is scraped to extract a .json file of block definitions. The array
 * passed to defineBlocksWithJsonArray(..) must be strict JSON: double quotes
 * only, no outside references, no functions, no trailing commas, etc. The one
 * exception is end-of-line comments, which the scraper will remove.
 * @author q.neutron@gmail.com (Quynh Neutron)
 * @author scanet@libreduc.cc (Sébastien CANET)
 */
'use strict';

goog.provide('Blockly.Constants.MathArduino');

// goog.require('Blockly.Blocks');

Blockly.Blocks['math_single'] = {
    /**
     * Block for advanced math operators with single operand.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": "%1 %2",
            "args0": [
                {
                    "type": "field_dropdown",
                    "name": "OP",
                    "options": [
                        [Blockly.Msg.MATH_SINGLE_OP_ABSOLUTE, 'ABS'],
                        [Blockly.Msg.MATH_SINGLE_OP_SQUARE, 'SQUARE'],
                        [Blockly.Msg.MATH_SINGLE_OP_ROOT, 'ROOT'],
                        // [Blockly.Msg.MATH_ROUND_OPERATOR_ROUND, 'ROUND'],
                        // ['-', 'NEG'],
                        // ['10^', 'POW10']
                    ]
                },
                {
                    "type": "input_value",
                    "name": "NUM",
                    "check": floatCompatibility
                }
            ],
            "output": floatCompatibility,
            "style": "math_blocks",
            "helpUrl": Blockly.Msg.MATH_SINGLE_HELPURL
        });
        // Assign 'this' to a variable for use in the tooltip closure below.
        var thisBlock = this;
        this.setTooltip(function () {
            var mode = thisBlock.getFieldValue('OP');
            var TOOLTIPS = {
                'ROOT': Blockly.Msg.MATH_SINGLE_TOOLTIP_ROOT,
                'ABS': Blockly.Msg.MATH_SINGLE_TOOLTIP_ABS,
                'NEG': Blockly.Msg.MATH_SINGLE_TOOLTIP_NEG,
                'POW10': Blockly.Msg.MATH_SINGLE_TOOLTIP_POW10
            };
            return TOOLTIPS[mode];
        });
    }
};

Blockly.Blocks['math_multiple2'] = {
    /**
     * Block for advanced math operators with two operands.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            message0: "%1 %2 %3",
            args0: [
                {
                    type: "field_dropdown",
                    name: "OP",
                    options: [
                        ['max', 'MAX'],
                        ['min', 'MIN'],
                        ['pow', 'POW'],
                    ]
                },
                { "type": "input_value", "name": "NUM1", "check": intCompatibility },
                { "type": "input_value", "name": "NUM2", "check": intCompatibility },
            ],
            inputsInline: true,
            output: intCompatibility,
            style: "math_blocks"
        });
        // Assign 'this' to a variable for use in the tooltip closure below.
        var thisBlock = this;
        this.setTooltip(function () {
            var mode = thisBlock.getFieldValue('OP');
            var TOOLTIPS = {
                'MAX': Blockly.Msg.MATH_MULTIPLE2_TOOLTIP_MAX,
                'MIN': Blockly.Msg.MATH_MULTIPLE2_TOOLTIP_MIN,
                'POW': Blockly.Msg.MATH_MULTIPLE2_TOOLTIP_POW,
            };
            return TOOLTIPS[mode];
        });
        // Assign 'this' to a variable for use in the helpurl closure below.
        this.setHelpUrl(function () {
            var mode = thisBlock.getFieldValue('OP');
            var HELPURLS = {
                'MAX': Blockly.Msg.MATH_MULTIPLE2_HELPURL_MAX,
                'MIN': Blockly.Msg.MATH_MULTIPLE2_HELPURL_MIN,
                'POW': Blockly.Msg.MATH_MULTIPLE2_HELPURL_POW,
            };
            return HELPURLS[mode];
        });
    }
};

Blockly.Blocks['math_trig'] = {
    /**
     * Block for trigonometry operators.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": "%1 %2 %3",
            "args0": [
                {
                    "type": "field_dropdown",
                    "name": "OP",
                    "options": [
                        [Blockly.Msg.MATH_TRIG_COS, 'COS'],
                        [Blockly.Msg.MATH_TRIG_SIN, 'SIN'],
                        [Blockly.Msg.MATH_TRIG_TAN, 'TAN']
                    ]
                },
                {
                    "type": "input_value",
                    "name": "NUM",
                  //"check": "Number"
                    "check": ["Number", "float"]
                },
                {
                    "type": "field_dropdown",
                    "name": "UNIT",
                    "options": [
                        ["degrees", "degrees"],
                        ["radians", "radians"],
                    ]
                },
            ],
            "output": "float",
            "style": "math_blocks",
            "helpUrl": Blockly.Msg.MATH_TRIG_HELPURL
        });
        // Assign 'this' to a variable for use in the tooltip closure below.
        var thisBlock = this;
        this.setTooltip(function () {
            var mode = thisBlock.getFieldValue('OP');
            var TOOLTIPS = {
                'SIN': Blockly.Msg.MATH_TRIG_TOOLTIP_SIN,
                'COS': Blockly.Msg.MATH_TRIG_TOOLTIP_COS,
                'TAN': Blockly.Msg.MATH_TRIG_TOOLTIP_TAN
            };
            return TOOLTIPS[mode];
        });
    }
};

Blockly.Blocks['math_constant'] = {
    /**
     * Block for constants: PI, E, the Golden Ratio, sqrt(2), 1/sqrt(2), INFINITY.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": "%1",
            "args0": [
                {
                    "type": "field_dropdown",
                    "name": "CONSTANT",
                    "options": [
                        ['\u03c0', 'PI'],
                        ['\u03c0/2', 'HALF_PI'],
                        ['2\u03c0',  'TWO_PI'],
                        // ['\u03c6', 'GOLDEN_RATIO'],
                        // ['sqrt(2)', 'SQRT2'],
                        // ['sqrt(\u00bd)', 'SQRT1_2']
                    ]
                }
            ],
          //"output": "float",
            "output": ["float", "Number"],
            "style": "math_blocks",
            "tooltip": Blockly.Msg.MATH_CONSTANT_TOOLTIP,
            "helpUrl": Blockly.Msg.MATH_CONSTANT_HELPURL
        });
    }
};

Blockly.Blocks['math_number_property'] = {
    /**
     * Block for checking if a number is even, odd, prime, whole, positive,
     * negative or if it is divisible by certain number.
     * @this Blockly.Block
     */
    init: function () {
        var PROPERTIES =
                [[Blockly.Msg.MATH_IS_EVEN, 'EVEN'],
                    [Blockly.Msg.MATH_IS_ODD, 'ODD'],
                    // [Blockly.Msg.MATH_IS_PRIME, 'PRIME'],
                    // [Blockly.Msg.MATH_IS_WHOLE, 'WHOLE'],
                    [Blockly.Msg.MATH_IS_POSITIVE, 'POSITIVE'],
                    [Blockly.Msg.MATH_IS_NEGATIVE, 'NEGATIVE'],
                    [Blockly.Msg.MATH_IS_DIVISIBLE_BY, 'DIVISIBLE_BY']];
        this.setStyle("math_blocks");
        this.appendValueInput('NUMBER_TO_CHECK')
                .setCheck(["float", "Number"]);
        var dropdown = new Blockly.FieldDropdown(PROPERTIES, function (option) {
            var divisorInput = (option == 'DIVISIBLE_BY');
            this.sourceBlock_.updateShape_(divisorInput);
        });
        this.appendDummyInput()
                .appendField(dropdown, 'PROPERTY');
        this.setInputsInline(true);
        this.setOutput(true, 'Boolean');
        this.setTooltip(Blockly.Msg.MATH_IS_TOOLTIP);
    },
    /**
     * Create XML to represent whether the 'divisorInput' should be present.
     * @return {Element} XML storage element.
     * @this Blockly.Block
     */
    mutationToDom: function () {
        var container = document.createElement('mutation');
        var divisorInput = (this.getFieldValue('PROPERTY') == 'DIVISIBLE_BY');
        container.setAttribute('divisor_input', divisorInput);
        return container;
    },
    /**
     * Parse XML to restore the 'divisorInput'.
     * @param {!Element} xmlElement XML storage element.
     * @this Blockly.Block
     */
    domToMutation: function (xmlElement) {
        var divisorInput = (xmlElement.getAttribute('divisor_input') == 'true');
        this.updateShape_(divisorInput);
    },
    /**
     * Modify this block to have (or not have) an input for 'is divisible by'.
     * @param {boolean} divisorInput True if this block has a divisor input.
     * @private
     * @this Blockly.Block
     */
    updateShape_: function (divisorInput) {
        // Add or remove a Value Input.
        var inputExists = this.getInput('DIVISOR');
        if (divisorInput) {
            if (!inputExists) {
                this.appendValueInput('DIVISOR')
                        .setCheck('Number');
            }
        } else if (inputExists) {
            this.removeInput('DIVISOR');
        }
    }
};

Blockly.Blocks['math_change'] = {
    /**
     * Block for adding to a variable in place.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": Blockly.Msg.MATH_CHANGE_TITLE,
            "args0": [
                {
                    "type": "field_variable",
                    "name": "VAR",
                    "variable": Blockly.Msg.MATH_CHANGE_TITLE_ITEM
                },
                {
                    "type": "input_value",
                    "name": "DELTA",
                    "check": "Number"
                }
            ],
            "previousStatement": null,
            "nextStatement": null,
            "colour": Blockly.Blocks.variables.HUE,
            "helpUrl": Blockly.Msg.MATH_CHANGE_HELPURL
        });
        // Assign 'this' to a variable for use in the tooltip closure below.
        var thisBlock = this;
        this.setTooltip(function () {
            return Blockly.Msg.MATH_CHANGE_TOOLTIP.replace('%1',
                    thisBlock.getFieldValue('VAR'));
        });
    }
};

Blockly.Blocks['math_round'] = {
    /**
     * Block for round number.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": "%1 %2",
            "args0": [
                {
                    "type": "field_dropdown",
                    "name": "OP",
                    "options": [
                        [Blockly.Msg.MATH_ROUND_OPERATOR_ROUND, 'round'],
                        // [Blockly.Msg.MATH_ROUND_OPERATOR_ROUNDUP, 'round_up'],
                        // [Blockly.Msg.MATH_ROUND_OPERATOR_ROUNDDOWN, 'round_down'],
                    ]
                },
                {
                    "type": "input_value",
                    "name": "NUM",
                    "check": ["Number", "float"]
                }
            ],
            "output": "Number",
            "style": "math_blocks",
            "tooltip": Blockly.Msg.MATH_ROUND_TOOLTIP,
            "helpUrl": Blockly.Msg.MATH_ROUND_HELPURL
        });
    }
};

Blockly.Blocks['math_modulo'] = {
    /**
     * Block for remainder of a division.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": Blockly.Msg.MATH_MODULO_TITLE,
            "args0": [
                {
                    "type": "input_value",
                    "name": "DIVIDEND",
                    "check": "Number"
                },
                {
                    "type": "input_value",
                    "name": "DIVISOR",
                    "check": "Number"
                }
            ],
            "inputsInline": true,
            "output": "Number",
            "style": "math_blocks",
            "tooltip": Blockly.Msg.MATH_MODULO_TOOLTIP,
            "helpUrl": Blockly.Msg.MATH_MODULO_HELPURL
        });
    }
};

Blockly.Blocks['math_constrain'] = {
    /**
     * Block for constrain number.
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": Blockly.Msg.MATH_CONSTRAIN_TITLE,
            "args0": [
                {
                    "type": "input_value",
                    "name": "VALUE",
                    "check": ["Number", "float"]
                },
                {
                    "type": "input_value",
                    "name": "LOW",
                    "check": ["Number", "float"]
                },
                {
                    "type": "input_value",
                    "name": "HIGH",
                    "check": ["Number", "float"]
                }
            ],
            "inputsInline": true,
            "output": "Number",
            "style": "math_blocks",
            "tooltip": Blockly.Msg.MATH_CONSTRAIN_TOOLTIP,
            "helpUrl": Blockly.Msg.MATH_CONSTRAIN_HELPURL
        });
    }
};

Blockly.Blocks['math_random_int'] = {
    /**
     * Block for random integer between [X] and [Y].
     * @this Blockly.Block
     */
    init: function () {
        this.jsonInit({
            "message0": Blockly.Msg.MATH_RANDOM_INT_TITLE,
            "args0": [
                {
                    "type": "input_value",
                    "name": "FROM",
                    "check": "Number"
                },
                {
                    "type": "input_value",
                    "name": "TO",
                    "check": "Number"
                }
            ],
            "inputsInline": true,
            "output": "int",
            "style": "math_blocks",
            "tooltip": Blockly.Msg.MATH_RANDOM_INT_TOOLTIP,
            "helpUrl": Blockly.Msg.MATH_RANDOM_INT_HELPURL
        });
    }
};

Blockly.Blocks['math_random_seed'] = {
    init: function () {
        this.jsonInit({
            message0: Blockly.Msg.MATH_RANDOM_SEED_TITLE,
            args0: [
                { type: "input_value", name: "SEED", check: intCompatibility }
            ],
            inputsInline: true,
            previousStatement: null,
            nextStatement: null,
            style: "math_blocks",
            tooltip: Blockly.Msg.MATH_RANDOM_SEED_TOOLTIP,
            helpUrl: Blockly.Msg.MATH_RANDOM_SEED_HELPURL
        });
    }
};

Blockly.Blocks['math_map_from'] = {
    init: function () {
        this.jsonInit({
            message0: "%1, such as",
            args0: [
                { type: "input_value", name: "NUM", check: intCompatibility },
            ],
            inputsInline: true,
            output: "math_map_from",
            style: "math_blocks",
        });
    },
    onchange: function() {          // Called whenever anything on the workspace changes
      if (!this.workspace) return;  // Block has been deleted
      // prevent disconnect from parent
      this.setDeletable(false);
      this.setMovable(false);
    }
};

Blockly.Blocks['math_map_to'] = {
    init: function () {
        this.jsonInit({
            message0: "%1 map to %2",
            args0: [
                { type: "input_value", name: "FROM", check: intCompatibility },
                { type: "input_value", name: "TO",   check: intCompatibility },
            ],
            inputsInline: true,
            output: "math_map_to",
            style: "math_blocks",
        });
    },
    onchange: function() {          // Called whenever anything on the workspace changes
      if (!this.workspace) return;  // Block has been deleted
      // prevent disconnect from parent
      this.setDeletable(false);
      this.setMovable(false);
    }
};

Blockly.Blocks['math_map'] = {
    /* JavaScript version */
    // init: function() {
    //     this.appendValueInput("Number")
    //         .setCheck("Number")
    //         .appendField("Mapped value of");
    //     this.appendValueInput("fromLow")
    //         .appendField("such as").setAlign(Blockly.ALIGN_RIGHT)
    //         .setCheck("math_map_to");
    //     this.appendValueInput("fromHigh")
    //         .appendField("and").setAlign(Blockly.ALIGN_RIGHT)
    //         .setCheck("math_map_to");
    //     this.setInputsInline(false);
    //     this.setOutput(true, intCompatibility);
    //     this.setStyle("math_blocks");
    //     this.setTooltip(Blockly.Msg.MATH_MAP_TOOLTIP);
    //     this.setHelpUrl(Blockly.Msg.MATH_MAP_HELPURL);
    // }

    /* JSON version */
    init: function () {
        this.jsonInit({
          //message0: "Mapped value of %1 such as %2 %3",
            message0: "Map from %1 %2 %3",
            args0: [
              //{ type: "input_value", name: "Number",   check: intCompatibility },
                { type: "input_value", name: "Number",   check: "math_map_from" },
                { type: "input_value", name: "fromLow",  check: "math_map_to", "align": "RIGHT" },
                { type: "input_value", name: "fromHigh", check: "math_map_to" }
            ],
            inputsInline: false,
            output: intCompatibility,
            style: "math_blocks",
            tooltip: Blockly.Msg.MATH_MAP_TOOLTIP,
            helpUrl: Blockly.Msg.MATH_MAP_HELPURL
        });
    }
};