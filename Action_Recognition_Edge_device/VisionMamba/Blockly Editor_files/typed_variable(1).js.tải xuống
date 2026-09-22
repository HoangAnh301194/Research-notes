/**
 * @license
 * Copyright 2020 Sébastien CANET
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @fileoverview Typed variable blocks for Blockly.
 * @author scanet@libreduc.cc (Sébastien CANET)
 */

'use strict';

goog.provide('Blockly.Constants.VariablesTyped');

goog.require('Blockly.Arduino');


/*==================================================================================================
                                      Global variables
==================================================================================================*/
// supported types
const Vars_ListConfig = ['global', 'int', 'float', 'String', 'bool'];

const Vars_DropdownConfig = [
  // display  , generate
    ['Auto'   , 'auto'  ],
    ['Number' , 'int'   ],
    ['Float'  , 'float' ],
    ['Text'   , 'String'],
    ['Bool'   , 'bool'  ],
];

// default shadow values for each type
const Vars_DefaultShadow = {
    'auto'  : '<xml><shadow type="math_number"><field name="NUM">0</field></shadow></xml>',
    'int'   : '<xml><shadow type="math_number"><field name="NUM">0</field></shadow></xml>',
    'float' : '<xml><shadow type="math_number"><field name="NUM">1.23</field></shadow></xml>',
    'String': '<xml><shadow type="text"><field name="TEXT">some text</field></shadow></xml>',
    'bool'  : '<xml><shadow type="logic_boolean"><field name="BOOL">FALSE</field></shadow></xml>',
};


/*==================================================================================================
                                      Local functions
==================================================================================================*/
/**
 * Get variable name in DB
 */
var Vars_getVarName = function(block, varField) {
    // Add this line to fix issue: "Deprecated call to Blockly.Names.prototype.getName without defining a variable map"
    Blockly.Arduino.variableDB_.setVariableMap(block.workspace.getVariableMap());
    // Get the variable name
    return Blockly.Arduino.variableDB_.getName(varField, Blockly.Variables.NAME_TYPE);
}


/**
 * Check if the block is inside of a variable declaration block, if so, create an error text on the block
 */ 
var Vars_checkVariableScope = function(block, fieldType) {
    // wait until the user is done dragging to check validity
    if (block.workspace.isDragging()) { return; }

    // get the current variable name
    const varName = Vars_getVarName(block, block.getFieldValue(fieldType));
    // console.log("varName ", varName);

    let foundDefinition = false;
    let upperBlock      = block.getParent();                              // start searching from the nearest block

    while ( (upperBlock !== null) && (foundDefinition === false) ) {      // loop until the root block or the definition is found
        // console.log("Current=", upperBlock.type);

        switch (upperBlock.type) {
            // found a local variable declaration block, check if it is declaring us
            case "vars_initialize_local":
            case "vars_initialize_local_auto":
                if ( (fieldType === "VAR_SET_INT") || (upperBlock !== block.getParent()) ) { // exclude its parent in case of getter "VAR_GET_INT"
                    foundDefinition = ( varName === Vars_getVarName(upperBlock, upperBlock.getFieldValue("VAR_SET_INT")) );
                }
                break;

            // found a function, check if we are in its parameter list
            case "procedures_defreturn":
            case "procedures_defnoreturn":
                for (let i = 0; (i < upperBlock.arguments_.length) && (foundDefinition === false); i++) {
                    foundDefinition = ( varName === Vars_getVarName(upperBlock, upperBlock.arguments_[i]) );
                }
                break;

            // found a for loop, check if it is surrounding us
            case "controls_for":
                let surroundParent = block.getSurroundParent();
                while (surroundParent !== null) {
                    if (surroundParent.id === upperBlock.id) {
                        foundDefinition = ( varName === Vars_getVarName(surroundParent, surroundParent.getFieldValue("VAR")) );
                        break;
                    }
                    surroundParent = surroundParent.getSurroundParent();  // get next surrounding parent
                }
                break;
        }

        upperBlock = upperBlock.getParent();                              // keep moving up the chain
    }

    // check result to set or clear warning
    block.setWarningText( (foundDefinition) ? (null) : ("Out of scope variable!"));
};


/**
 * Set the shadow as input block according to the sellected type
 */ 
var Vars_setShadowInput = function(block, type) {
    // wait until the user is done dragging to check validity
    if (block.workspace.isDragging()) { return; }

    var connection = block.getInput('VALUE').connection;
    if (connection.isConnected()) {
        var inputBlock = connection.targetConnection.getSourceBlock();
        if (inputBlock.isShadow()) { inputBlock.dispose(false); }  // remove current shadow block
        else                       { return;                    }  // do not touch normal block from user
    }

    // only overwrite shadow block, or there is no connection
    const text = Vars_DefaultShadow[type];
    connection.setShadowDom( Blockly.Xml.textToDom(text).firstChild );
};


/*==================================================================================================
                                      BLocks
==================================================================================================*/

/*-----------------------------------------------------------------*/
// global
Blockly.Blocks['vars_initialize_global_auto'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 initialize global")
                .appendField(new Blockly.FieldVariable("", null, Vars_ListConfig, 'global'), "VAR_SET_GLOBAL")
                .appendField("to")
                .setCheck(null);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_initialize_global'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 initialize global")
                .appendField(new Blockly.FieldVariable("", null, Vars_ListConfig, 'global'), "VAR_SET_GLOBAL")
                .appendField("as")
                .appendField(new Blockly.FieldDropdown(Vars_DropdownConfig, this.updateShape_.bind(this)), "VAR_TYPE")
                .appendField("to")
                .setCheck(null);
        this.updateShape_( this.getFieldValue('VAR_TYPE') );
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    },

    updateShape_: function(type) {
        Vars_setShadowInput(this, type)
    }
};


Blockly.Blocks['vars_set_global'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 set")
                .appendField(new Blockly.FieldVariable("", null, Vars_ListConfig, 'global'), "VAR_SET_GLOBAL")
                .appendField("to")
                .setCheck(null);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};


Blockly.Blocks['vars_get_global'] = {
    init: function () {
        this.appendDummyInput()
            .appendField(new Blockly.FieldVariable("", null, Vars_ListConfig, 'global'), "VAR_GET_GLOBAL");
        this.setOutput(true);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};


/*-----------------------------------------------------------------*/
/// local
Blockly.Blocks['vars_initialize_local_auto'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 initialize local")
                .appendField(new Blockly.FieldVariable("", null, [''], ''), "VAR_SET_INT")   // blank type
                .appendField("to")
                .setCheck(null);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_initialize_local'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 initialize local")
                .appendField(new Blockly.FieldVariable("", null, [''], ''), "VAR_SET_INT")   // blank type
                .appendField("as")
                .appendField(new Blockly.FieldDropdown(Vars_DropdownConfig, this.updateShape_.bind(this)), "VAR_TYPE")
                .appendField("to")
                .setCheck(null);
        this.updateShape_( this.getFieldValue('VAR_TYPE') );
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        // this.setStyle('leanbot_localparam_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    },

    updateShape_: function(type) {
        Vars_setShadowInput(this, type)
    }
};

Blockly.Blocks['vars_set_int_local'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 set")
                .appendField(new Blockly.FieldVariable("", null, [''], ''), "VAR_SET_INT")
                .appendField("to")
                .setCheck(null);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        // this.setStyle('leanbot_localparam_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    },
    onchange: function() {
        Vars_checkVariableScope(this, 'VAR_SET_INT');  // error check for setter
    }
};

Blockly.Blocks['vars_get_int_local'] = {
    init: function () {
        this.appendDummyInput()
            .appendField(new Blockly.FieldVariable("", null, [''], ''), "VAR_GET_INT");
        this.setOutput(true);
        this.setStyle('variable_blocks');
        // this.setStyle('leanbot_localparam_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    },
    onchange: function() {
        Vars_checkVariableScope(this, 'VAR_GET_INT');  // error check for getter
    }
};


/*-----------------------------------------------------------------*/
/// int
Blockly.Blocks['vars_set_int'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 set")
                .appendField(new Blockly.FieldVariable("", null, ['int'], 'int'), "VAR_SET_INT")
                .appendField("to")
                .setCheck(intCompatibility);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_get_int'] = {
    init: function () {
        this.appendDummyInput()
                .appendField(new Blockly.FieldVariable("", null, ['int'], 'int'), "VAR_GET_INT");
        this.setOutput(true, ["int", "Number"]);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};


/*-----------------------------------------------------------------*/
/// float
Blockly.Blocks['vars_set_float'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔢 set")
                .appendField(new Blockly.FieldVariable("", null, ['float'], 'float'), "VAR_SET_FLOAT")
                .appendField("to")
                .setCheck(floatCompatibility);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_get_float'] = {
    init: function () {
        this.appendDummyInput()
                .appendField(new Blockly.FieldVariable("", null, ['float'], 'float'), "VAR_GET_FLOAT");
        this.setOutput(true, "float");
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};


/*-----------------------------------------------------------------*/
/// string
Blockly.Blocks['vars_set_string'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🔤 set")
                .appendField(new Blockly.FieldVariable("", null, ['String'], 'String'), "VAR_SET_STRING")
                .appendField("to")
                .setCheck(stringCompatibility);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_get_string'] = {
    init: function () {
        this.appendDummyInput()
                .appendField(new Blockly.FieldVariable("", null, ['String'], 'String'), "VAR_GET_STRING");
        this.setOutput(true, "String");
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};


/*-----------------------------------------------------------------*/
/// bool
Blockly.Blocks['vars_set_boolean'] = {
    init: function () {
        this.appendValueInput("VALUE")
                .appendField("🇧 set")
                .appendField(new Blockly.FieldVariable("", null, ['bool'], 'bool'), "VAR_SET_BOOLEAN")
                .appendField("to")
                .setCheck(booleanCompatibility);
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};

Blockly.Blocks['vars_get_boolean'] = {
    init: function () {
        this.appendDummyInput()
                .appendField(new Blockly.FieldVariable("", null, ['bool'], 'bool'), "VAR_GET_BOOLEAN");
        this.setOutput(true, "Boolean");
        this.setStyle('variable_blocks');
        this.setTooltip("");
        this.setHelpUrl("");
    }
};