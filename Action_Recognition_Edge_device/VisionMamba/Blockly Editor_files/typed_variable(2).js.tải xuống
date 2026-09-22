/**
 * @license
 * Copyright 2020 Sébastien CANET
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @fileoverview Generating Arduino code for typed variable.
 * @author scanet@libreduc.cc (Sébastien CANET)
 */

'use strict';

goog.provide('Blockly.Arduino.VariablesTyped');

goog.require('Blockly.Arduino');


/*-----------------------------------------------------------------*/
const Arduino_DeclareVariable = function(varType, varName) {
    // workaround: avoid error in case this function is call before "Blockly.Arduino.init()" due to Async execution
    if (Blockly.Arduino.definitions_ === undefined) {
        return;
    }

    if (Blockly.Arduino.definitions_[varName] === undefined) {
        Blockly.Arduino.definitions_[varName] = varType + " " + varName + ";";  // auto declare if not available
    }
};


/*-----------------------------------------------------------------*/
// global
Blockly.Arduino['vars_initialize_global_auto'] = function (block) {
    var varType = 'auto';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_GLOBAL'), Blockly.Variables.NAME_TYPE);
    var varVal  = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';

    Blockly.Arduino.definitions_[varName] = varType + " " + varName + " = " + varVal + ";";
    return "";
};

Blockly.Arduino['vars_initialize_global'] = function (block) {
    var varType = block.getFieldValue("VAR_TYPE");
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_GLOBAL'), Blockly.Variables.NAME_TYPE);
    var varVal  = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';

    Blockly.Arduino.definitions_[varName] = varType + " " + varName + " = " + varVal + ";";
    return "";
};

Blockly.Arduino['vars_set_global'] = function (block) {
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_GLOBAL'), Blockly.Variables.NAME_TYPE);
    var varVal  = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';

    // identify variable type based on input block
    var inputType = block.inputList[0].connection.targetBlock().outputConnection.check_;
    var varType = "int";  // default type
         if (inputType.includes("float"))    { varType = "float";  }
    else if (inputType.includes("String"))   { varType = "String"; }
    else if (inputType.includes("Boolean"))  { varType = "bool";   }

    Arduino_DeclareVariable(varType, varName);
    return varName + ' = ' + varVal + ';\n';
};

Blockly.Arduino['vars_get_global'] = function (block) {
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_GLOBAL'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('int', varName);  // default type
    return [varName, Blockly.Arduino.ORDER_ATOMIC];
};

/*-----------------------------------------------------------------*/
// local
Blockly.Arduino['vars_initialize_local_auto'] = function (block) {
    var varType   = 'auto';
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName   = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_INT'), Blockly.Variables.NAME_TYPE);
    return varType + ' ' + varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_initialize_local'] = function (block) {
    var varType = block.getFieldValue("VAR_TYPE");
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_INT'), Blockly.Variables.NAME_TYPE);
    return varType + ' ' + varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_set_int_local'] = function (block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_INT'), Blockly.Variables.NAME_TYPE);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_get_int_local'] = function (block) {
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_INT'), Blockly.Variables.NAME_TYPE);
    return [varName, Blockly.Arduino.ORDER_ATOMIC];
};


/*-----------------------------------------------------------------*/
// int
Blockly.Arduino['vars_set_int'] = function (block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_INT'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('int', varName);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_get_int'] = function (block) {
    var code = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_INT'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('int', code);
    return [code, Blockly.Arduino.ORDER_ATOMIC];
};


/*-----------------------------------------------------------------*/
// float
Blockly.Arduino['vars_set_float'] = function (block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_FLOAT'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('float', varName);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_get_float'] = function (block) {
    var code = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_FLOAT'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('float', code);
    return [code, Blockly.Arduino.ORDER_ATOMIC];
};


/*-----------------------------------------------------------------*/
// String
Blockly.Arduino['vars_set_string'] = function (block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_STRING'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('String', varName);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_get_string'] = function (block) {
    var code = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_STRING'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('String', code);
    return [code, Blockly.Arduino.ORDER_ATOMIC];
};


/*-----------------------------------------------------------------*/
// bool
Blockly.Arduino['vars_set_boolean'] = function (block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_SET_BOOLEAN'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('bool', varName);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['vars_get_boolean'] = function (block) {
    var code = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR_GET_BOOLEAN'), Blockly.Variables.NAME_TYPE);
    Arduino_DeclareVariable('bool', code);
    return [code, Blockly.Arduino.ORDER_ATOMIC];
};


/*-----------------------------------------------------------------*/
Blockly.Arduino['variables_set'] = function(block) {
    var argument0 = Blockly.Arduino.valueToCode(block, 'VALUE', Blockly.Arduino.ORDER_ASSIGNMENT) || '0';
    var varName = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR'), Blockly.VARIABLE_CATEGORY_NAME);
    return varName + ' = ' + argument0 + ';\n';
};

Blockly.Arduino['variables_get'] = function(block) {
    var code = Blockly.Arduino.variableDB_.getName(block.getFieldValue('VAR'), Blockly.VARIABLE_CATEGORY_NAME);
    return [code, Blockly.Arduino.ORDER_ATOMIC];
};

