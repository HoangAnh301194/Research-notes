/**
 * @license
 * Copyright 2020 Sébastien CANET
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
'use strict';

goog.provide('Blockly.Msg.leanbot_vi');

goog.require('Blockly.Msg');


var LeanbotMsg_vi = {};

/*==================================================================================================
                                      Categories
==================================================================================================*/

LeanbotMsg_vi["Caregory"] = {
  Name       :    "LeanbotVN",
  Motion     :    "Di chuyển",
  Gripper    :    "Tay gắp",
  RGBLed     :    "Đèn Led RGB",
  Sound      :    "Phát nhạc",
  Sensors    :    "Cảm biến",
  IRLine     :    "Cảm biến vạch",
};


/*==================================================================================================
                                      Blocks
==================================================================================================*/

LeanbotMsg_vi["Warning"] = {
  OutOfRange             : "Ngoài khoảng cho phép",
};


LeanbotMsg_vi["Unit"] = {
  Second                 : "giây",
  Millisecond            : "mili giây",
  Centimeter             : "cm",
  Millimeter             : "mm",
  Degree                 : "độ",
  Hertz                  : "Hz",
  Kilohertz              : "kHz",
  StepPerSec             : "bước/giây",
  RevPerMin              : "vòng/phút",
};


LeanbotMsg_vi["Block"] = {

  "Leanbot.commentLine" : {
    message0 : "\u2800 %1",
    tooltip  : "Thêm mô tả",
    defaults : {
      commentString : "Mô tả",
    },
  },

  "Leanbot.section" : {
    message0 : "\u2800 %1 %2 %3",
    tooltip  : "Nhóm lệnh",
    defaults : {
      commentString : "Mô tả",
    },
  },

  "Leanbot.whenStarted" : {
    message0 : "Leanbot bắt đầu hoạt động",
    tooltip  : "Bắt đầu",
  },

  "Leanbot.delay" : {
    message0 : "Chờ %1 %2",
    tooltip  : "Keep running at current velocities for a given duration of time",
    text     : {"Running": "running", "Turning": "turning"},
  },


  /*---------------------------------------------------------------*/
  "LbMission.beginByTeacher_end" : {
    message0 : "Mission: Begin when %1 combo touched %2 %3 Mission: End",
    tooltip  : "Bắt đầu nhiệm vụ, thực hiện công việc và kết thúc",
  },

  "LbMission.beginByServer_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Bắt đầu nhiệm vụ, thực hiện công việc và kết thúc",
  },

  "LbMission.beginBySelect_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Bắt đầu nhiệm vụ, thực hiện công việc và kết thúc",
  },

  "LbMission.missionSelect" : {
    message0 : "%1 %2 %3",
    tooltip  : "Lựa chọn nhiệm vụ",
  },

  "LbMission.beginByName_end" : {
    message0 : "Mission: Begin Digital Twin %1 %2 %3 Mission: End",
    tooltip  : "Bắt đầu nhiệm vụ, thực hiện công việc và kết thúc",
  },

  "LbMission.beginImmediately_end" : {
    message0 : "Nhiệm vụ: Bắt đầu ngay lập tức %1 %2 Nhiệm vụ: Kết thúc",
    tooltip  : "Bắt đầu nhiệm vụ ngay lập tức, thực hiện công việc và kết thúc",
  },

  "LbMission.end" : {
    message0 : "Mission: End",
    tooltip  : "Kết thúc nhiệm vụ",
  },

  "LbMission.elapsedMillis" : {
    message0 : "Mission: elapsed Time in %1",
    tooltip  : "Tính khoảng thời gian từ lúc bắt đầu nhiệm vụ",
  },

  "Leanbot.DCMotor.setPower" : {
    message0 : "DC Motor: Run %1 at power level %2 %%",
    tooltip  : "Run DC Motor",
    text     : {"Forward": "forward", "Backward": "backward"},
  },


  /*---------------------------------------------------------------*/
  "LbMotion.run" : {
    message0 : "Di chuyển: %1 với vận tốc Trái %2 Phải %3 %4",
    tooltip  : "Cài đặt vận tốc cho 2 bánh xe",
    text     : {"Run": "Chạy", "Turn": "Rẽ"},
  },

  "LbMotion.waitDistanceMm" : {
    message0 : "Di chuyển: Chạy đoạn đường %1 %2",
    tooltip  : "Đợi tới khi Leanbot di chuyển được đoạn đường xấp xỉ",
  },

  "LbMotion.waitRotationDeg" : {
    message0 : "Di chuyển: Rẽ góc %1 %2",
    tooltip  : "Đợi tới khi Leanbot rẽ được góc xấp xỉ",
  },

  "LbMotion.wait" : {
    message0 : "Di chuyển: Tiếp tục %1 trong %2 %3",
    tooltip  : "Keep running at current velocities for a given duration of time",
    text     : {"Running": "chạy", "Turning": "rẽ"},
  },

  "LbMotion.stop" : {
    message0 : "Di chuyển: Dừng",
    tooltip  : "Giảm tốc độ và đợi tới khi Leanbot dừng hẳn",
  },

  "LbMotion.setZeroOrigin" : {
    message0 : "Di chuyển: Reset Distance and Heading",
    tooltip  : "Set current position as origin",
  },

  "LbMotion.approxDistance" : {
    message0 : "Di chuyển: approx. Distance in %1",
    tooltip  : "Get approximate distance from origin",
  },

  "LbMotion.approxHeading" : {
    message0 : "Di chuyển: approx. Heading in %1",
    tooltip  : "Get approximate heading from origin",
  },


  /*---------------------------------------------------------------*/
  "LbGripper.open": {
    message0 : "Gripper: Open",
    tooltip  : "Mở 2 cánh tay (di chuyển tới góc 0°)",
  },

  "LbGripper.close" : {
    message0 : "Gripper: Close",
    tooltip  : "Đóng 2 cánh tay (di chuyển tới góc 90°)",
  },

  "LbGripper.moveTo" : {
    message0 : "Gripper: Rotate to %1 %2",
    tooltip  : "Đưa 2 cánh tay tới một góc mong muốn",
  },

  "LbGripper.moveToLR" : {
    message0 : "Gripper: Rotate Left to %1 and Right to %2 %3 in %4 %5",
    tooltip  : "Đưa 2 cánh tay tới một góc mong muốn",
  },


  /*---------------------------------------------------------------*/
  "LbRGB.colour_picker" : {
    message0 : "Màu sắc: %1",
    tooltip  : "Chọn một màu sắc từ bảng màu",
  },

  "LbRGB.colour_random" : {
    message0 : "Màu sắc: Random",
    tooltip  : "Chọn một màu sắc ngẫu nhiên",
  },

  "LbRGB.colour_rgb" : {
    message0 : "Màu sắc: R %1 G %2 B %3",
    tooltip  : "Create a color with the specified amount of red, green, and blue. All values must be between 0 and 255",
  },

  "LbRGB.colour_blend" : {
    message0 : "Màu sắc: Blend %1 and %2 with ratio %3",
    tooltip  : "Blends two colours together with a given ratio [0.0, 1.0]",
  },

  "LbRGB.setColor" : {
    message0 : "RGB Leds: Set %1 to %2",
    tooltip  : "Cài đặt màu sắc cho ledX",
    warning  : "Missing color block",
  },

  "LbRGB.setColorText" : {
    message0 : "RGB Leds: Set %1 to %2",
    tooltip  : "Cài đặt màu sắc cho ledX",
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
    tooltip  : "Xóa toàn bộ đèn Led về màu đen",
  },

  "LbRGB.show" : {
    message0 : "RGB Leds: Display",
    tooltip  : "Hiển thị màu sắc ra đèn Led",
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
    tooltip  : "Kiểm tra xem nút cảm ứng có được chạm hay không",
  },

  "LbTouch.waitUntil" : {
    message0 : "Touch: wait until %1 touched",
    tooltip  : "Đợi tới khi nút cảm ứng được chạm",
  },

  "Leanbot.ping" : {
    message0 : "Ping: Front distance in %1",
    tooltip  : "Đo khoảng cách tới vật cản phía trước",
  },

  "Leanbot.objectInFront" : {
    message0 : "Ping: Object in front is within %1 %2 distance ?",
    tooltip  : "Kiểm tra xem có vật cản phía trước hay không",
  },


  /*---------------------------------------------------------------*/
  "LbIRLine.doManualCalibration" : {
    message0 : "IRLine: Do manual calibration with %1 button",
    tooltip  : "Do 3-step light level calibration with touchButtonX button",
  },

  "LbIRLine.setThreshold" : {
    message0 : "IRLine: Set threshold  2L %1 0L %2 1R %3 3R %4",
    tooltip  : "Cài đặt giá trị ngưỡng đen / trắng",
  },

  "LbIRLine.read" : {
    message0 : "IRLine: Read sensors",
    tooltip  : "Đọc cảm biến vạch",
  },

  "LbIRLine.value" : {
    message0 : "IRLine: Sensor values (2L, 0L, 1R, 3R)",
    tooltip  : "Lấy giá trị cảm biến vạch đã được đọc trước đó",
  },

  "LbIRLine.isBlackDetected" : {
    message0 : "IRLine: Black is detected ?",
    tooltip  : "Kiểm tra xem cảm biến có ở trên vạch hay không",
  },

  "LbIRLine.state" : {
    message0 : "IRLine: %1 %2 %3 %4",
    tooltip  : "Trạng thái của cảm biến vạch",
  },

  "LbIRLine.displayOnRGB" : {
    message0 : "IRLine: Display on RGB with %1",
    tooltip  : "Hiển thị trạng thái cảm biến vạch lên đèn Led RGB",
  },

};


/*==================================================================================================
                                      Definition
==================================================================================================*/

Blockly.Msg.Leanbot["vi"] = {
  Category      :      LeanbotMsg_vi["Caregory"],
  Warning       :      LeanbotMsg_vi["Warning"],
  Unit          :      LeanbotMsg_vi["Unit"],

  ...LeanbotMsg_vi["Block"],
};
