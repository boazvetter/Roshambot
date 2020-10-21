/*
Arduino script
Listens to serial communication (USB) and sets the servos accordingly

Author: Boaz Vetter
*/

 
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete
char rps = '\0'; // Initialize null

#include <Servo.h>
 
Servo servo1;  
Servo servo2;  
Servo servo3;  
Servo servo4;  
Servo servo5;  
 
int servoAngle = 0;   // servo position in degrees
 
void setup()
{
  Serial.begin(115200);  
  pinMode(LED_BUILTIN, OUTPUT);  
  servo1.attach(5);
  servo2.attach(6);
  servo3.attach(9);
  servo4.attach(10);
  servo5.attach(11);
}
 
 
void loop()
{
  char rx_byte;
  if (Serial.available() > 0) { //is a character available
    rx_byte = Serial.read();    //get the character                       
    if (rx_byte == 'r'){
//      digitalWrite(LED_BUILTIN, HIGH);
      servo1.write(180);
      servo2.write(180);
      servo3.write(180);
      servo4.write(180);
      servo5.write(180);
      Serial.println("Move: Rock");
    }    
    else if (rx_byte == 'p'){
//      digitalWrite(LED_BUILTIN, LOW);
      servo1.write(0);
      servo2.write(40);
      servo3.write(0);
      servo4.write(0);
      servo5.write(0);
      Serial.println("Move: Paper");
    }        
    else if (rx_byte == 's'){
//      digitalWrite(LED_BUILTIN, LOW);
      servo1.write(0);
      servo2.write(40);
      servo3.write(180);
      servo4.write(180);
      servo5.write(180);
      Serial.println("Move: Scissors");
    }    
  }
}
