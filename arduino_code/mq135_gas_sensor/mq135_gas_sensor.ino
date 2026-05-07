#define MQ135_PIN A0

float RLOAD = 10.0;     // kΩ — load resistor on the sensor board
float RZERO = 76.63;    // MUST calibrate in clean air (measure RS after 5-min warm-up & set RZERO = RS)

// ----------- Stable ADC -----------
int readADC() {
  long sum = 0;
  for (int i = 0; i < 50; i++) {
    sum += analogRead(MQ135_PIN);
    delay(5);
  }
  return sum / 50;
}

// ----------- Resistance -----------
float getResistance(int adc) {
  return ((1023.0 / adc) - 1.0) * RLOAD;
}

// ----------- Setup -----------
void setup() {
  Serial.begin(9600);
  Serial.println("Warming up sensor...");

  delay(300000); // 5-minute warm-up — DO NOT reduce (was 30000ms = only 30 sec, which is wrong)
}

// ----------- Loop -----------
void loop() {

  int adc = readADC();

  if (adc == 0) {
    Serial.println("Sensor error");
    delay(2000);
    return;
  }

  float rs = getResistance(adc);
  float ratio = rs / RZERO;

  // ----------- NH3 ppm calculation -----------
  float nh3 = 102.2 * pow(ratio, -2.473);

  // Clamp unrealistic values
  if (nh3 < 0) nh3 = 0;
  if (nh3 > 100) nh3 = 100;

  // ----------- Safety Classification -----------
  String status;

  if (nh3 < 5) status = "SAFE";
  else if (nh3 < 25) status = "CAUTION";
  else status = "UNSAFE";

  // ----------- Output -----------
  Serial.print("NH3: ");
  Serial.print(nh3);
  Serial.print(" ppm | Status: ");
  Serial.println(status);

  delay(2000);
}
