#include <DHT.h>

// ===== MQ135 CONFIG =====
#define MQ135_PIN A0
float RLOAD = 10.0;     
float RZERO = 76.63;    // MUST calibrate

// ===== DHT CONFIG =====
#define DHTPIN 3
#define DHTTYPE DHT11   // change to DHT22 if needed
DHT dht(DHTPIN, DHTTYPE);

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

// ----------- Simple Temp/Humidity Compensation -----------
float getCorrectionFactor(float t, float h) {
  // Empirical approximation (not perfect, but better than nothing)
  return 1.0 + 0.02 * (t - 20.0) + 0.01 * (h - 65.0);
}

// ----------- Setup -----------
void setup() {
  Serial.begin(9600);
  Serial.println("Warming up sensors...");

  dht.begin();

  delay(30000); // 5 min warm-up (don’t reduce blindly)
}

// ----------- Loop -----------
void loop() {

  // ----- Read DHT -----
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("DHT read error");
    delay(2000);
    return;
  }

  // ----- Read MQ135 -----
  int adc = readADC();

  if (adc == 0) {
    Serial.println("MQ135 error");
    delay(2000);
    return;
  }

  float rs = getResistance(adc);

  // ----- Apply compensation -----
  float correction = getCorrectionFactor(temperature, humidity);
  float corrected_rs = rs / correction;

  float ratio = corrected_rs / RZERO;

  // ----- NH3 ppm (approx) -----
  float nh3 = 102.2 * pow(ratio, -2.473);

  // Clamp
  if (nh3 < 0) nh3 = 0;
  if (nh3 > 100) nh3 = 100;

  // ----- Safety classification -----
  String status;
  if (nh3 < 5) status = "SAFE";
  else if (nh3 < 25) status = "CAUTION";
  else status = "UNSAFE";

  // ----- Output (CSV-friendly for Python data logging) -----
  // Format: timestamp_ms, temp, humidity, nh3_ppm, status
  Serial.print(millis());       // ms since boot — use for temporal analysis
  Serial.print(",");
  Serial.print(temperature);
  Serial.print(",");
  Serial.print(humidity);
  Serial.print(",");
  Serial.print(nh3);
  Serial.print(",");
  Serial.println(status);       // SAFE / CAUTION / UNSAFE

  delay(2000);
}
