#include <DHT.h>

// ===== CONFIG =====
#define DHTPIN 3       // Connect DATA pin to Arduino pin 3
#define DHTTYPE DHT11  // Change to DHT22 if needed

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  delay(2000);        // Allow sensor to stabilize
  dht.begin();
}

void loop() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  // Check if reading failed
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Error: Unable to read from DHT sensor");
    delay(2000);
    return;
  }

  // Print values (good for Serial Monitor)
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.print(" %\t");

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  delay(2000); // DHT needs delay
}