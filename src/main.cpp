/*
 * Accelerometer data send to Edge Impulse using the Data-Forwarder
 * Author: Christopher Mendez - 2022
 * Video tutorial: https://youtu.be/7KXUMV5muU8
 */

// /* Includes ---------------------------------------------------------------- */
// #include <Adafruit_MPU6050.h> // Click to download this library: http://librarymanager/All#Adafruit_MPU6050
// #include <Adafruit_Sensor.h>  // Click to download this library: https://github.com/adafruit/Adafruit_Sensor
// #include <Wire.h>

// /* Instances -------------------------------------------------------- */
// Adafruit_MPU6050 mpu;

// void setup(void)
// {
//     Serial.begin(115200);
//     while (!Serial)
//         delay(10); // wait for serial port

//     // Inicializar sensor!
//     if (!mpu.begin())
//     {
//         Serial.println("Error MPU6050!");
//         while (1)
//         {
//             delay(10);
//         }
//     }
//     Serial.println("MPU6050 found");

//     mpu.setAccelerometerRange(MPU6050_RANGE_2_G);

//     Serial.print("Accelerometer range set to: ");
//     switch (mpu.getAccelerometerRange())
//     {
//     case MPU6050_RANGE_2_G:
//         Serial.println("+-2G");
//         break;
//     case MPU6050_RANGE_4_G:
//         Serial.println("+-4G");
//         break;
//     case MPU6050_RANGE_8_G:
//         Serial.println("+-8G");
//         break;
//     case MPU6050_RANGE_16_G:
//         Serial.println("+-16G");
//         break;
//     }

//     mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

//     Serial.print("Filter bandwidth set to: ");
//     switch (mpu.getFilterBandwidth())
//     {
//     case MPU6050_BAND_260_HZ:
//         Serial.println("260 Hz");
//         break;
//     case MPU6050_BAND_184_HZ:
//         Serial.println("184 Hz");
//         break;
//     case MPU6050_BAND_94_HZ:
//         Serial.println("94 Hz");
//         break;
//     case MPU6050_BAND_44_HZ:
//         Serial.println("44 Hz");
//         break;
//     case MPU6050_BAND_21_HZ:
//         Serial.println("21 Hz");
//         break;
//     case MPU6050_BAND_10_HZ:
//         Serial.println("10 Hz");
//         break;
//     case MPU6050_BAND_5_HZ:
//         Serial.println("5 Hz");
//         break;
//     }

//     Serial.println("");
//     delay(100);
// }

// void loop()
// {

//     /* Gather sensor data */
//     sensors_event_t a, g, temp;
//     mpu.getEvent(&a, &g, &temp);

//     /* Print sensor values */
//     Serial.print(a.acceleration.x);
//     Serial.print(",");
//     Serial.print(a.acceleration.y);
//     Serial.print(",");
//     Serial.println(a.acceleration.z);
//     delay(5); // delay that determines output frequency > 100 hz at least
// }

/*
 * Air Quality detection analyzing vibrations from an Air Purifier.
 * Author: Christopher Mendez - 2022
 * Video tutorial: https://youtu.be/7KXUMV5muU8
 */

/* Includes ---------------------------------------------------------------- */
#include <AIOT_temperature-sensor_inferencing.h> //Edge Impulse Library with your trained model
#include <Adafruit_MPU6050.h>                    // Click to download this library: http://librarymanager/All#Adafruit_MPU6050
#include <Adafruit_Sensor.h>                     // Click to download this library: https://github.com/adafruit/Adafruit_Sensor
#include <Wire.h>

/* Constant defines -------------------------------------------------------- */

#define MAX_ACCEPTED_RANGE 2.0f // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

/* Instances -------------------------------------------------------- */
Adafruit_MPU6050 mpu;

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

/**
  @brief      Arduino setup
*/
void setup()
{

    Serial.begin(115200);
    while (!Serial)
        delay(10); // wait for serial port

    Serial.println("Condition of fan through vibration");

    if (!mpu.begin())
    {
        ei_printf("Error MPU6050!\r\n");
    }
    else
    {
        ei_printf("MPU6050 found\r\n");
    }

    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3)
    {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
        return;
    }
}

/**
 * @brief Return the sign of the number
 *
 * @param number
 * @return int 1 if positive (or 0) -1 if negative
 */
float ei_get_sign(float number)
{
    return (number >= 0.0) ? 1.0 : -1.0;
}

/**
 * @brief      Get data and run inferencing
 *
 * @param[in]  debug  Get debug info if true
 */
void loop()
{
    ei_printf("\nStarting inferencing in 5 seconds...\n");

    delay(5000);

    ei_printf("Sampling...\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3)
    {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        /* Gather sensor data */
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        buffer[ix] = a.acceleration.x;
        buffer[ix + 1] = a.acceleration.y;
        buffer[ix + 2] = a.acceleration.z;

        delayMicroseconds(next_tick - micros());
    }

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0)
    {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = {0};

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK)
    {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
              result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
    {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    }

    if (result.classification[0].value >= 0.6)
    {
        ei_printf("Damage\r\n");
    }
    if (result.classification[1].value >= 0.6)
    {
        ei_printf("Normal\r\n");
    }
    if (result.classification[2].value >= 0.6)
    {
        ei_printf("Silent\r\n");
    }
    if (result.classification[3].value >= 0.6)
    {
        ei_printf("Warning\r\n");
    }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
}