//
// Included Files
//
#include "F28x_Project.h"
#include <math.h>

//
// Global Variables
//
#define PWM_FREQ 50000      // Carrier wave frequency (50 kHz)
#define SINE_FREQ 50        // Sine wave frequency (50 Hz)
#define SAMPLE_RATE 500000  // Sampling frequency (500 kHz)
#define PI 3.14159265358979
#define TABLE_SIZE 10000     // Sine lookup table size

volatile float sineTable[TABLE_SIZE];
volatile double carrier = 2047;
volatile double carrierStep = 40.95;
volatile double sineWave = 0;
volatile int sineIndex = 0;
volatile uint16_t spwm_output = 0;

//
// Function Prototypes
//
//void generateSineTable(void);
__interrupt void cpu_timer0_isr(void);

//
// Main Function
//
void main(void)
{
    //
    // Step 1. Initialize System Control:
    //
    InitSysCtrl();

    //
    // Step 2. Initialize GPIO:
    //
    InitGpio();
    GPIO_SetupPinMux(5, GPIO_MUX_CPU1, 0);
    GPIO_SetupPinOptions(5, GPIO_OUTPUT, GPIO_PUSHPULL);  // Set GPIO5 as SPWM output

    //
    // Step 3. Initialize Interrupts:
    //
    DINT;
    InitPieCtrl();
    IER = 0x0000;
    IFR = 0x0000;
    InitPieVectTable();

    EALLOW;
    PieVectTable.TIMER0_INT = &cpu_timer0_isr;
    EDIS;

    //
    // Step 4. Initialize the Timer:
    //
    InitCpuTimers();
    ConfigCpuTimer(&CpuTimer0, 200, 2);  // 2 µs period (500 kHz sampling)
    CpuTimer0Regs.TCR.all = 0x4000;      // Enable Timer0

    IER |= M_INT1;   // Enable CPU INT1
    PieCtrlRegs.PIEIER1.bit.INTx7 = 1;   // Enable Timer0 interrupt

    //
    // Step 5. Generate Sine Lookup Table
    //
//    generateSineTable();

    //
    // Step 6. Enable Global Interrupts
    //
    EINT;
    ERTM;

    //
    // Infinite Loop
    //
    while(1);
}

//
// Generate 50 Hz Sine Lookup Table
//
//void generateSineTable(void)
//{
//    int i;
//    for (i = 0; i < TABLE_SIZE; i++)
//    {
//
//        sineTable[i] = (sinf(2 * PI * i / TABLE_SIZE)) * 2047;  // Scale to 0-4095
//    }
//}

//
// Timer0 ISR (Runs every 2 µs, Sampling the Carrier Wave)
//
__interrupt void cpu_timer0_isr(void)
{
    //
    // Generate Triangular Carrier Wave (Unipolar 25 kHz)
    //
    carrier += carrierStep;
    sineWave = 2047 + sinf(2 * PI * sineIndex / 10000 ) * 2047 ;
    if (carrier >= 2047*2) carrierStep = -40.95;
    if (carrier <= 0) carrierStep = 40.95;

    //
    // Compare with Sine Wave
    //
    if(sineWave>=2047)   //  positive half cycle
    {
        if (carrier <= sineWave)
            GpioDataRegs.GPASET.bit.GPIO5 = 1;
        else
            GpioDataRegs.GPACLEAR.bit.GPIO5 = 1;
    }
    else if(sineWave<2047)  //  negative half cycle
    {
        if (carrier >= sineWave)
            GpioDataRegs.GPASET.bit.GPIO5 = 1;
        else
            GpioDataRegs.GPACLEAR.bit.GPIO5 = 1;
    }

    //
    // Update GPIO5 with SPWM Output
    //
//    GpioDataRegs.GPASET.bit.GPIO5 = spwm_output;
//    GpioDataRegs.GPACLEAR.bit.GPIO5 = !spwm_output;

    //
    // Update Sine Table Index
    //
    sineIndex++;
    if (sineIndex >= 10000) sineIndex = 0;

    //
    // Acknowledge Interrupt
    //
    PieCtrlRegs.PIEACK.all = PIEACK_GROUP1;
}