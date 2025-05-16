//
    // Included Files
    //
    #include "F28x_Project.h"
    #include "math.h"
    
    // Prototype statements for functions found within this file.
    
    __interrupt void cpu_timer0_isr(void);
    
    void Gpio_select(void);
    
    void InitEPwm1Example(void);
    
    //
    //**Initialization
    
    #define pi 3.1415926
    
    #define pii 6.283185
    
    #define SPERIOD 2000
    
    
    
    #define w 2*50*3.1415926  // omegat
    #define EPWM_DBR   0x0002   //slower dead band
    
    #define EPWM_DBF   0x0002//1F40
    
    #define EPWM_DBRf   0x0002  //faster dead band
    
    #define EPWM_DBFf   0x0002
    
    
    
    float Sample;
    float ma;
    float a;
    float Vref;
    float alpha;
    int check = 0;
    float delta;
    float Carr;
    
    
    
    
    //
    // Main
    //
    void main(void)
    {
    //
    // Step 1. Initialize System Control:
    // PLL, WatchDog, enable Peripheral Clocks
    // This example function is found in the F2837xD_SysCtrl.c file.
    //
        InitSysCtrl();
    
    //
    // Step 2. Initialize GPIO:
    // This example function is found in the F2837xD_Gpio.c file and
    // illustrates how to set the GPIO to its default state.
    //
    
        InitGpio();
        GPIO_SetupPinMux(5, GPIO_MUX_CPU1, 0);
        GPIO_SetupPinOptions(5, GPIO_OUTPUT, GPIO_PUSHPULL);  // Set GPIO5 as SPWM output
    
        // For this example use the following configuration:
    
                    //Gpio_select();
    
    //
    // enable PWM1, PWM2 and PWM3
    //
        CpuSysRegs.PCLKCR2.bit.EPWM1=1;
    
    //
    // For this case just init GPIO pins for ePWM1, ePWM2, ePWM3
    // These functions are in the F2837xD_EPwm.c file
    //
        InitEPwm1Gpio();
    
    //
    // Step 3. Clear all interrupts and initialize PIE vector table:
    // Disable CPU interrupts
    //
        DINT;
    
    //
    // Initialize the PIE control registers to their default state.
    // The default state is all PIE interrupts disabled and flags
    // are cleared.
    // This function is found in the F2837xD_PieCtrl.c file.
    //
        InitPieCtrl();
    
    //
    // Disable CPU interrupts and clear all CPU interrupt flags:
    //
        IER = 0x0000;
        IFR = 0x0000;
    
    //
    // Initialize the PIE vector table with pointers to the shell Interrupt
    // Service Routines (ISR).
    // This will populate the entire table, even if the interrupt
    // is not used in this example.  This is useful for debug purposes.
    // The shell ISR routines are found in F2837xD_DefaultIsr.c.
    // This function is found in F2837xD_PieVect.c.
    //
        InitPieVectTable();
    
        // Interrupts that are used in this example are re-mapped to
    
        // ISR functions found within this file.
    
            EALLOW; // This is needed to write to EALLOW protected registers
    
            PieVectTable.TIMER0_INT = &cpu_timer0_isr;
    
            EDIS;   // This is needed to disable write to EALLOW protected registers
    
    //
    // Step 4. Initialize the Device Peripherals:
    //
            InitCpuTimers();   // For this example, only initialize the Cpu Timers
        EALLOW;
        CpuSysRegs.PCLKCR0.bit.TBCLKSYNC =0;
        EDIS;
    
        InitEPwm1Example();
    
        EALLOW;
        CpuSysRegs.PCLKCR0.bit.TBCLKSYNC =1;
        EDIS;
    
    #if (CPU_FRQ_200MHZ)
    
    // Configure CPU-Timer 0, 1, and 2 to interrupt every second:
    
    // 200MHz CPU Freq, 1 second Period (in uSeconds)
    
    
    
        ConfigCpuTimer(&CpuTimer0, 200, 2);//     //* 3.5uSec is kept as period
    
    #endif
    
    
    
       CpuTimer0Regs.TCR.all = 0x4001; // Use write-only instruction to set TSS bit = 0
    
    
    
    //
    // Step 5. User specific code, enable interrupts:
    // Initialize counters:
    //
       ma=0.8;
    
             alpha=0;
    
             Sample=1;
    
             delta= (2 * pi) / 10000;  // related to sample time::of 5uSec :: 20msec/5usec
    
             // Initial values
    
             // Enable CPU int1 which is connected to CPU-Timer 0, CPU int13
    
                    // which is connected to CPU-Timer 1, and CPU int 14, which is connected
    
                    // to CPU-Timer 2:
    
                       IER |= M_INT1;
    
    
                       // Enable TINT0 in the PIE: Group 1 interrupt 7
    
                                PieCtrlRegs.PIEIER1.bit.INTx7 = 1;
    
    
    
                                // Enable EPWM INTn in the PIE: Group 3 interrupt 1-3
    
                                   PieCtrlRegs.PIEIER3.bit.INTx1 = 1;
    
                                   // Enable global Interrupts and higher priority real-time debug events:
    
                                                   EINT;   // Enable Global interrupt INTM
    
                                                   ERTM;   // Enable Global realtime interrupt DBGM
    
    
    
    
    
                                                for(;;)
    
                                                   {
    
                                                       //common  run as many times i want..,
    
                                                     }
    
                                   }
    //*ISR
    
    
    
    __interrupt void cpu_timer0_isr(void)
    
       {
    
             CpuTimer0.InterruptCount++;
    
    
    
             if (alpha < pii)
    
                {
    
                     alpha = alpha+delta;
    
                     Sample++;
    
                }
    
             else
    
                {
    
                     a=Sample;
    
                     alpha=0;
    
                     Sample=1;
    
                }
    
    
    
                Vref = ma * sin(alpha);
    
    
                Vref=(1 + Vref) * SPERIOD * 0.5;
    
                Carr=EPwm1Regs.TBCTR;
    
    
    
                if(Vref >= SPERIOD*0.5){
                    if(Vref >= Carr)
                        GpioDataRegs.GPASET.bit.GPIO5 = 1;
                    else
                        GpioDataRegs.GPACLEAR.bit.GPIO5 = 1;
                }
    
                else if (Vref <= SPERIOD*0.5){
                    if(Vref <= Carr)
                        GpioDataRegs.GPASET.bit.GPIO5 = 1;
                    else
                        GpioDataRegs.GPACLEAR.bit.GPIO5 = 1;
                }
    
                /*check = check + 1;
                check = check%10000000;*/
    
                /*else{
                    GpioDataRegs.GPACLEAR.bit.GPIO5 = 1;
                }*/
    
    
    
                /*if (Level==1)
    
                {
    
                    GpioDataRegs.GPASET.bit.GPIO02 = 1; //S1
    
                    GpioDataRegs.GPACLEAR.bit.GPIO03 = 1;   //S2
    
                    GpioDataRegs.GPASET.bit.GPIO04 = 1; //S3
    
                    GpioDataRegs.GPASET.bit.GPIO05 = 1; //S7
    
                }*/
    
    
    
          // CpuTimer0.InterruptCount=CpuTimer0.InterruptCount+2;
    
           //EALLOW;
    
          // GpioDataRegs.GPASET.bit.GPIO8 = 0;
    
             // GpioDataRegs.GPASET.bit.GPIO8 = 1;
    
                     PieCtrlRegs.PIEACK.all = PIEACK_GROUP1;
    
          //EDIS;
    
    }
    
    
    
    //*Functions
    
    
    
    void InitEPwm1Example()  //Used for time clock
    
    {
    
    
    
       // Setup TBCLK
    
       EPwm1Regs.TBPRD = SPERIOD ;           // Set timer period 801 TBCLK
    
       EPwm1Regs.TBPHS.bit.TBPHS = 0x0000;           // Phase is 0
    
       EPwm1Regs.TBCTR = 0x0000;                      // Clear counter
    
    
    
       // Setup counter mode
    
       EPwm1Regs.TBCTL.bit.CTRMODE = TB_COUNT_UP; // Count up
    
       EPwm1Regs.TBCTL.bit.PHSEN = TB_DISABLE;        // Disable phase loading
    
       EPwm1Regs.TBCTL.bit.HSPCLKDIV = TB_DIV1;       // Clock ratio to SYSCLKOUT
    
       EPwm1Regs.TBCTL.bit.CLKDIV = TB_DIV1;
    
    
    
       // Setup shadowing
    
       EPwm1Regs.CMPCTL.bit.SHDWAMODE = CC_SHADOW;
    
       EPwm1Regs.CMPCTL.bit.SHDWBMODE = CC_SHADOW;
    
       EPwm1Regs.CMPCTL.bit.LOADAMODE = CC_CTR_ZERO;  // Load on Zero
    
       EPwm1Regs.CMPCTL.bit.LOADBMODE = CC_CTR_ZERO;
    
    
    
    
    
       // Set actions
    
       EPwm1Regs.AQCTLA.bit.CAU = AQ_SET;             // Set PWM1A on event A, up count
    
       EPwm1Regs.AQCTLA.bit.PRD = AQ_CLEAR;           // Clear PWM1A on event A, down count
    
    
    
      // EPwm1Regs.AQCTLB.bit.CAU = AQ_CLEAR;             // Set PWM1B on event B, up count
    
      // EPwm1Regs.AQCTLB.bit.CAD = AQ_SET;           // Clear PWM1B on event B, down count
    
    
    
       // Active Low PWMs - Setup Deadband
    
       EPwm1Regs.DBCTL.bit.OUT_MODE = DB_DISABLE;
    
       EPwm1Regs.DBCTL.bit.POLSEL = DB_ACTV_HIC;
    
       EPwm1Regs.DBCTL.bit.IN_MODE = DBA_ALL;
    
       EPwm1Regs.DBRED.all = EPWM_DBRf;
    
       EPwm1Regs.DBFED.all = EPWM_DBFf;
    
    }
    
    
    
    void Gpio_select(void)
    
    {
    
        EALLOW;
    
        GpioCtrlRegs.GPAMUX1.all = 0x00000003;  // Set as GPIO  4:15  Note: 0 & 1 are EPWM pins
    
    //    GpioCtrlRegs.GPAMUX2.all = 0x00000000;  // All GPIO  15:31
    
    //    GpioCtrlRegs.GPBMUX1.all = 0x00000000;  // All GPIO  32:47
    
    
    
        GpioCtrlRegs.GPADIR.all = 0xFFFFFFFC;   // All outputs  0:31
    
    //   GpioCtrlRegs.GPBDIR.all = 0x00001FFF;   // All outputs  32:63
    
        EDIS;
    
    
    
    }