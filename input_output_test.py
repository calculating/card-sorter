from gpiozero import Motor, Servo
import time

brr = Motor(27, 4, 22)
brr.forward(1)

time.sleep(3)
brr.stop()

bzz = Motor(23, 24, 25)
bzz.forward(1)

time.sleep(1)
bzz.stop()

trn = Servo(12, min_pulse_width = 0.0005, max_pulse_width = 0.0025)

for x in range(0, 20):
	trn.value = -1 + 0.1*x
	time.sleep(0.1) 
