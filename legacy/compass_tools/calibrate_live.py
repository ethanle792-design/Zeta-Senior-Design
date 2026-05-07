#!/usr/bin/env python3
import time
from smbus2 import SMBus

BUS=7
ADDR=0x2C
TRIG_REGS=[0x09,0x0A,0x0B]
TRIG_VAL=0x01

def s16(lo, hi):
    v=(hi<<8)|lo
    return v-65536 if v&0x8000 else v

def read_xyz(bus):
    for r in TRIG_REGS:
        bus.write_byte_data(ADDR, r, TRIG_VAL)
    time.sleep(0.03)
    b = bus.read_i2c_block_data(ADDR, 0x00, 7)
    x = s16(b[1], b[2])
    y = s16(b[3], b[4])
    z = s16(b[5], b[6])
    return x,y,z

with SMBus(BUS) as bus:
    print("Rotate 360° SLOWLY for 45 seconds, keep it FLAT, away from metal/Jetson.")
    xmin=ymin=zmin= 10**9
    xmax=ymax=zmax=-10**9

    t_end=time.time()+120
    while time.time()<t_end:
        x,y,z=read_xyz(bus)
        xmin,xmax=min(xmin,x),max(xmax,x)
        ymin,ymax=min(ymin,y),max(ymax,y)
        zmin,zmax=min(zmin,z),max(zmax,z)
        print(f"x={x:6d} y={y:6d} z={z:6d}   "
              f"x[{xmin},{xmax}] y[{ymin},{ymax}]", end="\r")

    print("\n\nResults:")
    x0=(xmax+xmin)/2.0
    y0=(ymax+ymin)/2.0
    sx=(xmax-xmin)/2.0
    sy=(ymax-ymin)/2.0
    avg=(sx+sy)/2.0
    print(f"X_OFFSET = {x0:.3f}")
    print(f"Y_OFFSET = {y0:.3f}")
    print(f"X_SCALE  = {avg/sx:.6f}")
    print(f"Y_SCALE  = {avg/sy:.6f}")
