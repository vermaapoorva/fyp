import numpy as np
from math import pi
from fractions import Fraction

def rad_to_pifrac(rad, max_denominator=8000):
    pifrac = Fraction(rad / pi).limit_denominator(max_denominator)
    if pifrac == 0:
        return '0'
    num = {1: '', -1: '-'}.get(pifrac.numerator, str(pifrac.numerator))
    denom = '/{}'.format(pifrac.denominator) if pifrac.denominator != 1 else ''
    return 'pi'.join((num, denom))


# angle = -np.pi * 2
# change = -3 * np.pi/4
# print("angle: ", rad_to_pifrac(angle + change))
# print(2+(3/4))
# new_angle = (angle + change) % (2 * np.pi)
# print("new angle: ", rad_to_pifrac(new_angle))


angle = -2 * np.pi
goal_angle = np.pi/9

diff = abs(angle - goal_angle)
print("absolute diff", rad_to_pifrac(diff))
min_diff = min(diff, 2*np.pi - diff)
calc_diff = (diff+np.pi) % (2*np.pi) - np.pi

print("min diff", rad_to_pifrac(min_diff))
print("calc diff", rad_to_pifrac(calc_diff))

print("diff: ", diff)
print("min diff: ", min_diff)
print("calc diff: ", calc_diff)