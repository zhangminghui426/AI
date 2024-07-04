import random
import string

def RStr(length=6):
    rstr = ''.join(random.sample(string.ascii_letters + string.digits, length))
    return rstr