import sys
import random
import math

if sys.argv[1] == "uniform":
    x = random.uniform(float(sys.argv[2]), float(sys.argv[3]))
    print(('{:0'+sys.argv[4]+'.'+sys.argv[5]+'f}').format(x))
elif sys.argv[1] == "loguniform":
    x = math.exp(random.uniform(math.log(float(sys.argv[2])), math.log(float(sys.argv[3]))))
    print(('{:0'+sys.argv[4]+'.'+sys.argv[5]+'f}').format(x))
elif sys.argv[1] == "randint":
    x = random.randint(int(sys.argv[2]), int(sys.argv[3]))
    print(x)
elif sys.argv[1] == "choice":
    x = random.choice(sys.argv[2:])
    print(x)
