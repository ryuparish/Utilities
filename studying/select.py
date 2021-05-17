import sys
from random import randint
import random
from datetime import datetime

# Usage python3 select.py <# of ?'s> <begin> <end> <idc(0) ,odd(1), even(2)>
if len(sys.argv) != 5:
    print("Usage: select.py <# of ?'s> <begin> <end> <idc(0) ,odd(1), even(2)>")
    sys.exit(1)

def main():

    # Setting the random seed
    random.seed(datetime.now())
    even_odd = int(sys.argv[4])
    unique_questions = set()

    while (len(unique_questions) < int(sys.argv[1])):
        x = randint(int(sys.argv[2]), int(sys.argv[3]))
        if (even_odd > 0):
            
            # If odd and the number selected is not
            if (even_odd == 1 and (x % 2) != 1):
                while (x % 2 != 1):
                    x = randint(int(sys.argv[2]), int(sys.argv[3]))

            # Else if even is selected and number is not even
            elif (even_odd == 2 and (x % 2) != 0): 
                while (x % 2 != 0):
                    x = randint(int(sys.argv[2]), int(sys.argv[3]))
        unique_questions.add(x)

    print(sorted(unique_questions))

    return 0
main()
