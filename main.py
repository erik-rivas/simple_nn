from colorama import Fore, Style
from colorama import init as colorama_init

import examples

if __name__ == "__main__":
    colorama_init()

    # simple_classifier.run()
    # mnist_test.run()
    examples.spiral_test.run()

    print(f"\n\n{Fore.GREEN}This is {Fore.RED}done! 🏁🏁🏁{Style.RESET_ALL}!!!\n\n")
