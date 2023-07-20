from colorama import Fore, Style
from colorama import init as colorama_init

import examples

if __name__ == "__main__":
    colorama_init()

    # examples.simple_classifier.run()
    # examples.mnist_test.run()
    # examples.simple_classifier.run()
    # examples.spiral_test.run()
    # examples.conv2d_test.run()
    # examples.max_pool_test.run()
    examples.mnist_conv.run()

    print(f"\n\n{Fore.GREEN}This is {Fore.RED}done! 🏁🏁🏁{Style.RESET_ALL}!!!\n\n")
