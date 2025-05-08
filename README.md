#### Simple module to automatically compute gradient. 

Thank you for the inspiration. [Andrej Karpathy](https://github.com/karpathy/micrograd)

The code follows pretty much what Andrej does in his repo. 

A simple `Value` module that does some elementary computation and has the ability to calculate gradients when `backward` method is called. This is similar in style to how it is done in pytorch. 

There is also a simple Neural Network that uses the `Value` class. 
