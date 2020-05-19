"""
Moduł służący do śledzenia postępów sieci neuronowej zaimplementowanej w module netowrk2.py
"""

import src.mnist_loader as mnist_loader
import src.network2 as network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 20, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data=training_data, epochs=50, mini_batch_size=10, eta=0.1, lmbda=5.0,
        evaluation_data=list(validation_data)[:100], monitor_evaluation_accuracy=True)