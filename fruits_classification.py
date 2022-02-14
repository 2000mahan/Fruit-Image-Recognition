import math

# Step 1: loading data_set
import random
import time
import timeit

from matplotlib import pyplot as plt

from Loading_Datasets import *


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def guess_the_fruit(m):
    maximum = -99999
    index = 0
    for i in range(len(m)):
        if m[i] > maximum:
            maximum = m[i]
            index = i
    return index


def vector_d(m):
    f = np.vectorize(sigmoid_d)
    return f(m)


def feedforward(weight0, weight1, weight2, bias0, bias1, bias2):
    true_guesses = 0
    for i in range(200):
        z1 = (weight0 @ train_set[i][0]) + bias0
        a1 = np.asarray([sigmoid(j) for j in z1]).reshape((150, 1))
        z2 = (weight1 @ a1) + bias1
        a2 = np.asarray([sigmoid(j) for j in z2]).reshape((60, 1))
        z3 = (weight2 @ a2) + bias2
        a3 = np.asarray([sigmoid(j) for j in z3]).reshape((4, 1))

        if guess_the_fruit(a3) == guess_the_fruit(train_set[i][1]):
            true_guesses = true_guesses + 1

    # Step 2: feedforward result Accuracy
    print("Accuracy:", (true_guesses / 200))


def backpropagation(weight0, weight1, weight2, bias0, bias1, bias2):
    l_r = 1
    epoch = 5
    batch_size = 10
    true_guesses = 0
    train = train_set[:200]
    costs = []
    start = time.time()
    for i in range(epoch):
        cost = 0
        random.shuffle(train)
        j = 0
        while j < 200:
            batch = train[j: j + 10]
            grad_w2 = np.zeros((4, 60))
            grad_w1 = np.zeros((60, 150))
            grad_w0 = np.zeros((150, 102))
            grad_a1 = np.zeros((150, 1))
            grad_a2 = np.zeros((60, 1))
            grad_a3 = np.zeros((4, 1))
            grad_b0 = np.zeros((150, 1))
            grad_b1 = np.zeros((60, 1))
            grad_b2 = np.zeros((4, 1))
            for data in batch:
                z1 = (weight0 @ data[0]) + bias0
                a1 = np.asarray([sigmoid(j) for j in z1]).reshape((150, 1))
                z2 = (weight1 @ a1) + bias1
                a2 = np.asarray([sigmoid(j) for j in z2]).reshape((60, 1))
                z3 = (weight2 @ a2) + bias2
                a3 = np.asarray([sigmoid(j) for j in z3]).reshape((4, 1))

                # derivatives
                for x in range(4):
                    for y in range(60):
                        grad_w2[x, y] += a2[y, 0] * sigmoid_d(z3[x, 0]) * (2 * a3[x, 0] - 2 * data[1][x, 0])

                for x in range(4):
                    grad_b2[x, 0] += sigmoid_d(z3[x, 0]) * (2 * a3[x, 0] - 2 * data[1][x, 0])

                for x in range(60):
                    for y in range(4):
                        grad_a2[x, 0] += weight2[y, x] * sigmoid_d(z3[y, 0]) * (2 * a3[y, 0] - 2 * data[1][y])

                for x in range(60):
                    for y in range(150):
                        grad_w1[x, y] += grad_a2[x, 0] * sigmoid_d(z2[x, 0]) * a1[y, 0]

                for x in range(60):
                    grad_b1[x, 0] += sigmoid_d(z2[x, 0]) * grad_a2[x, 0]

                for x in range(60):
                    for y in range(4):
                        grad_a1[x, 0] += weight1[y, x] * sigmoid_d(z2[y, 0]) * grad_a2[y, 0]

                for x in range(150):
                    for y in range(102):
                        grad_w0[x, y] += grad_a1[x, 0] * sigmoid_d(z1[x, 0]) * data[0][y]

                for x in range(150):
                    grad_b0[x, 0] += sigmoid_d(z1[x, 0]) * grad_a1[x, 0]

                # cost
                c = 0
                for x in range(4):
                    c += (data[1][x, 0] - a3[x, 0]) ** 2
                cost += c

            # update the w
            # update the b
            weight2 -= (grad_w2 / batch_size) * l_r
            weight1 -= (grad_w1 / batch_size) * l_r
            weight0 -= (grad_w0 / batch_size) * l_r
            bias2 -= (grad_b2 / batch_size) * l_r
            bias1 -= (grad_b1 / batch_size) * l_r
            bias0 -= (grad_b0 / batch_size) * l_r
            j += 10

        # print("average cost:", cost / 200)
        costs.append(cost / 200)

    for i in range(200):
        z1 = (weight0 @ train_set[i][0]) + bias0
        a1 = np.asarray([sigmoid(i) for i in z1]).reshape((150, 1))
        z2 = (weight1 @ a1) + bias1
        a2 = np.asarray([sigmoid(i) for i in z2]).reshape((60, 1))
        z3 = (weight2 @ a2) + bias2
        a3 = np.asarray([sigmoid(i) for i in z3]).reshape((4, 1))
        # print(guess_the_number(a3), guess_the_number(train_set[i][1]))
        if guess_the_fruit(a3) == guess_the_fruit(train_set[i][1]):
            true_guesses += 1
    stop = time.time()
    print("the accuracy of this model is:", true_guesses / 200)
    print("running time is: ", stop - start)
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, costs)
    plt.show()


def vectorization(weight0, weight1, weight2, bias0, bias1, bias2):
    l_r = 1
    epoch = 20
    batch_size = 5
    true_guesses = 0
    costs = []
    #d = train_set
    d = test_set
    start = time.time()
    for i in range(epoch):
        cost = 0
        random.shuffle(d)
        j = 0
        for t in range(0, 10):
            while j < 200:
                batch = d[j: j + 10]
                grad_w2 = np.zeros((4, 60))
                grad_w1 = np.zeros((60, 150))
                grad_w0 = np.zeros((150, 102))
                grad_a1 = np.zeros((150, 1))
                grad_a2 = np.zeros((60, 1))
                grad_a3 = np.zeros((4, 1))
                grad_b0 = np.zeros((150, 1))
                grad_b1 = np.zeros((60, 1))
                grad_b2 = np.zeros((4, 1))
                for data in batch:
                    z1 = (weight0 @ data[0]) + bias0
                    a1 = np.asarray([sigmoid(j) for j in z1]).reshape((150, 1))
                    z2 = (weight1 @ a1) + bias1
                    a2 = np.asarray([sigmoid(j) for j in z2]).reshape((60, 1))
                    z3 = (weight2 @ a2) + bias2
                    a3 = np.asarray([sigmoid(j) for j in z3]).reshape((4, 1))

                    grad_w2 += (2 * vector_d(z3) * (a3 - data[1])) @ (np.transpose(a2))
                    grad_b2 += (2 * vector_d(z3) * (a3 - data[1]))

                    grad_a2 += np.transpose(weight2) @ (2 * vector_d(z3) * (a3 - data[1]))
                    grad_b1 += vector_d(z2) * grad_a2
                    grad_w1 += (vector_d(z2) * grad_a2) @ np.transpose(a1)

                    grad_a1 += np.transpose(weight1) @ (vector_d(z2) * grad_a2)
                    grad_w0 += (vector_d(z1) * grad_a1) @ np.transpose(data[0])
                    grad_b0 += vector_d(z1) * grad_a1

                    # cost
                    c = 0
                    for x in range(4):
                        c += (data[1][x, 0] - a3[x, 0]) ** 2
                    cost += c

                # update the w
                # update the b
                weight2 -= (grad_w2 / batch_size) * l_r
                weight1 -= (grad_w1 / batch_size) * l_r
                weight0 -= (grad_w0 / batch_size) * l_r
                bias2 -= (grad_b2 / batch_size) * l_r
                bias1 -= (grad_b1 / batch_size) * l_r
                bias0 -= (grad_b0 / batch_size) * l_r
                j += 10

        # print("average cost:", cost / 200)
        costs.append(cost / 200)

    for i in range(200):
        z1 = (weight0 @ d[i][0]) + bias0
        a1 = np.asarray([sigmoid(i) for i in z1]).reshape((150, 1))
        z2 = (weight1 @ a1) + bias1
        a2 = np.asarray([sigmoid(i) for i in z2]).reshape((60, 1))
        z3 = (weight2 @ a2) + bias2
        a3 = np.asarray([sigmoid(i) for i in z3]).reshape((4, 1))
        # print(guess_the_number(a3), guess_the_number(train_set[i][1]))
        if guess_the_fruit(a3) == guess_the_fruit(train_set[i][1]):
            true_guesses += 1
    stop = time.time()
    print("the accuracy of this model is:", true_guesses / 200)
    print("running time is: ", stop - start)
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, costs)
    plt.show()


# creating the matrix of weights randomly and the biases with all zeros.
w0 = np.random.randn(150, 102)
w1 = np.random.randn(60, 150)
w2 = np.random.randn(4, 60)
b0 = np.zeros((150, 1))
b1 = np.zeros((60, 1))
b2 = np.zeros((4, 1))
# feedforward
#feedforward(w0, w1, w2, b0, b1, b2)
# backpropagation
#backpropagation(w0, w1, w2, b0, b1, b2)
# vectorization
vectorization(w0, w1, w2, b0, b1, b2)
