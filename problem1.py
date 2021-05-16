import csv
import sys
import numpy as np


class Perceptron(object):

    def __init__(self, inputF, outputF, w1=0, w2=0, b=0, learnRate=1):
        self.inputF = inputF
        self.outputF = outputF
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.learnRate = learnRate

    def main(self):
        while True:
            weights = self.w1, self.w2, self.b

            # creating test set
            samples = []
            with open(self.inputF, "r") as read:

                fileRead = csv.reader(read)

                for line in fileRead:
                    samples.append(line)

            for sample in samples:
                res = self.predict(sample)
                if res != sample[2]:
                    self.train(res, sample)

            # writing output

            with open(self.outputF, "a", newline='') as data:
                fileWrite = csv.writer(data)
                text = self.w1, self.w2, self.b
                fileWrite.writerow(text)

            if weights[0] == self.w1 and weights[1] == self.w2 and weights[2] == self.b:
                return self.w1, self.w2, self.b

    def predict(self, inputs):

        v1 = self.w1 * int(inputs[0])
        v2 = self.w2 * int(inputs[1])

        activation = v1 + v2 + self.b

        if activation >= 0:
            return 1
        else:
            return -1

    def train(self, res, sample):

        self.w1 = np.round(self.w1 + self.learnRate * (int(sample[2]) - res) * int(sample[0]), 2)

        self.w2 = np.round(self.w2 + self.learnRate * (int(sample[2]) - res) * int(sample[1]), 2)

        self.b = np.round(self.b + self.learnRate * (int(sample[2]) - res), 2)


Test = Perceptron(sys.argv[1], sys.argv[2])
Test.main()
