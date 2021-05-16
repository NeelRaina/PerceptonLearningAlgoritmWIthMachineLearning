import csv
import sys
import numpy as np


class LinReg(object):

    def __init__(self, inputF, outputF, learnRate):
        self.inputF = inputF
        self.outputF = outputF
        self.learn = learnRate
        self.w = np.zeros(3)

    @staticmethod
    def dataprep(inputFile):
        info = np.loadtxt(inputFile, delimiter=',')
        vec1 = np.ones((info.shape[0], 1))

        info = np.concatenate((vec1, info), axis=1)

        info[:, 1] = LinReg.standardize(info[:, 1])
        info[:, 2] = LinReg.standardize(info[:, 2])
        return info

    def main(self):
        for time in range(100):
            data = LinReg.dataprep(self.inputF)
            a = data.shape[0]

            for r in data:
                predict = np.dot(r[:3], self.w)
                correct = r[3]
                for x in range(len(r) - 1):
                    self.w[x] -= self.learn / a * (predict - correct) * r[x]
        # output
        with open(self.outputF, "a", newline='') as info:
            fileWrite = csv.writer(info)

            result = self.learn, 100, round(self.w[0], 6), round(self.w[1], 6), round(self.w[2], 6)

            fileWrite.writerow(result)

    @staticmethod
    def standardize(data):
        standard = np.std(data)
        mean = np.mean(data)
        for r in range(data.shape[0]):
            data[r] = (data[r] - mean) / standard
        return data


inputF = sys.argv[1]
outputF = sys.argv[2]
learnRates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

for iteration in learnRates:
    Test = LinReg(inputF, outputF, iteration)
    Test.main()
